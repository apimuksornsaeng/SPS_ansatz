import psutil, os, numpy as np
def mem_MB():
    return process.memory_info().rss / 1024**2

import numpy as np
import torch
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.models.spins import SpinChain, SpinModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite, SpinSite
from tenpy.models.lattice import Chain, Lattice
from tenpy.models.model import CouplingModel, CouplingMPOModel
from torch.nn.parallel import DistributedDataParallel as DDP
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import dmrg
import json
import ast
import matplotlib.pyplot as plt
import time
import os
import threading

class PeakRSS:
    """Context manager that samples RSS in a background thread and records the peak (in MB)."""
    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._peak_bytes = 0
        self._thr = None
        self._proc = psutil.Process(os.getpid())

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024**2)

    def _sample(self):
        # Lower priority sampling loop
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak_bytes:
                    self._peak_bytes = rss
            except Exception:
                pass
            # Sleep a bit; tune if your allocations are extremely bursty
            time.sleep(self.interval_s)

    def __enter__(self):
        self._stop.clear()
        self._peak_bytes = 0
        self._thr = threading.Thread(target=self._sample, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thr is not None:
            self._thr.join()

class LongRangeSpinModel(CouplingMPOModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        
    def init_sites(self, model_params):
        # Define the local Hilbert space
        site = SpinHalfSite(conserve=None)
        return site

    def init_lattice(self, model_params):
        # Initialize a 1D chain lattice
        from tenpy.models.lattice import Chain
        L = model_params['L']
        site = SpinHalfSite(conserve=None)
        return Chain(L, site)

    def init_terms(self, model_params):
        # Define the Hamiltonian terms
        Jz = model_params.get('Jz', 1.0)
        hx = model_params.get('hx', 0.)
        hz = model_params.get('hz', 0.)
        alpha = model_params.get('alpha', 1.0)
        L = self.lat.N_sites

        # Generate all-to-all coupling pairs
        pairs_all_to_all = []
        for dx in range(1, L):
            pairs_all_to_all.append((0, 0, [dx]))
        self.lat.pairs['all_to_all'] = pairs_all_to_all
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hz, u, 'Sz')
        # Add spin-spin couplings
        if model_params['connection_type'] == 'longrange':
            if model_params['dim'] == '1D':
                for u1, u2, dx in self.lat.pairs['all_to_all']:
                    r = abs(dx[0])
                    J_eff = Jz / r**alpha
                    self.add_coupling(J_eff, u1, 'Sz', u2, 'Sz', dx)
            elif model_params['dim'] == '3D':
                for pair in model_params['graph']:
                    i, j = pair
                    dist = snake_distance(i, j, model_params['Lx'], model_params['Ly'], model_params['Lz'])
                    print(f'Pair ({i},{j}): Dist = {dist}')
                    self.add_coupling_term(Jz/dist**alpha, i, j, 'Sz', 'Sz')
        elif model_params['connection_type'] != 'longrange':
            for pair in model_params['graph']:
                i, j = pair
                self.add_coupling_term(Jz, i, j, 'Sz', 'Sz')

def snake_index_to_xyz(i, Lx, Ly, Lz):
    """
    Convert snake index i (0-based) to (x,y,z) for a 3D lattice (Lx, Ly, Lz)
    """
    i = torch.as_tensor(i, dtype=torch.long)
    layer_size = Lx * Ly
    z = i // layer_size
    r = i % layer_size

    # y order flips depending on z parity
    y_linear = r // Lx
    y = torch.where((z % 2) == 0, y_linear, (Ly - 1 - y_linear))

    # x direction flips depending on y parity (actual y after the z-flip)
    x_in_row = r % Lx
    x = torch.where((y % 2) == 0, x_in_row, (Lx - 1 - x_in_row))

    return x, y, z


def xyz_to_snake_index(x, y, z, Lx, Ly, Lz):
    """
    Inverse of snake_index_to_xyz.
    """
    x = torch.as_tensor(x, dtype=torch.long)
    y = torch.as_tensor(y, dtype=torch.long)
    z = torch.as_tensor(z, dtype=torch.long)

    # Recover the linear row index (before z-based y-flip)
    y_linear = torch.where((z % 2) == 0, y, (Ly - 1 - y))

    # Recover x within the row (before y-based x-flip)
    x_in_row = torch.where((y % 2) == 0, x, (Lx - 1 - x))

    r = y_linear * Lx + x_in_row
    i = z * (Lx * Ly) + r
    return i


def snake_distance(i, j, Lx, Ly, Lz, spacing=(1.0, 1.0, 1.0)):
    """
    Euclidean distance between site(s) i and j given in 3D snake indexing.
    """
    ax, ay, az = spacing
    xi, yi, zi = snake_index_to_xyz(i, Lx, Ly, Lz)
    xj, yj, zj = snake_index_to_xyz(j, Lx, Ly, Lz)

    dx = (xi - xj).to(torch.float32) * ax
    dy = (yi - yj).to(torch.float32) * ay
    dz = (zi - zj).to(torch.float32) * az

    return torch.sqrt(dx*dx + dy*dy + dz*dz).item()


def GS_TIM_MPS(system_param):
    model_params = {
    'dim': system_param['dim'],
    'Jx': 0 ,'Jy': 0 , 'Jz': system_param['Jz'] * 4 , 'hx': system_param['hx'] * 2, 'hz': system_param['hz'] * 2, 'alpha': system_param['alpha'],
    'L': system_param['num_site'],
    'bc_MPS': 'finite', 'connection_type': system_param['lattice'], 'graph': system_param['graph']
    }
    if system_param['dim'] == '3D':
        model_params['Lx'] = system_param['Lx']
        model_params['Ly'] = system_param['Ly']
        model_params['Lz'] = system_param['Lz']
    chi = system_param['chi']
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'max_E_err': 1e-8,
        'max_S_err': 0.000001,
        'max_trunc_err': None,
        'max_sweeps': 10000,
        'N_sweeps_check': 5,
        'mixer': True,
        'verbose': 1
    }
    if model_params['connection_type'] == 'random':
        dmrg_params['max_trunc_err'] = 0.01
        dmrg_params['max_sweeps'] = 1000000
    elif model_params['connection_type'] == 'longrange':
        if model_params['alpha'] >= 1:
            dmrg_params['max_trunc_err'] = 0.01
            dmrg_params['max_sweeps'] = 1000000
    if system_param['lattice'] == 'longrange' or system_param['lattice'] == 'random':
        model_params['graph'] = system_param['graph']
        model = LongRangeSpinModel(model_params)
    elif system_param['dim'] == '2D' or system_param['dim'] == '3D':
        model_params['graph'] = system_param['graph']
        model = LongRangeSpinModel(model_params)
    elif system_param['lattice'] == 'square' and system_param['dim'] == '1D':
        model = SpinModel(model_params)
    hamiltonian_mpo = model.calc_H_MPO()
    print(hamiltonian_mpo._W)

    if system_param['Jz'] > 0 and system_param['hz'] < 1.0:
        product_state1 = ["up", "down"] * (model_params["L"] // 2)  # this selects a charge sector!
        psi_DMRG1 = MPS.from_product_state(model.lat.mps_sites(), product_state1)
        psi_DMRG = psi_DMRG1.copy()
    else:
        product_state = ["up"] * (model_params['L'])
        psi_DMRG = MPS.from_product_state(model.lat.mps_sites(), product_state)
    
    engine = dmrg.TwoSiteDMRGEngine(psi_DMRG, model, dmrg_params)
    GSE_DMRG, psi_DMRG = engine.run()

    return GSE_DMRG/model_params['L'], psi_DMRG

def GS_2DTFIM_MPS(system_param):
    model_params = {
    'dim': system_param['dim'], 
    'Jx': 0 ,'Jy': 0 ,'Jz': system_param['Jz'] * 4 , 'hx': system_param['hx'] * 2, 'hz': system_param['hz'] * 2,
    'lattice' : 'Square',
    'Lx': system_param['Lx'], 'Ly': system_param['Ly'],
    'bc_x': 'open', 'bc_y': 'ladder',
    'bc_MPS': 'finite'
    #'conserve': 'parity'
    }
    num_site = system_param['Lx'] * system_param['Ly']
    chi = system_param['chi']
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'max_E_err': 1e-8,
        'max_S_err': 0.000001,
        'max_trunc_err': None,
        'max_sweeps': 10000,
        'mixer': True,
        'verbose': 1,
        'max_trunc_err': None,
    }
    model = SpinModel(model_params)
    if system_param['Jz'] == 1.0 and np.abs(system_param['hx']) < 1:
        sites = model.lat.mps_sites()        # list of Site objects (length N_sites)
        init_state = ["up" if i % 2 == 0 else "down" for i in range(len(sites))]
        psi_DMRG = MPS.from_product_state(sites, init_state, bc="finite")

    else:
        psi_DMRG = MPS.from_lat_product_state(model.lat, [[['up']] * (model_params['Lx'])])
    engine = dmrg.TwoSiteDMRGEngine(psi_DMRG, model, dmrg_params)
    GSE_DMRG, psi_DMRG = engine.run()
    return GSE_DMRG/num_site, psi_DMRG

def coupling_indices(system_param):
    sys_dim = system_param['dim']
    if sys_dim == '1D':
        if system_param['lattice'] == 'square':
            if system_param['bc'] == 'finite':
                Lx = system_param['Lx']
                return torch.arange(Lx - 1).unsqueeze(-1) + torch.tensor([0, 1])
            elif system_param['bc'] == 'periodic':
                Lx = system_param['Lx']
                return (torch.arange(Lx).unsqueeze(-1) + torch.tensor([0, 1])) % Lx
        elif system_param['lattice'] == 'longrange':
            Lx = system_param['Lx']
            neighbors = []
            for i in range(Lx-1):
                for j in range(i+1 , Lx):
                   neighbors.append([i,j])
            neighbors = torch.tensor(neighbors)
            return neighbors
        elif system_param['lattice'] == 'random':
            Lx = system_param['Lx']
            p = system_param['alpha']
            neighbors = []
            np.random.seed(5)
            for i in range(Lx-1):
                for j in range(i+1 , Lx):
                    if np.random.rand() < p:
                        neighbors.append([i,j])
            neighbors = torch.tensor(neighbors)
            return neighbors
    elif sys_dim == '2D':
        if system_param['lattice'] == 'square':
            Lx = system_param['Lx']
            Ly = system_param['Ly']
            neighbors = []

            def snake_label(x, y):
                if x % 2 == 0:
                    return x * Ly + y
                else:
                    return x * Ly + (Ly - 1 - y)

            for x in range(Lx):
                for y in range(Ly):
                    current_label = snake_label(x, y)

                    if x + 1 < Lx:
                        neighbor = snake_label(x + 1, y)
                        neighbors.append(sorted([current_label, neighbor]))
                    if y + 1 < Ly:
                        neighbor = snake_label(x, y + 1)
                        neighbors.append(sorted([current_label, neighbor]))

            neighbors = torch.tensor(neighbors)
            return neighbors
        elif system_param['lattice'] == 'triangle':
            Lx = system_param['Lx']
            Ly = system_param['Ly']
            pairs = []
            def get_snake_index(x, y, Lx):
                if y % 2 == 0:
                    return y * Lx + x
                else:
                    return y * Lx + (Lx - 1 - x)
            for y in range(Ly):
                for x in range(Lx):
                    i = get_snake_index(x, y, Lx)

                    # Right neighbor
                    if x + 1 < Lx:
                        j = get_snake_index(x + 1, y, Lx)
                        pairs.append(sorted([i, j]))

                    # Down neighbor
                    if y + 1 < Ly:
                        j = get_snake_index(x, y + 1, Lx)
                        pairs.append(sorted([i, j]))

                    # Diagonal down-right (regardless of even/odd row)
                    if x + 1 < Lx and y + 1 < Ly:
                        j = get_snake_index(x + 1, y + 1, Lx)
                        pairs.append(sorted([i, j]))
            pairs = torch.tensor(pairs)
            return pairs
    elif sys_dim == '3D':
        if system_param['lattice'] == 'square':
            Lx = system_param['Lx']
            Ly = system_param['Ly']
            Lz = system_param['Lz']
            neighbors = []

            # Define the snake-style label
            def snake_label(x, y, z):
                if x % 2 == 0:
                    if y % 2 == 0:
                        return x * (Ly * Lz) + y * Lz + z
                    else:
                        return x * (Ly * Lz) + y * Lz + (Lz - 1 - z)
                else:
                    if y % 2 == 0:
                        return x * (Ly * Lz) + (Ly - 1 - y) * Lz + z
                    else:
                        return x * (Ly * Lz) + (Ly - 1 - y) * Lz + (Lz - 1 - z)

            for x in range(Lx):
                for y in range(Ly):
                    for z in range(Lz):
                        current_label = snake_label(x, y, z)

                        # Check neighbors in positive directions (according to snake logic)
                        if x + 1 < Lx:
                            neighbors.append(sorted([current_label, snake_label(x + 1, y, z)]))
                        if y + 1 < Ly:
                            neighbors.append(sorted([current_label, snake_label(x, y + 1, z)]))
                        if z + 1 < Lz:
                            neighbors.append(sorted([current_label, snake_label(x, y, z + 1)]))
            neighbors = torch.tensor(neighbors)
            return neighbors
        elif system_param['lattice'] == 'longrange':
            Lx = system_param['Lx']
            Ly = system_param['Ly']
            Lz = system_param['Lz']
            num_site = Lx * Ly * Lz
            neighbors = []
            for i in range(num_site-1):
                for j in range(i+1 , num_site):
                   neighbors.append([i,j])
            neighbors = torch.tensor(neighbors)
            return neighbors

def Pauli(s: str, opt_method: dict):
    datatype = opt_method['dtype']
    device = opt_method['device']
    if s == 'X':
        return torch.tensor([[0,1],[1,0]], dtype=datatype, device=device)
    elif s == 'Z':
        return torch.tensor([[1,0],[0,-1]], dtype=datatype, device=device)
    elif s == 'I':
        return torch.eye(2, dtype=datatype, device=device)

def kron_list(ops):
    """Compute the Kronecker product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = torch.kron(result, op)
    return result

def Tilted_Ising_Hamiltonian_tensor(system_param, opt_method):
    num_site = system_param['num_site']
    datatype = opt_method['dtype']
    device = opt_method['device']
    alpha = system_param['alpha']
    I = Pauli('I', opt_method)
    sigma_x = Pauli('X', opt_method)
    sigma_z = Pauli('Z', opt_method)
    dim = 2 ** num_site
    H = torch.zeros((dim, dim), dtype=datatype, device=device)
    for (i, j) in system_param['graph']:
        # Create a list of operators with identity at all sites.
        ops = [I.clone() for _ in range(num_site)]
        # Place sigma_z at positions i and j.
        ops[i] = sigma_z
        ops[j] = sigma_z
        # Compute the Kronecker product for this term.
        if system_param['lattice'] == 'longrange':
            coupling_J = system_param['Jz']/abs(i-j)**alpha
        else:
            coupling_J = system_param['Jz']
        H += coupling_J * kron_list(ops)
    
    # Add the transverse field term: -h * sum_i sigma_x^i.
    for i in range(num_site):
        ops = [I.clone() for _ in range(num_site)]
        ops[i] = sigma_x
        H += -system_param['hx'] * kron_list(ops)
        ops[i] = sigma_z
        H += -system_param['hz'] * kron_list(ops)
    return H

def exact_diagonalization(system_param, opt_method):
    num_site = system_param['num_site']
    Hamiltonian = Tilted_Ising_Hamiltonian_tensor(system_param, opt_method)
    l, v = torch.linalg.eigh(Hamiltonian)
    l = l/num_site
    print(f'First four eigenenergies: {l[0].item(),l[1].item(),l[2].item(),l[3].item()}')
    gs_energy_exact = torch.real(l[0])
    print(gs_energy_exact.dtype)
    gs_exact = v.T[0].clone().detach()
    return {'energy': gs_energy_exact, 'state': gs_exact}

def reference_energy(system_param, opt_method):
    num_site = system_param['num_site']
    if system_param['dim'] == '1D':
        time_exact_start = time.perf_counter()
        if num_site <= 14:
            exact_daig = exact_diagonalization(system_param, opt_method)
            gs_energy_exact = exact_daig['energy']
            gs_exact = exact_daig['state']
        else:
            gs_energy_exact, gs_exact = GS_TIM_MPS(system_param)
    elif system_param['dim'] == '2D':
        if system_param['lattice'] == 'square':
            param_tilted_ising = [system_param['Jz'], system_param['hx'], system_param['hz']]
            print(param_tilted_ising)
            time_exact_start = time.perf_counter()
            gs_energy_exact, gs_exact = GS_2DTFIM_MPS(system_param)
        elif system_param['lattice'] == 'triangle':
            gs_energy_exact, gs_exact = GS_TIM_MPS(system_param)
    elif system_param['dim'] == '3D':
        gs_energy_exact, gs_exact = GS_TIM_MPS(system_param)
    return gs_energy_exact.item(), gs_exact


lattice = 'longrange'
system_param = {
    'dim': '3D',
    'lattice': lattice, #['square','longrange','random','triangle'],
    'bc': 'finite', #['finite','periodic']
    'Lx': 4, #int(os.environ['L']),
    'Ly': 4, #int(np.sqrt(num_site)),
    'Lz': 4,
    'Jz': -1.0,
    'hx': 7.0,
    'hz': 0.25
}

if system_param['dim'] == '1D':
    num_site = system_param['Lx']
elif system_param['dim'] == '2D':
    num_site = system_param['Lx'] * system_param['Ly']
elif system_param['dim'] == '3D':
    num_site = system_param['Lx'] * system_param['Ly'] * system_param['Lz']

system_param['num_site'] = num_site

parameters = [system_param['Jz'],system_param['hx'],system_param['hz']]

use_gpu = True
do_parallel = True
if use_gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f'We use {device}')

opt_method = {
        'type': None,
        'opt_mode': 'energy', #['supervise', 'energy']
        'device': device,
        'dtype': torch.float32,
        'SPS_type': 'Real',
        'parallel': do_parallel,
        'method': 'vmap',
        'epochs': 50000,
    }
χ_range = [2**i for i in range(11)]
α_range = [0,0.5,1.0,2.0] #Also the range for connection probability in random lattice

if __name__ == "__main__":
    file_name = f'dmrg/dmrg_{lattice}_{parameters}_{num_site}'
    if os.path.exists(file_name):
        # Load existing data
        with open(file_name, 'r') as file:
            energy_list_raw = json.load(file)
            energy_list = {ast.literal_eval(k):v for k,v in energy_list_raw.items()}
            file.close()
    else:
        energy_list = {}
    for α in α_range:
        system_param['alpha'] = α
        system_param['graph'] = coupling_indices(system_param)
        print(system_param['graph'])
        for χ in χ_range:
            pid = os.getpid()
            process = psutil.Process(pid)
            system_param['chi'] = χ
            mem_before = process.memory_info().rss / 1024**2
            time_start = time.time()
            with PeakRSS(interval_s=0.01) as mon:
                gs_energy_exact, gs_exact = reference_energy(system_param, opt_method)
            peak_mb = mon.peak_mb
            time_end = time.time()
            mem_after = process.memory_info().rss / 1024**2  # MB
            mem_cpu_used = mem_after - mem_before
            print(f'memory used: {mem_cpu_used:.2f} MB')
            bond_dim = max(gs_exact.chi)
            print(f'Bond dimension: {bond_dim}')
            time_exc = time_end - time_start
            energy_list.update({(α,bond_dim,time_exc,mem_cpu_used,peak_mb): gs_energy_exact})
            print(f'α: {α}, χ: {χ}, energy: {gs_energy_exact}')

    energy_list_json = {str(k): v for k, v in energy_list.items()}
    Lx = system_param['Lx']
    Ly = system_param['Ly']
    with open(file_name,'w') as f:
        json.dump(energy_list_json, f, indent=4)