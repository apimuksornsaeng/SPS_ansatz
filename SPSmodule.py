import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from multiprocessing import Manager
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import time
import csv
import tracemalloc
import tenpy.linalg.np_conserved as npc
from tenpy.models.spins import SpinChain, SpinModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Chain, Lattice
from tenpy.models.model import CouplingModel, CouplingMPOModel
from torch.nn.parallel import DistributedDataParallel as DDP
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import dmrg
import gc
import scipy
from torch.cuda.amp import GradScaler, autocast
from functorch import vmap
import os
import csv
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import logging
import psutil
import h5py
import socket

class SPS_ddp(nn.Module):
    def __init__(self, system_param: dict, opt_method: dict):
        super(SPS_ddp, self).__init__()
        self.system_param = system_param
        self.opt_method = opt_method
        if system_param['dim'] == '1D':
            self.num_site = system_param['Lx']
        elif system_param['dim'] == '2D':
            self.num_site = system_param['Lx'] * system_param['Ly']
        elif system_param['dim'] == '3D':
            self.num_site = system_param['Lx'] * system_param['Ly'] * system_param['Lz']
        self.M = system_param['M']
        self.K = system_param['K']
        self.M_old = system_param['M']
        self.Jz = system_param['Jz']
        self.hx = system_param['hx']
        self.hz = system_param['hz']
        self.datatype = opt_method['dtype']
        self.device = opt_method['device']
        self.SPS_type = opt_method['SPS_type']
        self.method = opt_method['method']
        self.lattice = system_param['lattice']
        self.graph = system_param['graph']
        self.dist = system_param['dist']
        self.opt_mode = opt_method['opt_mode']
        self.α = system_param['alpha']
        if self.opt_mode == 'supervise':
            self.gs_exact = opt_method['gs_exact']
        if self.SPS_type == 'typical':
            print(f'We use typical')
            self.theta = nn.Parameter(torch.DoubleTensor(self.M,self.num_site).uniform_(-torch.pi/2, torch.pi/2).to(self.device))
            self.coef = nn.Parameter(torch.DoubleTensor(self.K * self.M).uniform_(-1, 1).to(self.device))

        elif self.SPS_type == 'max_entropy':
            L = self.num_site//2
            if self.M > 1:
                required_bits = (self.M - 1).bit_length()
            else:
                required_bits = 1
                
            if self.num_site < required_bits:
                raise ValueError(
                    f"L/2={L} is not large enough to represent all numbers up to M-1={self.M-1}. "
                    f"At least L={required_bits} bits are required."
                )
            m_values = torch.arange(self.M).unsqueeze(1)
            bit_shifts = torch.arange(L - 1, -1, -1)
            theta_base = (m_values >> bit_shifts) & 1 
            theta = torch.cat([theta_base, theta_base], dim=1) * torch.pi / 2
            self.theta = nn.Parameter(theta).to(self.datatype).to(self.device)
            self.coef = nn.Parameter(torch.ones(self.M, dtype = self.datatype, device = self.device))
        elif self.SPS_type == 'plus':
            self.theta = nn.Parameter(torch.DoubleTensor(self.M,self.num_site).uniform_(-1, 1) * 0.01)
            self.coef = nn.Parameter(1 + torch.DoubleTensor(self.M).uniform_(-1, 1) * 0.01)
        elif self.SPS_type == 'paramagnetic':
            print(f'We use paramagnetic')
            self.theta = nn.Parameter(np.arctan(self.hx/self.hz) + torch.DoubleTensor(self.M,self.num_site).uniform_(-1, 1).to(self.device) * 0.01)
            self.coef = nn.Parameter(1 + torch.DoubleTensor(self.K * self.M).uniform_(-1, 1).to(self.device) * 0.01)

        
        if self.SPS_type == 'Complex':
            self.ϕ = nn.Parameter(torch.DoubleTensor(self.M,self.num_site).uniform_(-torch.pi/2, torch.pi/2))
            self.phase = nn.Parameter(torch.DoubleTensor(self.M).uniform_(-1, 1))
        
    
    def update_opt_mode(self,value):
        self.opt_mode = value

    def check_c(self):
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        threshold = 1e-4
        print(f'Max |C|: {max(abs(self.coef))}')
        print(f'Min |C|: {min(abs(self.coef))}')
        with torch.no_grad():
            local_mask = (abs(self.coef) > threshold)
        if dist.get_rank() == 0:
            global_mask = local_mask.clone()
        else:
            global_mask = torch.empty_like(local_mask)
        dist.broadcast(global_mask, src=0)
        new_theta = self.theta.clone()
        new_coef = self.coef.clone()
        positions_to_replace = ~global_mask
        num_to_add = positions_to_replace.sum().item()
        if num_to_add > 0:
            add_coef = torch.DoubleTensor(num_to_add).uniform_(-1, 1).to(device)
            new_coef[positions_to_replace] = add_coef
        print(new_theta.shape)
        self.theta = nn.Parameter(new_theta)
        self.coef  = nn.Parameter(new_coef)
        self.M = self.theta.shape[0]
        dist.barrier()

    
    def energy(self, local_indices):
        c_chunk = self.coef[local_indices]
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        theta = (torch.arange(1, self.K+1).view(self.K,1,1).to(device) * self.theta[None,:,:].to(device)).reshape(self.K * self.M ,self.num_site)
        theta_chunk = theta[local_indices]
        def expval_einsum_batch(c_m, theta_m):
            theta_tensor_m_m = theta_m.unsqueeze(0) - theta
            theta_tensor_m_p = theta_m.unsqueeze(0) + theta
            product_term = torch.prod(torch.cos(theta_tensor_m_m), dim=-1)
            normalize_term = c_m * (self.coef.dot(product_term))
            # x term
            Pauli_m = torch.sin(theta_tensor_m_p) / (torch.cos(theta_tensor_m_m) + 1e-10) # Shape (M,L)
            x_term = torch.sum(Pauli_m, dim = -1)
            # z term
            Pauli_m = torch.cos(theta_tensor_m_p) / (torch.cos(theta_tensor_m_m) + 1e-10)
            z_term = torch.sum(Pauli_m, dim = -1)
            # zz term
            k_indices, k_prime_indices = self.graph[:, 0].to(device), self.graph[:, 1].to(device)
            coupling_J = self.dist
            Pauli_mm = coupling_J * Pauli_m[..., k_indices] * Pauli_m[..., k_prime_indices] # Size (M,L')
            zz_term = torch.sum(Pauli_mm, dim= -1)
            #Combine
            expectation_value = c_m * torch.dot(torch.mul(self.coef, product_term), self.Jz * zz_term - self.hx * x_term - self.hz * z_term)
            return normalize_term, expectation_value
        if self.method == 'vmap':
            normalize_term_batch, expectation_value_batch = torch.vmap(expval_einsum_batch)(c_chunk, theta_chunk)
            normalize_term = torch.sum(normalize_term_batch)
            expectation_value = torch.sum(expectation_value_batch)
        elif self.method == 'loop':
            normalize_term_batch = []
            expectation_value_batch = []
            for c_m, theta_m in zip(c_chunk, theta_chunk):
                normalize_term_m, expectation_value_m = expval_einsum_batch(c_m, theta_m)
                normalize_term_batch.append(normalize_term_m)
                expectation_value_batch.append(expectation_value_m)
            normalize_term = torch.sum(torch.stack(normalize_term_batch))
            expectation_value = torch.sum(torch.stack(expectation_value_batch))
        return normalize_term, expectation_value
    
    def normalization_const(self, local_indices):
        c_chunk = self.coef[local_indices]
        theta = (torch.arange(1, self.K+1).view(self.K,1,1).to(self.device) * self.theta[None,:,:]).reshape(-1,self.num_site)
        theta_chunk = theta[local_indices]
        def normalize_batch(c_m, theta_m):
            theta_tensor_m_m = theta_m.unsqueeze(0) - self.theta
            product_term = torch.prod(torch.cos(theta_tensor_m_m), dim=1)
            normalize_term = c_m * (self.coef.dot(product_term))
            return normalize_term
        if self.method == 'vmap':
            normalize_term_batch = torch.vmap(normalize_batch)(c_chunk, theta_chunk)
            normalize_term_chuck = torch.sum(normalize_term_batch)
        elif self.method == 'loop':
            normalize_term_batch = []
            for c_m, theta_m in zip(c_chunk, theta_chunk):
                normalize_term_m = normalize_batch(c_m, theta_m)
                normalize_term_batch.append(normalize_term_m)
            normalize_term_chuck = torch.sum(torch.stack(normalize_term_batch))
        dist.barrier()
        normalize_term = AllReduce.apply(normalize_term_chuck)
        return normalize_term
    
    def fubini_study(self, local_indices):
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        gs_exact = self.gs_exact
        tensor_array = gs_exact.get_B
        target_MPS = [torch.tensor(tensor_array(i)._data[0]).to(device) for i in range(self.num_site)]
        c_chunk = self.coef[local_indices]
        theta_chunk = self.theta[local_indices]
        def fubini_study_batch(c_m, theta_m):
            theta_tensor_m_m = theta_m.unsqueeze(0) - self.theta
            product_term = torch.prod(torch.cos(theta_tensor_m_m), dim=1)
            normalize_term = c_m * (self.coef.dot(product_term))
            cos_theta = torch.cos(theta_m)
            sin_theta = torch.sin(theta_m)
            MPS1 = torch.stack((cos_theta,sin_theta),dim=1)
            inner_product_list = [torch.einsum('p,ipj-> ij', mps1,mps2).to(device) for (mps1,mps2) in zip(MPS1,target_MPS)]
            inner_product = c_m * torch.linalg.multi_dot(inner_product_list).reshape(-1)
            return normalize_term, inner_product
        if self.method == 'vmap':
            normalize_term_batch, inner_product_batch = torch.vmap(fubini_study_batch)(c_chunk, theta_chunk)
            normalize_term_chunk = torch.sum(normalize_term_batch)
            inner_product_chunk = torch.sum(inner_product_batch)
        elif self.method == 'loop':
            normalize_term_batch = []
            inner_product_batch = []
            for c_m, theta_m in zip(c_chunk, theta_chunk):
                normalize_term_m, inner_product_m = fubini_study_batch(c_m, theta_m)
                normalize_term_batch.append(normalize_term_m)
                inner_product_batch.append(inner_product_m)
            normalize_term_chunk = torch.sum(torch.stack(normalize_term_batch))
            inner_product_chunk = torch.sum(torch.stack(inner_product_batch))
        return normalize_term_chunk, inner_product_chunk

    def forward(self, local_indices):
        if self.opt_mode == 'energy':
            normalize_term_chunk, expectation_value_chunk = self.energy(local_indices)
            agg_expectation = AllReduce.apply(expectation_value_chunk)
            agg_normalize = AllReduce.apply(normalize_term_chunk)
            energy = agg_expectation / (agg_normalize)
            return energy
        elif self.opt_mode == 'supervise':
            normalize_term_chunk, inner_product_chunk = self.fubini_study(local_indices)
            agg_inner_product = AllReduce.apply(inner_product_chunk)
            agg_normalize = AllReduce.apply(normalize_term_chunk)
            inner_product = torch.sqrt(agg_inner_product**2/agg_normalize)
            loss = torch.arccos(inner_product)
            return loss

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.world_size = dist.get_world_size()
        output = tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)
        return grad_input

def find_free_port():
    """
    Find an available port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Let the OS pick an available port
        return s.getsockname()[1]

def get_unique_nodes():
    """
    Reads PBS_NODEFILE and returns a list of unique nodes in the order they appear.
    """
    pbs_nodefile = os.environ.get("PBS_NODEFILE")
    if pbs_nodefile and os.path.exists(pbs_nodefile):
        with open(pbs_nodefile, 'r') as f:
            nodes = [line.strip() for line in f if line.strip()]
        unique_nodes = []
        for node in nodes:
            if node not in unique_nodes:
                unique_nodes.append(node)
        return unique_nodes
    return ["127.0.0.1"]

def get_master_addr(unique_nodes):
    """
    Choose the first node in the unique list as the master node.
    """
    return unique_nodes[0]

def get_node_rank(unique_nodes):
    """
    Determines the node rank by comparing the current hostname with the list of unique nodes.
    """
    current_host = socket.gethostname().strip()
    if current_host in unique_nodes:
        return unique_nodes.index(current_host)
    for idx, node in enumerate(unique_nodes):
        if node in current_host or current_host in node:
            return idx
    return 0

def get_allocated_gpu_count():
    """
    Count the GPUs allocated to the job based on CUDA_VISIBLE_DEVICES.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        # Count the comma-separated GPUs.
        return len(visible_devices.split(","))
    else:
        # If CUDA_VISIBLE_DEVICES is not set, fall back to torch's device count.
        return torch.cuda.device_count()

def NN_ansatz_DDP_multinode(system_param, opt_method):
    """
    Compute the energy using Distributed Data Parallel (DDP) across multiple nodes (if applicable).
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    USE_MPI_INIT = False
    unique_nodes = get_unique_nodes()
    master_addr = get_master_addr(unique_nodes)
    if USE_MPI_INIT:
        dist.init_process_group(backend=backend, init_method="mpi")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    else:
        global_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        master_port = str(find_free_port()) 
        print(master_port)
        os.environ["RANK"] = str(global_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        dist.init_process_group(backend=backend, init_method="env://")
    hostname = socket.gethostname()

    print(f"STARTING === Global Rank: {global_rank}, Local Rank: {local_rank}, "
          f"Host: {hostname}, World Size: {world_size} ===")
    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    M = system_param['M']
    K = system_param['K']
    lr = opt_method['lr']
    energy_list = []
    model = SPS_ddp(system_param, opt_method).to(device)
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Values: \n{param.data}\n")
        print("=========================")
    if torch.cuda.is_available():
        ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        ddp_model = nn.parallel.DistributedDataParallel(model) 
    
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr)
    num_epoch = opt_method['epochs']
    filename_result = system_param['filename']
    h5_filename = f"{model.num_site}_qubits/{model.M}_layers/history_{filename_result}.h5"
    if os.path.exists(h5_filename):
        os.remove(h5_filename)
    if (opt_method['resampling_round'] > 0):
        resampling_epochs = num_epoch // (opt_method['resampling_round'] + 1)
    for epoch in range(num_epoch):
        time_exp_start = time.perf_counter()
        if (opt_method['resampling_round'] > 0):
            if ((epoch + 1) % resampling_epochs == 0) and ((epoch + 1) != num_epoch):
                prune_model = ddp_model.module
                if system_param['M'] != 1:
                    prune_model.check_c()
                ddp_model = nn.parallel.DistributedDataParallel(prune_model, device_ids=[local_rank], output_device=local_rank)
                M = ddp_model.module.M
                print(f'M={M}')
                print(f'Theta shape: {ddp_model.module.theta.shape}')
                optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr)
        optimizer.zero_grad()
        indices_per_proc = (K * M) // world_size
        start_idx = global_rank * indices_per_proc
        end_idx = (global_rank + 1) * indices_per_proc if global_rank != world_size - 1 else K * M
        local_indices = list(range(start_idx, end_idx))
        loss = ddp_model(local_indices)
        loss.backward()
        optimizer.step()
        if opt_method['opt_mode'] == 'energy':
            energy = loss/system_param['num_site']
            if global_rank == 0:
                energy_list.append(energy.detach().item())
                if (epoch + 1) % 100 == 0:
                    time_exp_end = time.perf_counter()
                    print(f'Epoch [{epoch+1}/{num_epoch}], energy: {energy:.8f}')
                    print(f'Optimized done in {time_exp_end-time_exp_start} seconds')

def coupling_indices(system_param):
    sys_dim = system_param['dim']
    if sys_dim == '1D':
        if system_param['lattice'] == 'square':
            Lx = system_param['Lx']
            neighbors = torch.arange(Lx - 1).unsqueeze(-1) + torch.tensor([0, 1])
            dists = torch.ones(neighbors.shape[0])
        elif system_param['lattice'] == 'longrange':
            Lx = system_param['Lx']
            neighbors = []
            for i in range(Lx-1):
                for j in range(i+1 , Lx):
                   neighbors.append([i,j])
                   dist = 1/abs(i - j)**system_param['alpha']
                   dists.append(dist)
            neighbors = torch.tensor(neighbors)
            dists = torch.tensor(dists)
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
            dists = torch.ones(neighbors.shape[0])

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
            dists = torch.ones(neighbors.shape[0])
        elif system_param['lattice'] == 'triangle':
            Lx = system_param['Lx']
            Ly = system_param['Ly']
            neighbors = []
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
                        neighbors.append(sorted([i, j]))

                    # Down neighbor
                    if y + 1 < Ly:
                        j = get_snake_index(x, y + 1, Lx)
                        neighbors.append(sorted([i, j]))

                    # Diagonal down-right (regardless of even/odd row)
                    if x + 1 < Lx and y + 1 < Ly:
                        j = get_snake_index(x + 1, y + 1, Lx)
                        neighbors.append(sorted([i, j]))
            neighbors = torch.tensor(neighbors)
            dists = torch.ones(neighbors.shape[0])
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
            dists = torch.ones(neighbors.shape[0])
        elif system_param['lattice'] == 'longrange':
            Lx = system_param['Lx']
            Ly = system_param['Ly']
            Lz = system_param['Lz']
            num_site = Lx * Ly * Lz
            neighbors = []
            dists = []
            for i in range(num_site-1):
                for j in range(i+1 , num_site):
                   neighbors.append([i,j])
                   dist = snake_distance(i, j, Lx, Ly, Lz)
                   dists.append(1/dist**system_param['alpha'])
            neighbors = torch.tensor(neighbors)
            dists = torch.tensor(dists)
    return neighbors, dists

def snake_index_to_xyz(i, Lx, Ly, Lz):
    """
    Convert snake index i (0-based) to (x,y,z) for a 3D lattice (Lx, Ly, Lz)
    using a serpentine order:
      - Within each z-layer, y-rows are traversed in opposite order for odd z.
      - Within each row, x is traversed left-to-right for even y, right-to-left for odd y.
    Supports tensor i (any shape) of dtype long.
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
    x,y,z can be tensors; returns 0-based snake index with same shape.
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
    i, j: scalars or tensors (broadcastable to same shape), dtype long.
    spacing: (ax, ay, az) lattice spacings (floats).
    Returns: tensor of distances with broadcasted shape of i and j.
    """
    ax, ay, az = spacing
    xi, yi, zi = snake_index_to_xyz(i, Lx, Ly, Lz)
    xj, yj, zj = snake_index_to_xyz(j, Lx, Ly, Lz)

    dx = (xi - xj).to(torch.float32) * ax
    dy = (yi - yj).to(torch.float32) * ay
    dz = (zi - zj).to(torch.float32) * az

    return torch.sqrt(dx*dx + dy*dy + dz*dz).item()

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
            for u1, u2, dx in self.lat.pairs['all_to_all']:
                r = abs(dx[0])
                J_eff = Jz / r**alpha
                self.add_coupling(J_eff, u1, 'Sz', u2, 'Sz', dx)
        elif model_params['connection_type'] == 'random' or model_params['connection_type'] == 'square':
            for pair in model_params['graph']:
                i, j = pair
                self.add_coupling_term(Jz, i, j, 'Sz', 'Sz')
        # print(self.all_coupling_terms().to_TermList() + self.all_onsite_terms().to_TermList())


def compute_CF(state, L: int, opt_method: dict) -> float:
    """
    Compute the correlation function
      C^F_{L-1} = 1/(L-1) * sum_{l=2}^{L} <psi | sigma^z_{L/2} sigma^z_{l} | psi>
    
    Parameters:
      state: torch.Tensor
             The quantum state as a vector of shape (2**L,). It should already be normalized.
      L: int
         Number of qubits (sites).
      opt_method: dict
         Dictionary containing options such as 'dtype' and 'device'.
    
    Returns:
      CF: float
         The computed correlation function.
    """
    # Use the provided Pauli function to get sigma^z and identity
    sigma_z = Pauli('Z', opt_method)
    identity = Pauli('I', opt_method)
    
    # Define the target (middle) site.
    # Here we assume the formula is written in 1-indexed notation:
    #   sigma^z_{L/2} becomes index (L//2 - 1) in 0-indexing for even L,
    #   and L//2 for odd L.
    if L % 2 == 0:
        mid = L // 2 - 1
    else:
        mid = L // 2
    
    total = 0.0
    count = 0
    if L <= 14:
        # Loop over all sites (0-indexed) except the middle site.
        for l in range(L):
            if l == mid:
                continue
            
            # Build operator O = kron(..., sigma_z at mid, ..., sigma_z at l, ..., identity)
            ops = []
            for i in range(L):
                if i == mid or i == l:
                    ops.append(sigma_z)
                else:
                    ops.append(identity)
            full_op = kron_list(ops)
            
            # Compute expectation value <psi|full_op|psi>
            exp_val = torch.matmul(state.conj().unsqueeze(0), torch.matmul(full_op, state.unsqueeze(1)))
            total += exp_val.item()  # exp_val is a 1x1 tensor.
            count += 1
    else:
        for l in range(L):
            if l == mid:
                continue
            SzSz = [('Sz', l), ('Sz', mid)]
            # print(f'i: {i}, Sz: {mag_i}')
            twopoint = state.expectation_value_term(SzSz)
            corr_val = 4*(twopoint)
            # Compute the expectation value ⟨σ^z_{central} σ^z_l⟩
            total += corr_val
            count += 1
    # There should be L-1 terms since we exclude the mid site.
    CF = total / count
    return CF

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
    #print(H[0].shape)
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

def GS_TIM_MPS(system_param):
    model_params = {
    'Jx': 0 ,'Jy': 0 , 'Jz': system_param['Jz'] * 4 , 'hx': system_param['hx'] * 2, 'hz': system_param['hz'] * 2, 'alpha': system_param['alpha'],
    'L': system_param['num_site'],
    'bc_MPS': 'finite', 'connection_type': system_param['lattice'], 'graph': system_param['graph']
    }
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
        'max_sweeps': 10000,  # NOTE: this is not enough to fully converge at the critical point!
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
        # hamiltonian_mpo = model.calc_H_MPO()
        # print(hamiltonian_mpo._W)
        # for coor in system_param['graph']:
        #     i, j = coor
        #     model.add_coupling_term(model_params['Jz'], i, j, "Sz", "Sz", op_string=None, plus_hc=False)
    elif system_param['lattice'] == 'square' and system_param['dim'] == '3D':
        model_params['graph'] = system_param['graph']
        model = LongRangeSpinModel(model_params)
    else:
        model = SpinModel(model_params)
        # print(model.lat.mps_sites())
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
    'Jx': 0 ,'Jy': 0 ,'Jz': system_param['Jz'] * 4 , 'hx': system_param['hx'] * 2, 'hz': system_param['hz'] * 2,
    'lattice' : 'Square',
    'Lx': system_param['Lx'], 'Ly': system_param['Ly'],
    'bc_x': 'open', 'bc_y': 'ladder',
    'bc_MPS': 'finite'
    #'conserve': 'parity'
    }
    num_site = system_param['Lx'] * system_param['Ly']
    chi = 2**14
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'max_E_err': 1e-8,
        'max_S_err': 0.000001,
        'max_trunc_err': None,
        'max_sweeps': 10000,  # NOTE: this is not enough to fully converge at the critical point!
        'mixer': True,
        'verbose': 1,
        'max_trunc_err': None,
    }
    model = SpinModel(model_params)
    psi_DMRG = MPS.from_lat_product_state(model.lat, [[['up']] * (model_params['Lx'])])
    engine = dmrg.TwoSiteDMRGEngine(psi_DMRG, model, dmrg_params)
    GSE_DMRG, psi_DMRG = engine.run()
    return GSE_DMRG/num_site, psi_DMRG

def magnetization_MPS(gs_exact):
    L = gs_exact.L
    magnetization = []
    for i in range(L):
        Sz = [('Sz', i)]  # Sz operator at site i
        mag_i = 2*gs_exact.expectation_value_term(Sz)  # Expectation value of Sz at site i
        magnetization.append(mag_i.item())
    return magnetization

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
        param_tilted_ising = [system_param['Jz'], system_param['hx']]
        print(param_tilted_ising)
        time_exact_start = time.perf_counter()
        if num_site <= 15:
            exact_daig = exact_diagonalization(system_param, opt_method)
            gs_energy_exact = exact_daig['energy']
            gs_exact = exact_daig['state']
        else:
            gs_energy_exact, gs_exact = GS_2DTFIM_MPS(system_param['Jz']*4,system_param['hx']*2,num_site)
    elif system_param['dim'] == '3D':
        gs_energy_exact, gs_exact = GS_TIM_MPS(system_param)
    return gs_energy_exact.item(), gs_exact