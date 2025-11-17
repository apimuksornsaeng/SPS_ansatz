import torch
import torch.nn as nn
import numpy as np
import itertools
import time
import os
import time, resource, platform
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

class ManualAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = list(params)
        self.lr = lr; self.beta1, self.beta2 = betas; self.eps = eps; self.wd = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_num = 0

    @torch.no_grad()
    def step(self, grads):
        self.step_num += 1
        b1t = 1 - self.beta1 ** self.step_num
        b2t = 1 - self.beta2 ** self.step_num
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if g is None: continue
            # Adam moments
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = m / b1t
            v_hat = v / b2t
            # Decoupled weight decay
            if self.wd != 0:
                p.add_(p, alpha=-self.lr * self.wd)
            # Update
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

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
            dists = []
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

class SPS(nn.Module):
    def __init__(self, system_param: dict, opt_method: dict):
        super(SPS, self).__init__()
        self.system_param = system_param
        self.opt_method = opt_method
        self.num_site = system_param['num_site']   # L
        self.M = system_param['M']
        self.B = system_param.get('bs', 64)        # block size for n
        self.Jz = system_param['Jz']
        self.hx = system_param['hx']
        self.hz = system_param['hz']
        self.datatype = opt_method['dtype']
        self.device = opt_method['device']
        self.graph = system_param['graph']         # list[(k,k')]
        self.strength = system_param['strength']   # weights per edge
        self.theta = nn.Parameter(
            torch.DoubleTensor(self.M, self.num_site).uniform_(-torch.pi/2, torch.pi/2).to(self.device)
        )
        self.coef  = nn.Parameter(
            torch.DoubleTensor(self.M).uniform_(-1, 1).to(self.device)
        )
    def precheck_c(self):
        threshold = 1e-4
        local_mask = (abs(self.coef) > threshold)
        positions_to_replace = ~local_mask
        num_to_add = positions_to_replace.sum().item()
        return True if num_to_add > 0 else False

    def check_c(self):
        threshold = 1e-4
        print(f'Max |C|: {max(abs(self.coef))}')
        print(f'Min |C|: {min(abs(self.coef))}')
        with torch.no_grad():
            local_mask = (abs(self.coef) > threshold)
        new_theta = self.theta.clone()
        new_coef = self.coef.clone() 
        positions_to_replace = ~local_mask
        num_to_add = positions_to_replace.sum().item()
        if num_to_add > 0:

            add_coef = torch.DoubleTensor(num_to_add).uniform_(-1, 1).to(self.device)
            new_coef[positions_to_replace] = add_coef
            
            # full_new_theta = torch.cat((new_theta,torch.zeros(add_M,self.num_site).to(device)))
            # full_new_coef = torch.cat((new_coef,torch.ones(add_M).to(device)))
            print(new_theta.shape)
            self.theta = nn.Parameter(new_theta)
            self.coef  = nn.Parameter(new_coef)
            self.M = self.theta.shape[0]

    # ---------- FAST & LOW-MEM VERSION ----------
    @torch.no_grad()
    def grad_energy(self, bs: int = None, stabil_eps: float = 1e-12):
        """
        Exact energy & grads with O(M*bs) memory using on-the-fly two-pass scheme.
        Returns:
            dE_dtheta : (M, L)
            dE_dcoef  : (M,)
            energy_per_site : scalar
        """
        device, dtype = self.device, self.datatype
        M, L = self.M, self.num_site
        bs = self.B if bs is None else int(bs)

        c  = self.coef.to(device=device, dtype=dtype)
        th = self.theta.to(device=device, dtype=dtype)

        # Precompute sin/cos for "m" rows
        cos_m = torch.cos(th)   # (M,L)
        sin_m = torch.sin(th)   # (M,L)

        # Numerator/Denominator and their grads
        N = torch.zeros((), dtype=dtype, device=device)
        D = torch.zeros((), dtype=dtype, device=device)
        dN_dc     = torch.zeros_like(c)
        dD_dc     = torch.zeros_like(c)
        dN_dtheta = torch.zeros_like(th)
        dD_dtheta = torch.zeros_like(th)

        hx = torch.as_tensor(self.hx, dtype=dtype, device=device)
        hz = torch.as_tensor(self.hz, dtype=dtype, device=device)
        Jz = torch.as_tensor(self.Jz, dtype=dtype, device=device)

        edges = [(int(u), int(v)) for (u, v) in self.graph]
        strengths = torch.as_tensor(self.strength, dtype=dtype, device=device) if len(edges) else None

        # ---- helpers (site-indexed to avoid shape mismatch) ----
        def cos_delta_row(mrow: int, l: int, cos_b, sin_b):   # (B,)
            return cos_m[mrow, l] * cos_b[:, l] + sin_m[mrow, l] * sin_b[:, l]

        def sin_sum_row(mrow: int, l: int, cos_b, sin_b):     # (B,)
            return sin_m[mrow, l] * cos_b[:, l] + cos_m[mrow, l] * sin_b[:, l]

        def cos_sum_row(mrow: int, l: int, cos_b, sin_b):     # (B,)
            return cos_m[mrow, l] * cos_b[:, l] - sin_m[mrow, l] * sin_b[:, l]

        for start in range(0, M, bs):
            end   = min(start + bs, M)
            B     = end - start

            c_b   = c[start:end]                  # (B,)
            th_b  = th[start:end]                 # (B,L)
            cos_b = torch.cos(th_b)               # (B,L)
            sin_b = torch.sin(th_b)               # (B,L)

            for m in range(M):
                cm = c[m]

                # ---- pass 1: build P, U, V (B,) ----
                P = torch.ones(B, dtype=dtype, device=device)
                U = torch.zeros(B, dtype=dtype, device=device)  # sum_l sinΣ_l / cosΔ_l
                V = torch.zeros(B, dtype=dtype, device=device)  # sum_l cosΣ_l / cosΔ_l

                for l in range(L):
                    cosd = cos_delta_row(m, l, cos_b, sin_b)                     # cosΔ_l
                    den  = cosd + stabil_eps if stabil_eps else cosd
                    sins = sin_sum_row(m, l, cos_b, sin_b)                       # sinΣ_l
                    coss = cos_sum_row(m, l, cos_b, sin_b)                       # cosΣ_l
                    P *= cosd
                    U += sins / den
                    V += coss / den

                # ZZ aggregate over edges: Szz = Σ_e s_e * a_k * a_k'
                if edges:
                    Szz = torch.zeros(B, dtype=dtype, device=device)
                    for e_idx, (k, kp) in enumerate(edges):
                        s = strengths[e_idx]
                        a_k  = cos_sum_row(m, k,  cos_b, sin_b)
                        denk = cos_delta_row(m, k, cos_b, sin_b)
                        a_k  = a_k / (denk + stabil_eps if stabil_eps else denk)
                        a_kp  = cos_sum_row(m, kp,  cos_b, sin_b)
                        denkp = cos_delta_row(m, kp, cos_b, sin_b)
                        a_kp  = a_kp / (denkp + stabil_eps if stabil_eps else denkp)
                        Szz += s * (a_k * a_kp)
                else:
                    Szz = 0.0

                F_single = hz * V + hx * U
                F_total  = F_single + Jz * Szz
                base     = (cm * c_b) * P

                # accumulate N, D and grads wrt c
                N        += (base * F_total).sum()
                D        += base.sum()
                dN_dc[m] += 2.0 * (c_b * P * F_total).sum()
                dD_dc[m] += 2.0 * (c_b * P).sum()

                # ---- pass 2: θ-gradients (numerator & denominator) ----
                for q in range(L):
                    cosd = cos_delta_row(m, q, cos_b, sin_b)  # cosΔ_q  (B,)
                    den  = cosd + stabil_eps if stabil_eps else cosd
                    inv  = 1.0 / den
                    inv2 = inv * inv

                    # sinΔ_q, sinΣ_q, cosΣ_q (each (B,))
                    sind = sin_m[m, q] * cos_b[:, q] - cos_m[m, q] * sin_b[:, q]
                    sins = sin_sum_row(m, q, cos_b, sin_b)
                    coss = cos_sum_row(m, q, cos_b, sin_b)

                    # dD/dθ_{m,q} from P: P * (-tanΔ_q)
                    dD_pair = (cm * c_b) * P * (-(sind * inv))
                    dD_dtheta[m, q] += 2.0 * dD_pair.sum()

                    # single-site θ-term: G_xz(q)
                    Gxz = (-hz * sins + hx * coss) * inv + (hz * coss + hx * sins) * (sind * inv2)

                    # ZZ incident sum for q
                    if edges:
                        inc_sum = torch.zeros(B, dtype=dtype, device=device)
                        for e_idx, (u, v) in enumerate(edges):
                            if u == q or v == q:
                                nbr = v if u == q else u
                                s = strengths[e_idx]
                                a_nbr  = cos_sum_row(m, nbr,  cos_b, sin_b)
                                den_n  = cos_delta_row(m, nbr, cos_b, sin_b)
                                a_nbr  = a_nbr / (den_n + stabil_eps if stabil_eps else den_n)
                                inc_sum += s * a_nbr
                        # g_q = d/dθ (cosΣ_q / cosΔ_q)
                        Gzz = ( (-sins * inv) + (coss * (sind * inv2)) ) * inc_sum * Jz
                    else:
                        Gzz = 0.0

                    dN_pair = (cm * c_b) * P * (-(sind * inv) * F_total + Gxz + Gzz)
                    dN_dtheta[m, q] += 2.0 * dN_pair.sum()

        # quotient rule
        E = N / D
        dE_dc     = (dN_dc     - E * dD_dc)     / D
        dE_dtheta = (dN_dtheta - E * dD_dtheta) / D

        return dE_dtheta, dE_dc, E / self.num_site
    @torch.no_grad()
    def grad_energy_profiled(self):
        """
        Calls self.grad_energy() and returns (dE_dtheta, dE_dcoef, energy, stats)
        where stats includes wall time and memory usage.
        Works on CPU and CUDA (if your device is CUDA).
        """
        # ---- time start ----
        t0 = time.perf_counter()

        # ---- memory before ----
        # RSS (resident set) is the most useful cross-platform metric
        rss_before_mb = None
        if _HAS_PSUTIL:
            rss_before_mb = psutil.Process().memory_info().rss / (1024**2)

        # track peak resident set (OS-reported)
        ru0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # track CUDA memory (if applicable)
        is_cuda = torch.cuda.is_available()
        # print(is_cuda)
        if is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)
            cuda_before_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
        else:
            cuda_before_mb = None

        # ---- do the work ----
        dE_dtheta, dE_dcoef, energy = self.grad_energy_vectorized()

        # ---- time end ----
        if is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # ---- memory after ----
        if _HAS_PSUTIL:
            rss_after_mb = psutil.Process().memory_info().rss / (1024**2)
        else:
            rss_after_mb = None

        ru1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # ru_maxrss units differ by OS:
        #   Linux: kilobytes, macOS: bytes. Convert delta to MB.
        if platform.system() == "Linux":
            ru_delta_mb = (ru1 - ru0) / 1024.0
        else:
            ru_delta_mb = (ru1 - ru0) / (1024.0 * 1024.0)

        if is_cuda and torch.cuda.is_available():
            cuda_after_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
            cuda_peak_mb  = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        else:
            cuda_after_mb = None
            cuda_peak_mb  = None

        stats = {
            "wall_ms": (t1 - t0) * 1000.0,
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": rss_after_mb,
            "rss_delta_mb": (None if (rss_before_mb is None or rss_after_mb is None)
                             else rss_after_mb - rss_before_mb),
            "ru_maxrss_delta_mb": ru_delta_mb,
            "cuda_alloc_before_mb": cuda_before_mb,
            "cuda_alloc_after_mb": cuda_after_mb,
            "cuda_peak_mb": cuda_peak_mb,
        }
        return dE_dtheta, dE_dcoef, energy, stats
    
    @torch.no_grad()
    def grad_energy_vectorized(self, bs: int = None, stabil_eps: float = 1e-12):
        """
        Exact energy & grads with O(M*bs) memory using a fully vectorized two-pass scheme.
        No loops over m, l, q, or edges inside the batch; only the outer batching over B remains.
        Returns:
            dE_dtheta : (M, L)
            dE_dcoef  : (M,)
            energy_per_site : scalar
        """
        device, dtype = self.device, self.datatype
        M, L = self.M, self.num_site
        bs = self.B if bs is None else int(bs)

        c  = self.coef.to(device=device, dtype=dtype)          # (M,)
        th = self.theta.to(device=device, dtype=dtype)         # (M,L)
        cos_m = torch.cos(th)                                  # (M,L)
        sin_m = torch.sin(th)                                  # (M,L)

        hx = torch.as_tensor(self.hx, dtype=dtype, device=device)
        hz = torch.as_tensor(self.hz, dtype=dtype, device=device)
        Jz = torch.as_tensor(self.Jz, dtype=dtype, device=device)

        edges = [(int(u), int(v)) for (u, v) in self.graph]
        has_edges = len(edges) > 0
        strengths = torch.as_tensor(self.strength, dtype=dtype, device=device) if has_edges else None

        # Build a weighted (possibly sparse) neighbor matrix W where W[q, n] = sum of strengths on edges (q,n)
        # Used to compute inc_sum = Σ_n W[q,n] * A[..., n] for all q in parallel.
        if has_edges:
            uq = torch.tensor([u for (u, v) in edges], device=device)
            vq = torch.tensor([v for (u, v) in edges], device=device)
            s  = strengths
            # Undirected: add both directions; coalesce to handle multi-edges
            idx = torch.stack([torch.cat([uq, vq]), torch.cat([vq, uq])], dim=0)  # (2, 2E)
            vals = torch.cat([s, s])
            W = torch.sparse_coo_tensor(idx, vals, (L, L), dtype=dtype, device=device).coalesce()
        else:
            W = None

        # Numerator/Denominator and their grads (accumulated over batches)
        N = torch.zeros((), dtype=dtype, device=device)
        D = torch.zeros((), dtype=dtype, device=device)
        dN_dc     = torch.zeros_like(c)        # (M,)
        dD_dc     = torch.zeros_like(c)        # (M,)
        dN_dtheta = torch.zeros_like(th)       # (M,L)
        dD_dtheta = torch.zeros_like(th)       # (M,L)

        # Batch over target rows (b-index)
        for start in range(0, M, bs):
            end   = min(start + bs, M)
            B     = end - start

            c_b   = c[start:end]                       # (B,)
            th_b  = th[start:end]                      # (B,L)
            cos_b = torch.cos(th_b)                    # (B,L)
            sin_b = torch.sin(th_b)                    # (B,L)

            # --- All pair (m,b,l) trigs via broadcasting ---
            # Shapes: (M,1,L)*(1,B,L) -> (M,B,L)
            cosd = cos_m[:, None, :] * cos_b[None, :, :] + sin_m[:, None, :] * sin_b[None, :, :]     # cos(θ_m - θ_b)
            sins = sin_m[:, None, :] * cos_b[None, :, :] + cos_m[:, None, :] * sin_b[None, :, :]     # sin(θ_m + θ_b)
            coss = cos_m[:, None, :] * cos_b[None, :, :] - sin_m[:, None, :] * sin_b[None, :, :]     # cos(θ_m + θ_b)

            den  = cosd if stabil_eps == 0 else (cosd + stabil_eps)
            inv  = 1.0 / den
            inv2 = inv * inv
            sind = sin_m[:, None, :] * cos_b[None, :, :] - cos_m[:, None, :] * sin_b[None, :, :]     # sin(θ_m - θ_b)

            # --- First pass: P (product over L), U, V (sum over L) ---
            P = torch.prod(cosd, dim=-1)                                                            # (M,B)
            U = torch.sum(sins * inv, dim=-1)                                                       # (M,B)
            V = torch.sum(coss * inv, dim=-1)                                                       # (M,B)

            # A_l := cosΣ_l / cosΔ_l for ZZ terms, shape (M,B,L)
            A = coss * inv

            # Szz = Σ_e s_e * A_k * A_k'   -> vectorized with gather
            if has_edges:
                # Gather A on edge endpoints
                A_k  = A[:, :, uq]    # (M,B,E)
                A_kp = A[:, :, vq]    # (M,B,E)
                # strengths broadcast to (1,1,E)
                Szz = (A_k * A_kp * s.view(1, 1, -1)).sum(dim=-1)                                  # (M,B)
            else:
                Szz = 0.0

            F_single = hz * V + hx * U                                                              # (M,B)
            F_total  = F_single + Jz * Szz                                                          # (M,B)

            cm = c.view(M, 1)                                                                       # (M,1)
            cb = c_b.view(1, B)                                                                     # (1,B)
            base = (cm * cb) * P                                                                    # (M,B)

            # --- Accumulate N, D and grads wrt c ---
            N        = N + (base * F_total).sum()
            D        = D + base.sum()
            dN_dc    = dN_dc + 2.0 * ( (cb * P * F_total).sum(dim=1) )                              # (M,)
            dD_dc    = dD_dc + 2.0 * ( (cb * P).sum(dim=1) )                                        # (M,)

            # --- Second pass: θ-gradients (all q in parallel) ---
            # dD/dθ_{m,q} = 2 * Σ_b (cm*cb)*P * (-(sinΔ_q / cosΔ_q))
            factor_mb = (cm * cb) * P                                                               # (M,B)
            dD_pair = factor_mb[:, :, None] * (-(sind * inv))                                       # (M,B,L)
            dD_dtheta = dD_dtheta + 2.0 * dD_pair.sum(dim=1)                                        # sum over B -> (M,L)

            # Gxz(q) term, shape (M,B,L)
            Gxz = (-hz * sins + hx * coss) * inv + (hz * coss + hx * sins) * (sind * inv2)

            # Gzz(q): need inc_sum(q) = Σ_n W[q,n] * A[...,n]
            if has_edges:
                # tensordot over site dim: (M,B,L) x (L,L)^T -> (M,B,L)
                # Using sparse matmul: reshape (M*B,L) first to keep memory low
                AB = A.reshape(-1, L)                                                               # (M*B, L)
                # inc_sum = AB @ W.T   (sparse)
                inc_sum = torch.sparse.mm(W.transpose(0, 1), AB.transpose(0, 1)).transpose(0, 1)    # (M*B, L)
                inc_sum = inc_sum.reshape(M, B, L)
                Gzz = ( (-sins * inv) + (coss * (sind * inv2)) ) * inc_sum * Jz                     # (M,B,L)
            else:
                Gzz = 0.0

            dN_pair = factor_mb[:, :, None] * (-(sind * inv) * F_total[:, :, None] + Gxz + Gzz)     # (M,B,L)
            dN_dtheta = dN_dtheta + 2.0 * dN_pair.sum(dim=1)                                        # (M,L)

        # quotient rule
        E = N / D
        dE_dc     = (dN_dc     - E * dD_dc)     / D
        dE_dtheta = (dN_dtheta - E * dD_dtheta) / D

        return dE_dtheta, dE_dc, E / self.num_site
    
def training(system_param, opt_method):
    model = SPS(system_param, opt_method).to(opt_method['device'])
    optimizer = ManualAdamW(model.parameters(), lr=1e-1, weight_decay=1e-4)
    ref_dict = {('1D','square'): (-3.1037996506898637, 1e-3), 
                ('3D','square'): (-7.09401002059795, 1.89e-3), 
                ('1D','longrange'): (-3.1647095942288903, 1e-3),
                ('3D','longrange'): (-9.536033167577786, 3.71e-5),
                ('1D','random'): (-8.659422191166822, 1.16e-5)}
    ref_tuple = ref_dict[(system_param['dim'],system_param['lattice'])]
    ref_energy = ref_tuple[0]
    total_time = 0
    total_mem = 0
    num_epoch = opt_method['epochs']
    collect = False
    if (opt_method['resampling_round'] > 0):
        resampling_epochs = num_epoch // (opt_method['resampling_round'] + 1)
    for epoch in range(num_epoch):
        if (opt_method['resampling_round'] > 0):
            if model.precheck_c():
                if ((epoch + 1) % resampling_epochs == 0) and ((epoch + 1) != num_epoch):
                    if system_param['M'] != 1:
                        model.check_c()
                    M = model.M
                    print(f'M={M}')
                    print(f'Theta shape: {model.theta.shape}')
                    optimizer = ManualAdamW(model.parameters(), lr=1e-1, weight_decay=1e-4)
        dE_dtheta, dE_dcoef, energy, stats = model.grad_energy_profiled()
        optimizer.step([dE_dtheta, dE_dcoef])
        rel_error = (ref_energy - energy.item())/ref_energy
        total_time += stats['wall_ms']
        total_mem += stats['ru_maxrss_delta_mb']
        # Print both energy and perf
        print(f"Epoch {epoch:3d}: "
              f"Energy = {energy.item():.6f} | "
              f"Ref Error = {rel_error:.6f} | "
              f"time = {stats['wall_ms']:.2f} ms | "
              f"rssΔ ≈ {stats['rss_delta_mb'] if stats['rss_delta_mb'] is not None else float('nan'):.2f} MB | "
              f"peakΔ ≈ {stats['ru_maxrss_delta_mb']:.2f} MB")
        
        if rel_error < ref_tuple[1]:
            collect = True
            break
    print(f'Total time = {total_time / 1000:.2f} s |'
          f'Total peak memory = {total_mem} MB')

    return model, total_time/1000, collect

system_param = {
        'dim': '1D',
        'lattice': 'random', #['square','longrange','random','triangle'],
        'alpha': 0.4, #Also connected probability in random lattice
        'Lx': 40, #int(os.environ['L']),
        'Ly': 4, #int(np.sqrt(num_site)),
        'Lz': 4,
        'Jz': -1.0,
        'hx': 3.0,
        'hz': 0.25,
        'M': 32,
}

if system_param['dim'] == '1D':
        num_site = system_param['Lx']
elif system_param['dim'] == '2D':
        num_site = system_param['Lx'] * system_param['Ly']
elif system_param['dim'] == '3D':
        num_site = system_param['Lx'] * system_param['Ly'] * system_param['Lz']
system_param['num_site'] = num_site


opt_method = {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'dtype': torch.double,
        'epochs': 1000,
        'resampling_round': 1,
}

pairs_list, strength = coupling_indices(system_param)
system_param['graph'] = pairs_list
system_param['strength'] = strength

print(opt_method['device'])

num_samples = 40
total_time_list = []
for _ in range(num_samples):
    _, total_time, collect = training(system_param, opt_method)
    if collect:
        total_time_list.append(total_time)

savetxt = [np.mean(total_time_list),np.std(total_time_list)]

print(f'Time = {np.mean(total_time_list)} ± {np.std(total_time_list)}')
np.savetxt(f'time_{system_param['dim']}_{system_param['lattice']}',savetxt)
