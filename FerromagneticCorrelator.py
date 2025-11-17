from SPSmodule import *

if __name__ == '__main__':
    #Computer
    datatype = torch.double
    use_gpu = True
    do_parallel = True
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"We use {device}")

    #SPS information
    SPS_type = 'Real'
    dim = '3D'
    system_param = {
        'dim': dim,
        'lattice': 'longrange', #['square','longrange','random'],
        'bc': 'finite', #['finite','periodic']
        'alpha': 0,
        'Lx': 4,
        'Ly': 4, 
        'Lz': 4,
        'Jz': -1.0,
    }
    if system_param['dim'] == '1D':
        num_site = system_param['Lx']
    elif system_param['dim'] == '2D':
        num_site = system_param['Lx'] * system_param['Ly']
    elif system_param['dim'] == '3D':
        num_site = system_param['Lx'] * system_param['Ly'] * system_param['Lz']
    system_param['num_site'] = num_site  

    opt_method = {
        'type': None,
        'opt_mode': 'energy', #['supervise', 'energy']
        'device': device,
        'dtype': datatype,
        'SPS_type': 'Real',
        'parallel': do_parallel,
        'method': 'vmap',
        'epochs': 5000,
    }
    resampling_every = 5000
    opt_method['resampling_round'] = (opt_method['epochs'] // resampling_every) - 1

    filename_result = f'phase_diagram_{dim}_{num_site}_qubits'

    if system_param['lattice'] == 'longrange':
        filename_result += '_longrange'
    
    elif system_param['lattice'] == 'random':
        filename_result += '_random' 

    if opt_method['opt_mode'] == 'supervise':
        filename_result += '_supervise'
    
    if system_param['lattice'] == 'square':
        graph, dists = coupling_indices(system_param)
        system_param['graph'] = graph
        system_param['dist'] = dists
        print(system_param['graph'])
        system_param['chi'] = 256
        hx_start = float(os.environ['hx'])
        hx_range = np.arange(hx_start + 0.02, hx_start + 0.12, 0.02) #np.linspace(0.0,5.0,120)
        hz_range = [0.25] #np.linspace(0.0,3.0,120)
        CF = []
        for hx in hx_range:
            CF_hx = []
            for hz in hz_range:
                print(f'hx = {hx}, hz = {hz}')
                system_param['hx'] = hx
                system_param['hz'] = hz
                gs_energy_exact, gs_exact = reference_energy(system_param, opt_method)
                cf = compute_CF(gs_exact, num_site, opt_method)
                print(cf)
                CF_hx.append(cf)
            CF.append(CF_hx)
    elif system_param['lattice'] == 'longrange':
        graph, dists = coupling_indices(system_param)
        system_param['graph'] = graph
        system_param['dist'] = dists
        print(system_param['graph'])
        system_param['chi'] = 32
        system_param['hx'] = 7.0
        system_param['hz'] = 0.25
        hx_start = float(os.environ['hx'])
        α_range = np.arange(hx_start, hx_start + 0.12, 0.02) #np.arange(hx_start, hx_start + 0.12, 0.02) #np.arange(0.0, 5.0 + 0.02, 0.02)
        CF = []
        for α in α_range:
            system_param['alpha'] = α
            print(f"α = {α}")
            gs_energy_exact, gs_exact = reference_energy(system_param, opt_method)
            cf = compute_CF(gs_exact, num_site, opt_method)
            print(cf)
            CF.append(cf)
    
    elif system_param['lattice'] == 'random':
        system_param['chi'] = 250
        system_param['hx'] = 3.0
        system_param['hz'] = 0.25
        p_range = np.arange(0.0, 1.0 + 0.02, 0.02)
        CF = []
        for p in p_range:
            system_param['alpha'] = p
            print(f'p = {p}')
            graph, dists = coupling_indices(system_param)
            system_param['graph'] = graph
            system_param['dist'] = dists
            gs_energy_exact, gs_exact = reference_energy(system_param, opt_method)
            cf = compute_CF(gs_exact, num_site, opt_method)
            print(cf)
            CF.append(cf)
    
    np.savetxt(f"{filename_result}_{hx_start}.csv", CF, delimiter=",", fmt="%f")

 