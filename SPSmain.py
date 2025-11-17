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
    system_param = {
        'dim': '3D',
        'lattice': 'longrange', #['square','longrange','random','triangle'],
        'bc': 'finite', #['finite','periodic']
        'alpha': float(os.environ['L']), #Also connected probability in random lattice
        'Lx': 4, #int(os.environ['L']),
        'Ly': 4, #int(np.sqrt(num_site)),
        'Lz': 4,
        'Jz': -1.0, #-1.0,
        'hx': 7.0, #float(os.environ['L']), #float(os.environ['L'])
        'hz': 0.25,
        'M': 2**int(os.environ['M']),
        'K': 1    
    }
    if system_param['dim'] == '1D':
        num_site = system_param['Lx']
    elif system_param['dim'] == '2D':
        num_site = system_param['Lx'] * system_param['Ly']
    elif system_param['dim'] == '3D':
        num_site = system_param['Lx'] * system_param['Ly'] * system_param['Lz']
    system_param['num_site'] = num_site  
    M = system_param['M'] #int(os.environ['M'])
    trained = True
    system_param['chi'] = 1000
    main_folder = f'{num_site}_qubits'
    sub_folder = f'{M}_layers'
    sub_folder_path = os.path.join(main_folder,sub_folder)

    # Check if folder does not already exist
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
        print(f"Created folder '{main_folder}'.")
    else:
        print(f"Folder '{main_folder}' already exists.")

    if not os.path.exists(sub_folder_path):
        os.mkdir(sub_folder_path)
        print(f"Created sub folder '{sub_folder}'.")
    else:
        print(f"Subfolder '{sub_folder}' already exists.")

    opt_method = {
        'type': None,
        'opt_mode': 'energy', #['supervise', 'energy']
        'device': device,
        'dtype': datatype,
        'SPS_type': 'typical', # 'typical', 'max_entropy', 'plus', 'paramagnetic'
        'parallel': do_parallel,
        'method': 'vmap',
        'epochs': 20000,
    }
    resampling_every = 5000
    opt_method['resampling_round'] = (opt_method['epochs'] // resampling_every) - 1
    pairs_list, strength = coupling_indices(system_param)
    system_param['graph'] = pairs_list
    system_param['dist'] = strength.to(device)
    print(pairs_list)

    param_tilted_ising = [system_param['Jz'], system_param['hx'], system_param['hz']]
    
    filename_result = f'{M}_{param_tilted_ising}_{opt_method['SPS_type']}'

    if system_param['lattice'] == 'longrange':
        filename_result += f'_longrange_α_{system_param['alpha']}' 
    elif system_param['lattice'] == 'random':
        filename_result += f'_random_p_{system_param['alpha']}' 
    elif system_param['lattice'] != 'square':
        filename_result += system_param['lattice'] 

    if opt_method['opt_mode'] == 'supervise':
        filename_result += '_supervise'
    
    if opt_method['resampling_round'] != 0:
        filename_result += '_resampling'

    

    if trained:
        filename_result = 'trained_' + filename_result
        learning_rate = 1e-3
        opt_method['lr'] = learning_rate

    time_start = time.time()

    time_end = time.time()
    print(f'Reference use {time_end - time_start} seconds')

    
    system_param['filename'] = filename_result


    num_samples = 10
    num_epoch = opt_method['epochs']
    energy_list_M = []
    time_list = []
    mem_list = []

    for _ in range(num_samples):
        # Identify current process (for DDP, rank helps distinguish logs)
        pid = os.getpid()
        process = psutil.Process(pid)

        # --- Reset GPU memory stats ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        # --- Record initial CPU memory ---
        mem_before = process.memory_info().rss / 1024**2  # MB
        time_start = time.time()
        energy, _ = NN_ansatz_DDP_multinode(system_param, opt_method)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_end = time.time()
        # --- Record memory after computation ---
        mem_after = process.memory_info().rss / 1024**2  # MB
        mem_cpu_used = mem_after - mem_before
        # --- GPU memory info ---
        if torch.cuda.is_available():
            peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            curr_gpu_mem = torch.cuda.memory_allocated() / 1024**2
        else:
            peak_gpu_mem, curr_gpu_mem = 0.0, 0.0
        energy_list_M.append(energy)
        time_total = time_end - time_start
        time_list.append(time_total/10**2)
        if torch.cuda.is_available():
            mem_list.append(peak_gpu_mem)
        else:
            mem_list.append(mem_cpu_used)
        del energy
        gc.collect()

    with open(os.path.join(sub_folder_path,f"energy_{filename_result}_GPU.csv"), "w") as p:
        writer = csv.writer(p)
        writer.writerow(energy_list_M)
        writer.writerow(time_list)
        writer.writerow(mem_list)