for keep_ratio in [1.0, 0.8, 0.6, 0.4, 0.2]:
    for method in ['hist_avg', 'static']:
        for data_folder in ['manhattan', 'sel_green', 'solar_energy_10min']:
            if data_folder == 'manhattan':
                datafoldername = 'nyc_taxi/manhattan'
                resfoldername = 'nyctaxi'
                input_len, output_len, seq_diff = 1440, 480, 48
            elif data_folder == 'sel_green':
                datafoldername = 'nyc_taxi/sel_green'
                resfoldername = 'nyctaxi_green'
                input_len, output_len, seq_diff = 1440, 480, 48
            elif data_folder == 'solar_energy_10min':
                datafoldername = 'solar_energy_10min'
                resfoldername = 'solar_energy_10min'
                input_len, output_len, seq_diff = 1440, 432, 36
            if keep_ratio == 0.8:
                suffix = ''
            elif keep_ratio == 1.0:
                suffix = '_full'
            else:
                suffix = '_0{:d}'.format(int(keep_ratio * 10))
            cmd = 'python simple_baselines.py --method {method} --data_folder ../data/{datafoldername} --res_folder simple_baseline_res/{resfoldername}{suffix} --input_len {input_len} --output_len {output_len} --seq_diff {seq_diff} --keep_ratio {keep_ratio} --scale --baseres 30min'.format(method=method, datafoldername=datafoldername, resfoldername=resfoldername, suffix=suffix, input_len=input_len, output_len=output_len, seq_diff=seq_diff, keep_ratio=keep_ratio)
            print(cmd)