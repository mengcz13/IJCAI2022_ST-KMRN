# target: finest resolution only
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 150 --seq_diff 48

# python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/solar_energy --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 150 --seq_diff 12
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/solar_energy --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 600 --seq_diff 48

# target: all resolutions at the same time
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 150 --seq_diff 48 --target_res "all"

python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/pems/METR-LA --seq_in_len 1440 --seq_out_len 288 --batch_size 2 --epochs 150 --seq_diff 48 --target_res "all"

# target: one model for each resolution
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 150 --seq_diff 48 --target_res "day"
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 150 --seq_diff 48 --target_res "6-hour"


# 0417
python exp_train_targets_repeat.py --root_path ../../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_ar --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/pems/PEMS-BAY --target_res all --seq_len 1440 --pred_len 288 --batch_size 1 --epochs 50 --cuda_devices 1

# 0419, sample resolutions
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --resolution_type sample --seq_len 1440 --pred_len 480 --batch_size 2 --epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/solar_energy --target_res all --resolution_type sample --seq_len 1440 --pred_len 480 --batch_size 1 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_ar --target_res all --resolution_type sample --seq_len 1440 --pred_len 480 --batch_size 1 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 1440 --pred_len 288 --batch_size 1 --epochs 50 --cuda_devices 0

# verification on PEMS-BAY
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/pems/PEMS-BAY --target_res 5min --resolution_type sample --seq_in_len 12 --seq_out_len 12 --seq_diff 1 --batch_size 8 --epochs 150
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/pems/PEMS-BAY --target_res 5min --resolution_type sample --seq_in_len 12 --seq_out_len 12 --seq_diff 48 --batch_size 8 --epochs 500

# seq_diff = 1
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --resolution_type sample --seq_len 1440 --pred_len 480 --seq_diff 1 --batch_size 2 --epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/solar_energy --target_res all --resolution_type sample --seq_len 1440 --pred_len 480 --seq_diff 1 --batch_size 1 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_ar --target_res all --resolution_type sample --seq_len 1440 --pred_len 480 --seq_diff 1 --batch_size 1 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 1440 --pred_len 288 --seq_diff 1 --batch_size 1 --epochs 50 --cuda_devices 0

python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --resolution_type sample --seq_len 480 --pred_len 480 --seq_diff 1 --batch_size 2 --epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/solar_energy --target_res all --resolution_type sample --seq_len 480 --pred_len 480 --seq_diff 1 --batch_size 1 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_ar --target_res all --resolution_type sample --seq_len 480 --pred_len 480 --seq_diff 1 --batch_size 1 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 288 --pred_len 288 --seq_diff 1 --batch_size 1 --epochs 50 --cuda_devices 0

# 0427 test smaller horizons
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/pems/PEMS-BAY --seq_in_len 144 --seq_out_len 144 --seq_diff 1 --batch_size 4 --epochs 50 --target_res all --resolution_type sample --seed 42
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/pems/PEMS-BAY --seq_in_len 72 --seq_out_len 72 --seq_diff 1 --batch_size 8 --epochs 50 --target_res all --resolution_type sample --seed 42
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/pems/PEMS-BAY --seq_in_len 36 --seq_out_len 36 --seq_diff 1 --batch_size 8 --epochs 50 --target_res all --resolution_type sample --seed 42

python exp_train_targets_repeat.py --root_path ../../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --batch_size 8 --epochs 50 --cuda_devices 0


# ECL dataset
python exp_train_targets_repeat.py --root_path ../../data/ecl --target_res all --seq_len 720 --pred_len 240 --seq_diff 24 --batch_size 1 --epochs 50 --cuda_devices 0

# test larger receptive fields
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --blocks 4 --layers 4 --kernel_size 4 --patience 50 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 1 --blocks 4 --layers 4 --kernel_size 4 --patience 50 --cuda_devices 0


# NYCTaxi Green
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --batch_size 2 --patience 50 --cuda_devices 0

# NYCTaxi FHV, Manhattan
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan_fhv --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --batch_size 2 --patience 50 --cuda_devices 0


# NYCTaxi, NYCTaxi(Green), fully observed data
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 1 --batch_size 2 --patience 20 --cuda_devices 0
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 1 --batch_size 2 --patience 20 --cuda_devices 0

# Solar Energy 10min, fully observed
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --batch_size 2 --patience 20 --cuda_devices 0

# other missing rates
{ python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.6 0.4 0.2 --batch_size 2 --patience 20 --seed 42 --cuda_devices 0 ; python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.6 0.4 0.2 --batch_size 2 --patience 20 --seed 42 --cuda_devices 0 } | cat
## more seeds
{ python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.6 0.4 0.2 --batch_size 2 --patience 20 --seed 43 44 --cuda_devices 0 ; python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.6 0.4 0.2 --batch_size 2 --patience 20 --seed 43 44 --cuda_devices 0 } | cat