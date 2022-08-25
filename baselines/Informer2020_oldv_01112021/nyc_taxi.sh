python main_informer.py --model informer --root_path ../../data/nyc_taxi/manhattan --data NYCTaxi_30d-7d --attn prob --features M --batch_size 4 --train_epochs 300
python main_informer.py --model informer --root_path ../../data/nyc_taxi/manhattan --data NYCTaxi_48h-24h --attn prob --features M --batch_size 8 --train_epochs 300

# 7d->7d, stride 1d
python main_informer.py --model informer --root_path ../../data/nyc_taxi/manhattan --data NYCTaxi_7d-7d-1stride --attn prob --features M --batch_size 4 --train_epochs 300


# Multires, all nodes
## 30d->10d
python main_informer.py --model informer --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 50 --num_workers 8

python main_informer.py --model informer --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 50 --num_workers 8 --seq_diff 48

# python main_informer.py --model informer --data SolarEnergy_Multires_6h-1d --root_path ../../data/solar_energy --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 50 --num_workers 8 --seq_diff 12
python main_informer.py --model informer --data SolarEnergy_Multires_6h-1d --root_path ../../data/solar_energy --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 300 --num_workers 8 --seq_diff 48 --patience 20 

python main_informer.py --model gru --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 150 --num_workers 16 --seq_diff 48 --e_layers 2 --d_layers 2

# python main_informer.py --model gru --data SolarEnergy_Multires_6h-1d --root_path ../../data/solar_energy --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 150 --num_workers 8 --seq_diff 12 --e_layers 2 --d_layers 2
python main_informer.py --model gru --data SolarEnergy_Multires_6h-1d --root_path ../../data/solar_energy --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 300 --num_workers 8 --seq_diff 48 --e_layers 2 --d_layers 2 --patience 20


# test higher missing rates, keep bs=1
python main_informer.py --model informer --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 1 --train_epochs 50 --num_workers 8 --seq_diff 48 --keep_ratio 0.8
python main_informer.py --model informer --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 1 --train_epochs 50 --num_workers 8 --seq_diff 48 --keep_ratio 0.5
python main_informer.py --model informer --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 1 --train_epochs 50 --num_workers 8 --seq_diff 48 --keep_ratio 0.2


# (1) target: finest resolution only [done]
python main_informer.py --model informer --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 50 --num_workers 8 --seq_diff 48
python main_informer.py --model informer --data SolarEnergy_Multires_6h-1d --root_path ../../data/solar_energy --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 300 --num_workers 8 --seq_diff 48 --patience 20
python main_informer.py --model informer --data METR-LA_Multires_1h-4h --root_path ../../data/pems/METR-LA --attn prob --features M --seq_len 1440 --pred_len 288 --label_len 288 --batch_size 4 --train_epochs 300 --num_workers 8 --seq_diff 48 --patience 20 --target_res "5min"

python main_informer.py --model gru --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 150 --num_workers 16 --seq_diff 48 --e_layers 2 --d_layers 2
python main_informer.py --model gru --data SolarEnergy_Multires_6h-1d --root_path ../../data/solar_energy --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 300 --num_workers 8 --seq_diff 48 --e_layers 2 --d_layers 2 --patience 20

# (2) target: all resolutions at the same time
python main_informer.py --model informer --data NYCTaxi_Multires_6h-1d --root_path ../../data/nyc_taxi/manhattan --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 4 --train_epochs 50 --num_workers 8 --seq_diff 48 --target_res "all"


# (3) target: one model for each resolution [done for Informer]


python exp_train_targets_repeat.py --model gru --data "SolarEnergy_Multires_6h-1d" --root_path "../../data/solar_energy" --target_res all --seq_len 1440 --pred_len 480 --patience 20 --cuda_devices 0


# 0417
python exp_train_targets_repeat.py --model gru --data "SolarEnergyAr_Multires_6h-1d" --root_path "../../data/solar_energy_ar" --target_res all 30min 6-hour day --seq_len 1440 --pred_len 480 --patience 20 --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "SolarEnergyAr_Multires_6h-1d" --root_path "../../data/solar_energy_ar" --target_res all 30min 6-hour day --seq_len 1440 --pred_len 480 --patience 20 --cuda_devices 1

python exp_train_targets_repeat.py --model gru --data "PEMS-BAY_Multires_1h-4h" --root_path "../../data/pems/PEMS-BAY" --target_res all 5min 1-hour 4-hour --seq_len 1440 --pred_len 288 --patience 20 --cuda_devices 2
python exp_train_targets_repeat.py --model informer --data "PEMS-BAY_Multires_1h-4h" --root_path "../../data/pems/PEMS-BAY" --target_res all 5min 1-hour 4-hour --seq_len 1440 --pred_len 288 --patience 20 --cuda_devices 3

# 0419, test sampling resolutions
python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --patience 10 --resolution_type sample --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "SolarEnergy_Multires_6h-1d" --root_path "../../data/solar_energy" --target_res all --seq_len 1440 --pred_len 480 --patience 10 --resolution_type sample --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "SolarEnergyAr_Multires_6h-1d" --root_path "../../data/solar_energy_ar" --target_res all --seq_len 1440 --pred_len 480 --patience 10 --resolution_type sample --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "PEMS-BAY_Multires_1h-4h" --root_path "../../data/pems/PEMS-BAY" --target_res all --seq_len 1440 --pred_len 288 --patience 10 --resolution_type sample --cuda_devices 0
# 0419, test sampling resolutions
python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 480 --pred_len 480 --seq_diff 1 --patience 10 --resolution_type sample --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "SolarEnergy_Multires_6h-1d" --root_path "../../data/solar_energy" --target_res all --seq_len 480 --pred_len 480 --seq_diff 1 --patience 10 --resolution_type sample --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "SolarEnergyAr_Multires_6h-1d" --root_path "../../data/solar_energy_ar" --target_res all --seq_len 480 --pred_len 480 --seq_diff 1 --patience 10 --resolution_type sample --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "PEMS-BAY_Multires_1h-4h" --root_path "../../data/pems/PEMS-BAY" --target_res all --seq_len 288 --pred_len 288 --seq_diff 1 --patience 10 --resolution_type sample --cuda_devices 0

python exp_train_targets_repeat.py --model informer --data "PEMS-BAY_Multires_1h-4h" --root_path "../../data/pems/PEMS-BAY" --target_res all --attn full --seq_len 36 --pred_len 36 --seq_diff 1 --patience 10 --resolution_type sample --cuda_devices 0
python exp_train_targets_repeat.py --model gru --data "PEMS-BAY_Multires_1h-4h" --root_path "../../data/pems/PEMS-BAY" --target_res all --seq_len 36 --pred_len 36 --seq_diff 1 --patience 10 --resolution_type sample --cuda_devices 0


# ECL data, all resolutions at the same time
python exp_train_targets_repeat.py --model informer --data ECL_Multires_6h-1d --root_path "../../data/ecl" --target_res all --seq_len 720 --pred_len 240 --seq_diff 24 --patience 20 --num_workers 8 --cuda_devices 0
python exp_train_targets_repeat.py --model gru --data ECL_Multires_6h-1d --root_path "../../data/ecl" --target_res all --seq_len 720 --pred_len 240 --seq_diff 24 --patience 20 --num_workers 8 --cuda_devices 0

# NYCTaxi Green data
python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --train_epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res "30min" --seq_len 1440 --pred_len 480 --batch_size 4 --train_epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res "6-hour" --seq_len 1440 --pred_len 480 --batch_size 4 --train_epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res "day" --seq_len 1440 --pred_len 480 --batch_size 4 --train_epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --model gru --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --train_epochs 150 --cuda_devices 0

# NYCTaxi FHV Manhattan data
python exp_train_targets_repeat.py --model informer --data "NYCTaxiFHV_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan_fhv" --target_res all 30min 6-hour day --seq_len 1440 --pred_len 480 --batch_size 4 --train_epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --model gru --data "NYCTaxiFHV_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan_fhv" --target_res all 30min 6-hour day --seq_len 1440 --pred_len 480 --batch_size 4 --train_epochs 150 --cuda_devices 0


# NYCTaxi, NYCTaxi(Green), fully observed data
python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all 30min 6-hour day --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all 30min 6-hour day --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 50 --cuda_devices 0

python exp_train_targets_repeat.py --model gru --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all 30min 6-hour day --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 150 --cuda_devices 0
python exp_train_targets_repeat.py --model gru --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all 30min 6-hour day --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 150 --cuda_devices 0


# Solar Energy, 10min, fully observed
python exp_train_targets_repeat.py --model informer --data "SolarEnergy10min_Multires_1h-6h" --root_path "../../data/solar_energy_10min" --target_res all 10min 1-hour 6-hour --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --batch_size 4 --train_epochs 50 --cuda_devices 0
python exp_train_targets_repeat.py --model gru --data "SolarEnergy10min_Multires_1h-6h" --root_path "../../data/solar_energy_10min" --target_res all 10min 1-hour 6-hour --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --batch_size 4 --train_epochs 50 --cuda_devices 0


# test effect of adding minute embeddings
# python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 50 --chrono_arr_cats "mdwhm" --cuda_devices 0

# NYCTaxi, 60%, 40%, 20% observation in input
{ python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --batch_size 4 --train_epochs 50 --seed 42 --cuda_devices 0 ; python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --batch_size 4 --train_epochs 50 --seed 42 --cuda_devices 0} | cat
{ python exp_train_targets_repeat.py --model gru --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --batch_size 4 --train_epochs 150 --seed 42 --cuda_devices 1 ; python exp_train_targets_repeat.py --model gru --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --batch_size 4 --train_epochs 150 --seed 42 --cuda_devices 1} | cat 
# more random seeds (43, 44)
{ python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --batch_size 4 --train_epochs 50 --seed 43 44 --cuda_devices 0 ; python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --batch_size 4 --train_epochs 50 --seed 43 44 --cuda_devices 0} | cat
{ python exp_train_targets_repeat.py --model gru --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --batch_size 4 --train_epochs 150 --seed 43 44 --cuda_devices 1 ; python exp_train_targets_repeat.py --model gru --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --batch_size 4 --train_epochs 150 --seed 43 44 --cuda_devices 1} | cat 


# test other settings: bs=2, lr=1e-4, no mask in input, single res in input\
python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res "30min" --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 2 --train_epochs 100 --patience 15 --no_input_mask --single_input_res --cuda_devices 0 | parallel -j1
python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res "6-hour" --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 2 --train_epochs 100 --patience 15 --no_input_mask --single_input_res --cuda_devices 0 | parallel -j1
python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res "day" --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 2 --train_epochs 100 --patience 15 --no_input_mask --single_input_res --cuda_devices 0 | parallel -j1