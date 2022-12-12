# Physics-Informed Long-Sequence Forecasting From Multi-Resolution Spatiotemporal Data

This repository is the official PyTorch implementation of "Physics-Informed Long-Sequence Forecasting From Multi-Resolution Spatiotemporal Data". [\[PDF\]](https://www.ijcai.org/proceedings/2022/0304.pdf)

## Bibtex

```bibtex
@inproceedings{meng2022physics,
  title={Physics-Informed Long-Sequence Forecasting From Multi-Resolution Spatiotemporal Data},
  author={Meng, Chuizheng and Niu, Hao and Habault, Guillaume and Legaspi, Roberto and Wada, Shinya and Ono, Chihiro and Liu, Yan},
  booktitle={Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence},
  year={2022}
}
```

## Environment
```bash
conda create -n mrst "python<3.8"
conda activate mrst
bash install.sh
```

## Preprocessed Datasets

Run the following command to decompress:
```bash
tar -xjvf data.tar.bz2
```

The preprocessed datasets are located in the following folders:
- YellowCab: `data/nyc_taxi/manhattan`
- GreenCab: `data/nyc_taxi/sel_green`
- Solar Energy: `data/solar_energy_10min`.

## Preprocessing

If you want to repeat the preprocessing step to reproduce data from the above section, use the following commands:

```bash
# YellowCab
python dataset/nyc_taxi/preprocessing.py --raw_data_folder /path/to/folder_of_yellow_trip_data_csv_files --save_folder data/nyc_taxi/manhattan --region_borough_mapping dataset/nyc_taxi/manhattan_taxizones_uhf42.csv --num_workers 4 --type yellow

# GreenCab
python dataset/nyc_taxi/preprocessing.py --raw_data_folder /path/to/folder_of_green_trip_data_csv_files --save_folder data/nyc_taxi/sel_green --region_borough_mapping dataset/nyc_taxi/greencab_taxizones_uhf42.csv --num_workers 4 --type green

# Solar Energy
pushd dataset/solar_energy
python preprocessing.py --raw_dir /path/to/uncompressed_al-pv-2006_folder --output_dir ../../data/solar_energy_10min --base_res 10
popd
```

## Main Experiments
We use [GNU Parallel](https://www.gnu.org/software/parallel/) to run each experiment three times with different random seeds (42/43/44). You may need to modify the `--cuda_devices` argument based on the availability of GPUs.

### With Fully Observed Input
```bash
# Our method
pushd multitask
## YellowCab
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0 | parallel -j1
popd

# Baselines
## Historical Averaging
pushd baselines
### YellowCab
python simple_baselines.py --method hist_avg --data_folder ../data/nyc_taxi/manhattan --res_folder simple_baseline_res/nyctaxi_full --input_len 1440 --output_len 480 --seq_diff 48 --keep_ratio 1 --scale --baseres 30min
### GreenCab
python simple_baselines.py --method hist_avg --data_folder ../data/nyc_taxi/sel_green --res_folder simple_baseline_res/nyctaxi_green_full --input_len 1440 --output_len 480 --seq_diff 48 --keep_ratio 1 --scale --baseres 30min
### Solar Energy
python simple_baselines.py --method hist_avg --data_folder ../data/solar_energy_10min --res_folder simple_baseline_res/solar_energy_10min_full --input_len 1440 --output_len 432 --seq_diff 36 --keep_ratio 1 --scale --baseres 10min
popd baselines

## Static
pushd baselines
### YellowCab
python simple_baselines.py --method static --data_folder ../data/nyc_taxi/manhattan --res_folder simple_baseline_res/nyctaxi_full --input_len 1440 --output_len 480 --seq_diff 48 --keep_ratio 1 --scale --baseres 30min
### GreenCab
python simple_baselines.py --method static --data_folder ../data/nyc_taxi/sel_green --res_folder simple_baseline_res/nyctaxi_green_full --input_len 1440 --output_len 480 --seq_diff 48 --keep_ratio 1 --scale --baseres 30min
### Solar Energy
python simple_baselines.py --method static --data_folder ../data/solar_energy_10min --res_folder simple_baseline_res/solar_energy_10min_full --input_len 1440 --output_len 432 --seq_diff 36 --keep_ratio 1 --scale --baseres 10min
popd baselines

## GRU
pushd baselines/Informer2020_oldv_01112021
## YellowCab
python exp_train_targets_repeat.py --model gru --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 150 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --model gru --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 150 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --model gru --data "SolarEnergy10min_Multires_1h-6h" --root_path "../../data/solar_energy_10min" --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --batch_size 4 --train_epochs 50 --cuda_devices 0
popd

## Informer
pushd baselines/Informer2020_oldv_01112021
## YellowCab
python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 50 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --batch_size 4 --train_epochs 50 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --model informer --data "SolarEnergy10min_Multires_1h-6h" --root_path "../../data/solar_energy_10min" --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --batch_size 4 --train_epochs 50 --cuda_devices 0 | parallel -j1
popd

## Graph WaveNet
pushd baselines/Graph-WaveNet
## YellowCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 1 --batch_size 2 --patience 20 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 1 --batch_size 2 --patience 20 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --batch_size 2 --patience 20 --cuda_devices 0 | parallel -j1
popd

## MTGNN
pushd baselines/MTGNN
## YellowCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 1 --batch_size 4 --patience 20 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 1 --batch_size 4 --patience 20 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --batch_size 2 --patience 20 --cuda_devices 0 | parallel -j1
popd

## KoopmanAE
pushd baselines/koopmanAE
## YellowCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 1 --batch_size 32 --epochs 300 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 1 --batch_size 32 --epochs 150 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --batch_size 32 --epochs 10 --cuda_devices 0 | parallel -j1
popd
```

### With Partially Observed Input
```bash
# Our method
pushd multitask
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 42 43 44 --cuda_devices 0 | parallel -j1

python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 3 | parallel -j1

python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 0.8 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0
popd

# Baselines
## Historical Averaging & Static
pushd baselines
python simple_baseline_keep_ratio_script.py | parallel -j1
popd

## GRU
pushd baselines/Informer2020_oldv_01112021
python exp_train_targets_repeat.py --model gru --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 4 --train_epochs 150 --seed 42 43 44 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --model gru --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 4 --train_epochs 150 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --model gru --data "SolarEnergy10min_Multires_1h-6h" --root_path "../../data/solar_energy_10min" --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 4 --train_epochs 50 --cuda_devices 0 | parallel -j1
popd

## Informer
pushd baselines/Informer2020_oldv_01112021
python exp_train_targets_repeat.py --model informer --data "NYCTaxi_Multires_6h-1d" --root_path "../../data/nyc_taxi/manhattan" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 4 --train_epochs 50 --seed 42 43 44 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --model informer --data "NYCTaxiGreen_Multires_6h-1d" --root_path "../../data/nyc_taxi/sel_green" --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 4 --train_epochs 50 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --model informer --data "SolarEnergy10min_Multires_1h-6h" --root_path "../../data/solar_energy_10min" --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 4 --train_epochs 50 --cuda_devices 0 | parallel -j1
popd

## Graph WaveNet
pushd baselines/Graph-WaveNet
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 2 --patience 20 --seed 42 43 44 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 2 --patience 20 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 2 --patience 20 --cuda_devices 0 | parallel -j1
popd

## MTGNN
pushd baselines/MTGNN
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 2 --patience 20 --seed 42 43 44 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 4 --patience 20 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 2 --patience 20 --cuda_devices 0 | parallel -j1
popd

## KoopmanAE
pushd baselines/koopmanAE
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 32 --epochs 300 --cuda_devices 0 | parallel -j1
## GreenCab
python exp_train_targets_repeat.py --root_path ../../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --seq_diff 48 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 32 --epochs 150 --cuda_devices 0 | parallel -j1
## Solar Energy
python exp_train_targets_repeat.py --root_path ../../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 0.8 0.6 0.4 0.2 --batch_size 32 --epochs 10 --cuda_devices 0 | parallel -j1
popd

```

### Ablation Study
```bash
pushd multitask

# w/o Self-Attn
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 42 43 44 --no_multitask_attn --cuda_devices 0 | parallel -j1

# w/o Koopman
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 --epochs 150 --patience 50 --batch_size 4 --loss_type "masked_mae" --learning_rate 1e-4 --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 42 43 44 --cuda_devices 0 | parallel -j1

# w/o ups/ds
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0 | parallel -j1

popd
```

## Reproduce Result Tables

We use `results/gen_exp_results.ipynb` to print result tables in LaTeX format. 

To reproduce the tables, replace paths in the `res_configs_dict` in `results/metric_utils.py` with paths storing corresponding experimental results.
