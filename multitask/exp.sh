# test MTGNN+Multitask on NYCTaxi, Solar Energy (AR) in old avg res setting
python exp_train_targets_repeat.py --model mtgnn --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --cuda_devices 0
python exp_train_targets_repeat.py --model mtgnn --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --cuda_devices 0

# test GraphWaveNet+Multitask on NYCTaxi, Solar Energy (AR) in old avg res setting
python exp_train_targets_repeat.py --model gw --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --cuda_devices 0
## with larger rf (05-10-2021)
python exp_train_targets_repeat.py --model gw --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --larger_rf --batch_size 2 --patience 50 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --larger_rf --batch_size 1 --patience 50 --cuda_devices 0

# test GraphWaveNet+Multitask on ECL in old avg res setting
python exp_train_targets_repeat.py --model gw --root_path ../data/ecl --target_res all --seq_len 720 --pred_len 240 --seq_diff 24 --batch_size 1 --cuda_devices 0

# test GraphWaveNet+Multitask on PeMS-BAY in sampling res setting
python exp_train_targets_repeat.py --model gw --root_path ../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --batch_size 8 --epochs 50 --cuda_devices 0

# test downsampling
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --root_path ../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --batch_size 8 --epochs 50 --cuda_devices 0

# test downsampling + upsampling
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --batch_size 8 --epochs 50 --cuda_devices 0
## fine-tune (05-12-2021)
### lr 5e-4
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --learning_rate 5e-4 --patience 30 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --learning_rate 5e-4 --patience 30 --cuda_devices 0
### with lr_scheduler on plateau
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --lr_scheduler --patience 25 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --lr_scheduler --patience 25 --cuda_devices 0
### further tune of nyctaxi exps (05-13-2021)
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --learning_rate 1e-3 --patience 30 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 1 --learning_rate 1e-3 --patience 30 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --learning_rate 1e-4 --patience 50 --cuda_devices 0
### further tune of solar energy (05-14-2021)
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --learning_rate 1e-4 --patience 30 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 1 --learning_rate 1e-3 --patience 30 --cuda_devices 0
### loss type changed to MSE
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --patience 30 --loss_type masked_mse --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --learning_rate 5e-4 --patience 30 --loss_type masked_mse --cuda_devices 0

### test conv fusion of upsampling/downsampling
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --updown_fusion "conv" --agg_res_hdim 32 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 4 --learning_rate 1e-4 --patience 50 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --updown_fusion "conv" --agg_res_hdim 32 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --learning_rate 5e-4 --patience 30 --cuda_devices 0


# test unet upsampler
python exp_train_targets_repeat.py --model gw --upsampler unet --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 2 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --upsampler unet --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 1 --cuda_devices 0
python exp_train_targets_repeat.py --model gw --upsampler unet --use_downsampling_pred --use_upsampling_pred --root_path ../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --batch_size 8 --epochs 50 --cuda_devices 0


# try compositional koopman branch
## one resolution (30min)
### Graph-WaveNet, mse loss
python exp_train_targets_repeat.py --model gw --root_path ../data/nyc_taxi/manhattan --target_res "30min" --single_res_input_output --seq_len 1440 --pred_len 480 --epochs 30 --batch_size 4 --loss_type "masked_mse" --cuda_devices 0
python exp_train_targets_repeat.py --model gw --root_path ../data/solar_energy --target_res "30min" --single_res_input_output --seq_len 1440 --pred_len 480 --epochs 150 --batch_size 2 --loss_type "masked_mse" --cuda_devices 0
python exp_train_targets_repeat.py --model gw --root_path ../data/pems/PEMS-BAY --target_res "30min" --single_res_input_output --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --epochs 30 --batch_size 8 --loss_type "masked_mse" --cuda_devices 0

### Graph-WaveNet + CKO, l1 loss
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res "30min" --single_res_input_output --seq_len 480 --pred_len 480 --epochs 50 --batch_size 1 --loss_type "masked_mae" --g_dim 32 --pstep 2 --nf_relation 8 --nf_particle 8 --nf_effect 8 --fit_type "structured" --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/solar_energy --target_res "30min" --single_res_input_output --seq_len 240 --pred_len 480 --epochs 150 --batch_size 1 --loss_type "masked_mae" --g_dim 32 --pstep 2 --nf_relation 8 --nf_particle 8 --nf_effect 8 --fit_type "structured" --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res "30min" --single_res_input_output --seq_len 480 --pred_len 480 --epochs 50 --batch_size 1 --loss_type "masked_mae" --g_dim 32 --pstep 2 --nf_relation 8 --nf_particle 8 --nf_effect 8 --fit_type "structured" --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/solar_energy --target_res "30min" --single_res_input_output --seq_len 1440 --pred_len 480 --epochs 150 --batch_size 2 --loss_type "masked_mse" --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/pems/PEMS-BAY --target_res "30min" --single_res_input_output --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --epochs 30 --batch_size 8 --loss_type "masked_mse" --cuda_devices 0

### Graph-WaveNet + CKO + MLP enc/dec, l1 loss
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res "30min" --single_res_input_output --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/solar_energy --target_res "30min" --single_res_input_output --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --cuda_devices 0
#### Graph-WaveNet + CKO + MLP enc/dec, l1 loss, weight decay tune
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --weight_decay 1e-5 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --weight_decay 1e-5 --cuda_devices 0

python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --weight_decay 1e-3 --cuda_devices 0
#### tune cko_lrate, cko_wdecay
# python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --learning_rate 1e-3 --weight_decay 1e-4 --cko_lrate 1e-4 --cko_wdecay 0 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --learning_rate 5e-4 --weight_decay 1e-4 --cko_lrate 1e-4 --cko_wdecay 0 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --learning_rate 1e-4 --weight_decay 1e-4 --cko_lrate 1e-4 --cko_wdecay 0 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --learning_rate 5e-5 --weight_decay 1e-4 --cko_lrate 1e-4 --cko_wdecay 0 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --learning_rate 5e-5 --weight_decay 1e-4 --cko_lrate 1e-5 --cko_wdecay 1e-4 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 20 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --learning_rate 5e-5 --weight_decay 1e-4 --cko_lrate 5e-6 --cko_wdecay 1e-4 --cuda_devices 0

#### Graph-WaveNet + CKO + MLP enc/dec, with gate controling DL and CKO ratio for prediction
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
##### GW + CKO + MLP enc/dec + gate + ups + ds
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
##### GW + CKO + MLP enc/dec + gate + ups + ds + conv fusion
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0
##### MAE loss for solar_energy
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0


### Graph-WaveNet + KAE, l1 loss
python exp_train_targets_repeat.py --model gw_kae --root_path ../data/nyc_taxi/manhattan --target_res "30min" --single_res_input_output --seq_len 1440 --pred_len 480 --epochs 50 --batch_size 2 --loss_type "masked_mae" --alpha 8 --kae_lr 1e-4 --bottleneck 128 --cuda_devices 0


## MultiresAttn commands
python train_gw.py --upsampler simple --gcn_bool --adjtype doubletransition --addaptadj --randomadj --num_levels 3 --data ../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --seq_diff 48 --batch_size 1 --epochs 150 --patience -1 --target_res all --resolution_type agg --loss_type masked_mae --seed 42

python exp_train_targets_repeat.py --model multires_attn --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --batch_size 1 --patience 30 --cuda_devices 0
python exp_train_targets_repeat.py --model multires_attn --root_path ../data/solar_energy --target_res all --seq_len 1440 --pred_len 480 --batch_size 1 --patience 30 --cuda_devices 0


# GW + CKO + Gate, GW + CKO + ups/ds fusion, PEMS-BAY
python exp_train_targets_repeat.py --model gw --use_downsampling_pred --use_upsampling_pred --root_path ../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --batch_size 8 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --epochs 150 --patience 30 --batch_size 8 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --epochs 150 --patience 30 --batch_size 8 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/pems/PEMS-BAY --target_res all --resolution_type sample --seq_len 36 --pred_len 36 --seq_diff 1 --epochs 150 --patience 30 --batch_size 8 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0


# NYCTaxi Green
## GW + CKO + Gate
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
## GW + CKO + Gate + ups/ds fusion
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0

# NYCTaxi FHV, Manhattan
## GW + CKO + Gate
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan_fhv --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
## GW + CKO + Gate + ups/ds fusion
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan_fhv --target_res all --seq_len 1440 --pred_len 480 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0

# NYCTaxi, NYCTaxi(Green), fully observed data
## GW + CKO + Gate
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
## GW + CKO + Gate + ups/ds fusion
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0
## GW + CKO + Gate + ups/ds fusion, MAE loss
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0
## GW + CKO + Gate
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
## GW + CKO + Gate + ups/ds fusion
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0
## GW + CKO + Gate + ups/ds fusion, MAE loss
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0


# Solar Energy, 10min
## GW + CKO + Gate
python exp_train_targets_repeat.py --model gw_cko --root_path ../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --cuda_devices 0
## GW + CKO + Gate + ups/ds fusion
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/solar_energy_10min --target_res all --seq_len 1440 --pred_len 432 --seq_diff 36 --test_seq_diff 36 --keep_ratio 1 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --cuda_devices 0

# NYCTaxi, NYCTaxi(Green), 0.6/0.4/0.2 obs rate
{ python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 42 --cuda_devices 3 ; python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 42 --cuda_devices 3 } | cat 
## more seeds (43/44) (12 exps -> 48h)
# { python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 43 44 --cuda_devices 3 ; python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 43 44 --cuda_devices 3 } | cat 
{ python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 43 44 --cuda_devices 0 ; python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.6 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 43 44 --cuda_devices 0 } | cat 
{ python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.4 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 43 44 --cuda_devices 1 ; python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.4 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 43 44 --cuda_devices 1 } | cat 
{ python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 43 44 --cuda_devices 2 ; python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 43 44 --cuda_devices 2 } | cat 


# NYCTaxi, NYCTaxi(Green), 0.8/0.6/0.4/0.2, mae loss (24 exps -> 96h)
{ python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 42 43 44 --cuda_devices 0 ; python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/sel_green --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 0.6 0.4 0.2 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mae" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 42 43 44 --cuda_devices 0 } | cat 


# ablation study: NYCTaxi, 0.8, mse loss, no multitask
python exp_train_targets_repeat.py --model gw_cko --use_downsampling_pred --use_upsampling_pred --root_path ../data/nyc_taxi/manhattan --target_res all --seq_len 1440 --pred_len 480 --keep_ratio 0.8 --epochs 150 --patience 30 --batch_size 2 --loss_type "masked_mse" --enc_dec_type mlp --fit_type "unstructured" --g_dim 128 --chrono_arr_cats "mdwhm" --chrono_sincos_emb --cko_dl_gate --cko_dl_gate_hdim 32 --updown_fusion "conv" --agg_res_hdim 32 --seed 42 43 44 --no_multitask_attn --cuda_devices 0