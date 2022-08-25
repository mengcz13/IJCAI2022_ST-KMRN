CUDA_VISIBLE_DEVICES=2 python driver_multires.py \
--data ../../data/nyc_taxi/manhattan \
--seq_in_len 1440 \
--seq_out_len 480 \
--seq_diff 48 \
--single_res_input_output \
--alpha 8 --lr 1e-3 --epochs 1 --batch 64 --batch_test 64 --folder test/multires --steps 480 --steps_back 480 --bottleneck 128 --backward 1 --pred_steps 480 --seed 42