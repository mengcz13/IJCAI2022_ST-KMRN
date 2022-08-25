from argparse import ArgumentParser


commands = [
    r'python main_informer.py --model {model} --data {data} --root_path {root_path} --attn {attn} --features M --seq_len {seq_len} --pred_len {pred_len} --label_len {pred_len} --keep_ratio {keep_ratio} --batch_size {batch_size} --train_epochs {train_epochs} --patience {patience} --num_workers {num_workers} --seq_diff {seq_diff} --test_seq_diff {test_seq_diff} --target_res {target_res} --resolution_type {resolution_type} --expname_suffix "res-{target_res}-{resolution_type}" --chrono_arr_cats {chrono_arr_cats} --seed {seed}', # all resolutions at the same time
    # r'python main_informer.py --model {model} --data {data} --root_path {root_path} --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 2 --train_epochs 50 --num_workers {num_workers} --seq_diff 48 --target_res "30min" --expname_suffix "res-30min" --seed {seed}', # 30min resolutions at the same time
    # r'python main_informer.py --model {model} --data {data} --root_path {root_path} --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 2 --train_epochs 50 --num_workers {num_workers} --seq_diff 48 --target_res "6-hour" --expname_suffix "res-6hour" --seed {seed}', # 6hour resolutions at the same time
    # r'python main_informer.py --model {model} --data {data} --root_path {root_path} --attn prob --features M --seq_len 1440 --pred_len 480 --label_len 480 --batch_size 2 --train_epochs 50 --num_workers {num_workers} --seq_diff 48 --target_res "day" --expname_suffix "res-day" --seed {seed}', # day resolutions at the same time
]

parser = ArgumentParser()
parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
parser.add_argument('--cuda_devices', nargs='+', type=int, default=[0])
parser.add_argument('--model', type=str, default='informer')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--data', type=str, default='NYCTaxi_Multires_6h-1d')
parser.add_argument('--root_path', type=str, default='../../data/nyc_taxi/manhattan')
parser.add_argument('--with_ts_delta', action='store_true')
parser.add_argument('--no_input_mask', action='store_true')
parser.add_argument('--single_input_res', action='store_true')
parser.add_argument('--target_res', type=str, nargs='+', default=['all', '30min', '6-hour', 'day'])
parser.add_argument('--seq_len', type=int, default=1440)
parser.add_argument('--pred_len', type=int, default=480)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--resolution_type', type=str, default='agg')
parser.add_argument('--seq_diff', type=int, default=48)
parser.add_argument('--test_seq_diff', type=int, default=48)
parser.add_argument('--attn', type=str, default='prob')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--train_epochs', type=int, default=50)
parser.add_argument('--keep_ratio', nargs='+', type=float, default=[0.8])
parser.add_argument('--chrono_arr_cats', type=str, default='mdwh')

args = parser.parse_args()

cuda_devices = args.cuda_devices

formatted_commands = []
for seed in args.seeds:
    for command in commands:
        for keep_ratio in args.keep_ratio:
            for target_res in args.target_res:
                # for seed in seeds:
                    if args.model == 'gru':
                        command_i = command + ' --e_layers 2 --d_layers 2'
                    else:
                        command_i = command
                    if args.with_ts_delta:
                        command_i = command_i + ' --with_ts_delta'
                    if args.no_input_mask:
                        command_i += ' --no_input_mask'
                    if args.single_input_res:
                        command_i += ' --single_input_res'
                    formatted_commands.append(command_i.format(attn=args.attn, model=args.model, seed=seed, data=args.data, root_path=args.root_path, num_workers=args.num_workers, seq_len=args.seq_len, pred_len=args.pred_len, keep_ratio=keep_ratio, target_res=target_res, patience=args.patience, resolution_type=args.resolution_type, seq_diff=args.seq_diff, test_seq_diff=args.test_seq_diff, batch_size=args.batch_size, train_epochs=args.train_epochs, chrono_arr_cats=args.chrono_arr_cats))
formatted_commands = ['CUDA_VISIBLE_DEVICES={} '.format(cuda_devices[k % len(cuda_devices)]) + c for k, c in enumerate(formatted_commands)]
print('\n'.join(formatted_commands))