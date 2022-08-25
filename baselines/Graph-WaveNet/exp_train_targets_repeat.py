from argparse import ArgumentParser


commands = [
    r'python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data {root_path} --seq_in_len {seq_len} --seq_out_len {pred_len} --seq_diff {seq_diff} --test_seq_diff {test_seq_diff} --keep_ratio {keep_ratio} --batch_size {batch_size} --epochs {epochs} --target_res {target_res} --resolution_type {resolution_type} --blocks {blocks} --layers {layers} --kernel_size {kernel_size} --patience {patience} --seed {seed}',
    # r'python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 150 --seq_diff 48 --target_res "30min" --seed {}',
    # r'python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 150 --seq_diff 48 --target_res "6-hour" --seed {}',
    # r'python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --batch_size 2 --epochs 150 --seq_diff 48 --target_res "day" --seed {}',
]

parser = ArgumentParser()
parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
parser.add_argument('--cuda_devices', nargs='+', type=int, default=[0])
parser.add_argument('--root_path', type=str, default='../../data/nyc_taxi/manhattan')
parser.add_argument('--target_res', type=str, nargs='+', default=['all', '30min', '6-hour', 'day'])
parser.add_argument('--seq_len', type=int, default=1440)
parser.add_argument('--pred_len', type=int, default=480)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--resolution_type', type=str, default='agg')
parser.add_argument('--seq_diff', type=int, default=48)
parser.add_argument('--test_seq_diff', type=int, default=48)
parser.add_argument('--blocks', type=int, default=4)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--patience', type=int, default=-1, help='work when set to positive')
parser.add_argument('--keep_ratio', nargs='+', type=float, default=[0.8])

args = parser.parse_args()

cuda_devices = args.cuda_devices

formatted_commands = []
for seed in args.seeds:
    for command in commands:
        for keep_ratio in args.keep_ratio:
            for target_res in args.target_res:
                    formatted_commands.append(command.format(
                        seed=seed, root_path=args.root_path, seq_len=args.seq_len,
                        pred_len=args.pred_len, batch_size=args.batch_size,
                        target_res=target_res, epochs=args.epochs, resolution_type=args.resolution_type,
                        seq_diff=args.seq_diff, test_seq_diff=args.test_seq_diff, keep_ratio=keep_ratio,
                        blocks=args.blocks, layers=args.layers, kernel_size=args.kernel_size, patience=args.patience
                    ))
formatted_commands = ['CUDA_VISIBLE_DEVICES={} '.format(cuda_devices[k % len(cuda_devices)]) + c for k, c in enumerate(formatted_commands)]
print('\n'.join(formatted_commands))