from argparse import ArgumentParser


commands_dict = {
    # 'mtgnn': [
    #     r'python train_mtgnn.py --data {root_path} --seq_in_len {seq_len} --seq_out_len {pred_len} --seq_diff {seq_diff} --batch_size {batch_size} --epochs {epochs} --target_res {target_res} --resolution_type {resolution_type} --expname_suffix "res-{target_res}-{resolution_type}" --seed {seed}', # all resolutions at the same time
    # ],
    'gw': [
        r'python train_gw.py --upsampler {upsampler} --gcn_bool --adjtype doubletransition --addaptadj --randomadj{use_downsampling_pred}{use_upsampling_pred} --data {root_path} --seq_in_len {seq_len} --seq_out_len {pred_len} --seq_diff {seq_diff} --keep_ratio {keep_ratio} --batch_size {batch_size} --epochs {epochs} --patience {patience} --target_res {target_res}{single_res_input_output} --resolution_type {resolution_type} --loss_type {loss_type}{larger_rf} --learning_rate {learning_rate}{lr_scheduler} --chrono_arr_cats {chrono_arr_cats}{chrono_sincos_emb} --updown_fusion {updown_fusion} --agg_res_hdim {agg_res_hdim} --seed {seed}'
    ],
    # 'multires_attn': [
    #     r'python train_multires_attn.py --upsampler {upsampler} --gcn_bool --adjtype doubletransition --addaptadj --randomadj --num_levels {num_levels}{use_downsampling_pred}{use_upsampling_pred} --data {root_path} --seq_in_len {seq_len} --seq_out_len {pred_len} --seq_diff {seq_diff} --batch_size {batch_size} --epochs {epochs} --patience {patience} --target_res {target_res}{single_res_input_output} --resolution_type {resolution_type} --loss_type {loss_type}{larger_rf} --seed {seed}'
    # ],
    'gw_cko': [
        r'python train_gw_cko.py --upsampler {upsampler} --gcn_bool --adjtype doubletransition --addaptadj --randomadj{use_downsampling_pred}{use_upsampling_pred} --data {root_path} --seq_in_len {seq_len} --seq_out_len {pred_len} --seq_diff {seq_diff} --test_seq_diff {test_seq_diff} --keep_ratio {keep_ratio} --batch_size {batch_size} --epochs {epochs} --patience {patience} --target_res {target_res}{single_res_input_output} --resolution_type {resolution_type} --loss_type {loss_type} --g_dim {g_dim} --pstep {pstep} --nf_relation {nf_relation} --nf_particle {nf_particle} --nf_effect {nf_effect} --fit_type {fit_type} --enc_dec_type {enc_dec_type} --learning_rate {learning_rate} --weight_decay {weight_decay} --cko_lrate {cko_lrate} --cko_wdecay {cko_wdecay} --chrono_arr_cats {chrono_arr_cats}{chrono_sincos_emb}{cko_dl_gate} --cko_dl_gate_hdim {cko_dl_gate_hdim} --updown_fusion {updown_fusion} --agg_res_hdim {agg_res_hdim}{no_multitask_attn} --seed {seed}'
    ],
    # 'gw_kae': [
    #     r'python train_gw_kae.py --upsampler {upsampler} --gcn_bool --adjtype doubletransition --addaptadj --randomadj{use_downsampling_pred}{use_upsampling_pred} --data {root_path} --seq_in_len {seq_len} --seq_out_len {pred_len} --seq_diff {seq_diff} --batch_size {batch_size} --epochs {epochs} --patience {patience} --target_res {target_res}{single_res_input_output} --resolution_type {resolution_type} --loss_type {loss_type} --alpha {alpha} --kae_lr {kae_lr} --bottleneck {bottleneck} --backward 1 --print_every 10 --seed {seed}'
    # ]
}


# commands = [
#     r'python train_{model}.py --data {root_path} --seq_in_len {seq_len} --seq_out_len {pred_len} --seq_diff {seq_diff} --batch_size {batch_size} --epochs {epochs} --target_res {target_res} --resolution_type {resolution_type} --expname_suffix "res-{target_res}-{resolution_type}" --seed {seed}', # all resolutions at the same time
#     # r'python train_multi_step.py --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --seq_diff 48 --batch_size 4 --epochs 150 --target_res "30min" --expname_suffix "res-30min" --seed {}', # 30min resolutions at the same time
#     # r'python train_multi_step.py --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --seq_diff 48 --batch_size 4 --epochs 150 --target_res "6-hour" --expname_suffix "res-6hour" --seed {}', # 6hour resolutions at the same time
#     # r'python train_multi_step.py --data ../../data/nyc_taxi/manhattan --seq_in_len 1440 --seq_out_len 480 --seq_diff 48 --batch_size 4 --epochs 150 --target_res "day" --expname_suffix "res-day" --seed {}', # day resolutions at the same time
# ]

parser = ArgumentParser()
parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
parser.add_argument('--model', type=str, default='mtgnn', help='mtgnn/gw')
parser.add_argument('--cuda_devices', nargs='+', type=int, default=[0])
parser.add_argument('--root_path', type=str, default='../../data/nyc_taxi/manhattan')
parser.add_argument('--target_res', type=str, nargs='+', default=['all', '30min', '6-hour', 'day'])
parser.add_argument('--seq_len', type=int, default=1440)
parser.add_argument('--pred_len', type=int, default=480)
parser.add_argument('--keep_ratio', nargs='+', type=float, default=[0.8])
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--patience', type=int, default=-1)
parser.add_argument('--resolution_type', type=str, default='agg')
parser.add_argument('--seq_diff', type=int, default=48)
parser.add_argument('--test_seq_diff', type=int, default=48)
parser.add_argument('--use_downsampling_pred', action='store_true')
parser.add_argument('--use_upsampling_pred', action='store_true')
parser.add_argument('--upsampler', type=str, default='simple')
parser.add_argument('--single_res_input_output', action='store_true')
parser.add_argument('--loss_type', type=str, default='masked_mae')
parser.add_argument('--larger_rf', action='store_true')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--updown_fusion', type=str, default='param', help='param: c parameters for c sources; conv: b x t x n x c params from conv for c sources')
parser.add_argument('--agg_res_hdim', type=int, default=32)
# CKO params
parser.add_argument('--cko_lrate', type=float, default=1e-4)
parser.add_argument('--cko_wdecay', type=float, default=0)
parser.add_argument('--g_dim', type=int, default=32)
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--nf_relation', type=int, default=8)
parser.add_argument('--nf_particle', type=int, default=8)
parser.add_argument('--nf_effect', type=int, default=8)
parser.add_argument('--fit_type', default='structured')
parser.add_argument('--enc_dec_type', default='pn')
parser.add_argument('--chrono_arr_cats', type=str, default='mdwh', help='mdwh/mdwhm')
parser.add_argument('--chrono_sincos_emb', action='store_true')
parser.add_argument('--cko_dl_gate', action='store_true')
parser.add_argument('--cko_dl_gate_hdim', type=int, default=32)
# KAE params
parser.add_argument('--kae_lr', type=float, default=1e-3)
parser.add_argument('--kae_wd', type=float, default=0.0)
parser.add_argument('--kae_gradclip', type=float, default=0.05)
parser.add_argument('--kae_lr_decay', type=float, default=0.2)
parser.add_argument('--bottleneck', type=int, default='6',  help='size of bottleneck layer')
parser.add_argument('--backward', type=int, default=0, help='train with backward dynamics')
parser.add_argument('--alpha', type=int, default='1',  help='model width')
parser.add_argument('--init_scale', type=float, default=0.99, help='init scaling')
parser.add_argument('--lamb', type=float, default='1',  help='balance between reconstruction and prediction loss')
parser.add_argument('--nu', type=float, default='1e-1',  help='tune backward loss')
parser.add_argument('--eta', type=float, default='1e-2',  help='tune consistent loss')
# multires_attn params
parser.add_argument('--num_levels', type=int, default=1)
# ablation study
parser.add_argument('--no_multitask_attn', action='store_true')

args = parser.parse_args()

cuda_devices = args.cuda_devices

commands = commands_dict[args.model]

formatted_commands = []
for seed in args.seeds:
    for command in commands:
        for keep_ratio in args.keep_ratio:
            for target_res in args.target_res:
                    formatted_commands.append(command.format(
                        seed=seed, root_path=args.root_path, seq_len=args.seq_len,
                        pred_len=args.pred_len, keep_ratio=keep_ratio,
                        batch_size=args.batch_size,
                        target_res=target_res, epochs=args.epochs, patience=args.patience,
                        resolution_type=args.resolution_type,
                        seq_diff=args.seq_diff, test_seq_diff=args.test_seq_diff,
                        use_downsampling_pred=(' --use_downsampling_pred' if args.use_downsampling_pred else ''),
                        use_upsampling_pred=(' --use_upsampling_pred' if args.use_upsampling_pred else ''),
                        upsampler=args.upsampler,
                        single_res_input_output=(' --single_res_input_output' if args.single_res_input_output else ''),
                        loss_type=args.loss_type,
                        larger_rf=(' --larger_rf' if args.larger_rf else ''),
                        g_dim=args.g_dim,
                        pstep=args.pstep,
                        nf_relation=args.nf_relation,
                        nf_particle=args.nf_particle,
                        nf_effect=args.nf_effect,
                        fit_type=args.fit_type,
                        enc_dec_type=args.enc_dec_type,
                        alpha=args.alpha,
                        kae_lr=args.kae_lr,
                        bottleneck=args.bottleneck,
                        num_levels=args.num_levels,
                        learning_rate=args.learning_rate,
                        lr_scheduler=(' --lr_scheduler' if args.lr_scheduler else ''),
                        weight_decay=args.weight_decay,
                        cko_lrate=args.cko_lrate,
                        cko_wdecay=args.cko_wdecay,
                        chrono_arr_cats=args.chrono_arr_cats,
                        chrono_sincos_emb=(' --chrono_sincos_emb' if args.chrono_sincos_emb else ''),
                        cko_dl_gate=(' --cko_dl_gate' if args.cko_dl_gate else ''),
                        cko_dl_gate_hdim=args.cko_dl_gate_hdim,
                        updown_fusion=args.updown_fusion,
                        agg_res_hdim=args.agg_res_hdim,
                        no_multitask_attn=(' --no_multitask_attn' if args.no_multitask_attn else '')
                    ))
formatted_commands = ['CUDA_VISIBLE_DEVICES={} '.format(cuda_devices[k % len(cuda_devices)]) + c for k, c in enumerate(formatted_commands)]
print('\n'.join(formatted_commands))