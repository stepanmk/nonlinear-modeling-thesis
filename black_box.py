import argparse

from nlmodel.loaders import SingleFileLoader
from nlmodel.nets.rnn import TCNRNN, LinRNN, RNNBase
from nlmodel.trainers.basic import BasicTruncTrainer

prsr = argparse.ArgumentParser()
prsr.add_argument("--hidden_size", "-hs", default=16, type=int)
prsr.add_argument("--max_epochs", "-me", default=400, type=int)
prsr.add_argument("--model_type", "-mt", default="base", type=str)
prsr.add_argument("--cell_type", "-ct", default="GRU", type=str)
prsr.add_argument("--pre_size", "-ps", default=4, type=int)
prsr.add_argument("--run_name", "-rn", default="autoclip", type=str)
prsr.add_argument("--device_name", "-dn", default="ht1", type=str)
prsr.add_argument("--clip_grad", "-cg", default=1, type=int)
prsr.add_argument("--clip_method", "-cm", default="norm", type=str)
prsr.add_argument("--max_norm", "-mn", default=1, type=int)
prsr.add_argument("--jvm_gain", "-jg", default=8, type=int)

args = prsr.parse_args()

if __name__ == "__main__":
    g = str(args.jvm_gain)
    if args.device_name == "jvm":
        data_dir = f"./dataset/final/B5_M5_T5_G{g}/"
        inp_file = f"B5_M5_T5_G{g}-input.wav"
        tgt_file = f"B5_M5_T5_G{g}-speakerout.wav"
        train_start_sec = 0.0
        train_end_sec = 240.0
        val_start_sec = 240.0
        val_end_sec = 300.0
        test_start_sec = 300.0
        test_end_sec = 360.0
    else:
        data_dir = "./dataset/"
        inp_file = "ht1-input.wav"
        tgt_file = "ht1-target.wav"
        train_start_sec = 0.0
        train_end_sec = 315.0
        val_start_sec = 315.0
        val_end_sec = 375.0
        test_start_sec = 375.0
        test_end_sec = 435.0

    train = SingleFileLoader(
        data_dir=data_dir,
        inp_file=inp_file,
        tgt_file=tgt_file,
        start_sec=train_start_sec,
        end_sec=train_end_sec,
        seg_len=0.5,
        batch_size=40,
        shuffle=True,
        drop_last=(True if args.device_name == "ht1" else False),
    )

    val = SingleFileLoader(
        data_dir=data_dir,
        inp_file=inp_file,
        tgt_file=tgt_file,
        start_sec=val_start_sec,
        end_sec=val_end_sec,
        seg_len=5,
        batch_size=12,
        shuffle=False,
    )

    test = SingleFileLoader(
        data_dir=data_dir,
        inp_file=inp_file,
        tgt_file=tgt_file,
        start_sec=test_start_sec,
        end_sec=test_end_sec,
        seg_len=5,
        batch_size=12,
        shuffle=False,
    )

    if args.model_type == "lin":
        net = LinRNN(
            cell_type=args.cell_type,
            hidden_size=args.hidden_size,
            global_skip=True,
            lin_size=args.pre_size,
        )
    elif args.model_type == "tcn":
        net = TCNRNN(
            cell_type=args.cell_type,
            hidden_size=args.hidden_size,
            global_skip=True,
            num_layers=args.pre_size,
        )
    else:
        net = RNNBase(
            cell_type=args.cell_type,
            hidden_size=args.hidden_size,
            skip=True,
        )

    tr = BasicTruncTrainer(
        net=net,
        trunc_steps=2048,
        max_epochs=args.max_epochs,
        warmup=False if args.model_type == "tcn" else True,
        clip_grad=bool(args.clip_grad),
        max_norm=args.max_norm,
        clip_method=args.clip_method,
        cond=False,
        train_loader=train.get_loader(),
        val_loader=val.get_loader(),
        test_loader=test.get_loader(),
        run_name=f"{args.device_name}_{args.run_name}",
        logs_path="./custom_logs/",
    )

    tr.train()
