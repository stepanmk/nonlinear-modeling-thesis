from nlmodel.loaders import AmpLoader
from nlmodel.nets.amp import AmpModel, AmpModelGain
from nlmodel.trainers.basic import BasicTruncTrainer

net = AmpModelGain(freeze_cond_block=True)

file_names: tuple = (
    "B5_M5_T5_G1/B5_M5_T5_G1-speakerout.wav",
    "B5_M5_T5_G2/B5_M5_T5_G2-speakerout.wav",
    "B5_M5_T5_G4/B5_M5_T5_G4-speakerout.wav",
    "B5_M5_T5_G5/B5_M5_T5_G5-speakerout.wav",
    "B5_M5_T5_G6/B5_M5_T5_G6-speakerout.wav",
    "B5_M5_T5_G8/B5_M5_T5_G8-speakerout.wav",
    "B5_M5_T5_G10/B5_M5_T5_G10-speakerout.wav",
)

train = AmpLoader(
    data_dir="./dataset/gain/",
    file_names=file_names,
    start_sec=0.0,
    end_sec=240.0,
    seg_len=0.5,
    batch_size=80,
    shuffle=True,
    drop_last=False,
    use_gain_value=True,
)

val = AmpLoader(
    data_dir="./dataset/gain/",
    file_names=file_names,
    start_sec=240.0,
    end_sec=300.0,
    seg_len=5,
    batch_size=12,
    shuffle=False,
    use_gain_value=True,
)

test = AmpLoader(
    data_dir="./dataset/gain/",
    file_names=file_names,
    start_sec=300.0,
    end_sec=360.0,
    seg_len=5,
    batch_size=12,
    shuffle=False,
    use_gain_value=True,
)


tr = BasicTruncTrainer(
    net=net,
    trunc_steps=2048,
    val_steps=2048,
    max_epochs=350,
    lr=0.002,
    lr_reduce_patience=5,
    warmup=True,
    clip_grad=True,
    clip_method="auto",
    warmup_steps=2048,
    max_norm=5,
    train_loader=train.get_loader(),
    val_loader=val.get_loader(),
    test_loader=test.get_loader(),
    logs_path="./amp_logs/",
    run_name="gain_dataset",
)


tr.train()
