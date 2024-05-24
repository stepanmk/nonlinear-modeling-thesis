from nlmodel.loaders import PedalLoader
from nlmodel.nets.pedal import PedalModel
from nlmodel.trainers.basic import BasicTruncTrainer

net = PedalModel(pedal="ds1", freeze_cond_block=False)

file_names_train: tuple = (
    "T0-speakerout.wav",
    "T5-speakerout.wav",
    "T10-speakerout.wav",
)

file_names_val: tuple = (
    "T0-speakerout.wav",
    "T5-speakerout.wav",
    "T10-speakerout.wav",
)

file_names_test: tuple = (
    "T0-speakerout.wav",
    "T2-speakerout.wav",
    "T4-speakerout.wav",
    "T6-speakerout.wav",
    "T8-speakerout.wav",
    "T10-speakerout.wav",
)

train = PedalLoader(
    data_dir="./dataset/ds1/",
    file_names=file_names_train,
    start_sec=0.0,
    end_sec=240.0,
    seg_len=0.5,
    batch_size=80,
    shuffle=True,
    drop_last=False,
    use_gain_value=False,
)

val = PedalLoader(
    data_dir="./dataset/ds1/",
    file_names=file_names_val,
    start_sec=240.0,
    end_sec=300.0,
    seg_len=5,
    batch_size=12,
    shuffle=False,
    use_gain_value=False,
)

test = PedalLoader(
    data_dir="./dataset/ds1/",
    file_names=file_names_test,
    start_sec=300.0,
    end_sec=360.0,
    seg_len=5,
    batch_size=12,
    shuffle=False,
    use_gain_value=False,
)


tr = BasicTruncTrainer(
    net=net,
    trunc_steps=2048,
    val_steps=2048,
    max_epochs=150,
    lr=0.002,
    lr_reduce_patience=5,
    warmup=True,
    warmup_steps=2048,
    max_norm=5,
    train_loader=train.get_loader(),
    val_loader=val.get_loader(),
    test_loader=test.get_loader(),
    logs_path="./pedal_logs/",
    run_name="3_cond_big",
)

tr.train()
