import subprocess


def run_config(d):
    cmd = (
        f"C:/Users/Stepan/miniconda3/envs/nonlinear-modeling/python black_box.py"
        f' --hidden_size {d["hidden_size"]}'
        f' --max_epochs {d["max_epochs"]}'
        f' --model_type {d["model_type"]}'
        f' --cell_type {d["cell_type"]}'
        f' --pre_size {d["pre_size"]}'
        f' --run_name {d["run_name"]}'
        f' --device_name {d["device_name"]}'
        f' --clip_grad {d["clip_grad"]}'
        f' --max_norm {d["max_norm"]}'
        f' --clip_method {d["clip_method"]}'
        f' --jvm_gain {d["jvm_gain"]}'
    )
    subprocess.run(cmd)


me = 550

for hs in [16, 32, 48, 64, 96]:
    for ct in ["GRU", "LSTM"]:
        for dn in ["jvm", "ht1"]:
            cfg = {
                "hidden_size": hs,
                "max_epochs": me,
                "model_type": "base",
                "cell_type": ct,
                "pre_size": 1,
                "run_name": "no_clip_2",
                "device_name": dn,
                "clip_grad": 0,
                "max_norm": 1,
                "jvm_gain": 8,
                "clip_method": "norm",
            }
            run_config(cfg)


for hs in [16, 32, 48, 64, 96]:
    for ct in ["GRU", "LSTM"]:
        for dn in ["jvm", "ht1"]:
            cfg = {
                "hidden_size": hs,
                "max_epochs": me,
                "model_type": "base",
                "cell_type": ct,
                "pre_size": 1,
                "run_name": "no_clip_3",
                "device_name": dn,
                "clip_grad": 0,
                "max_norm": 1,
                "jvm_gain": 8,
                "clip_method": "norm",
            }
            run_config(cfg)


for hs in [16, 32, 48, 64, 96]:
    for ct in ["GRU", "LSTM"]:
        for dn in ["jvm", "ht1"]:
            cfg = {
                "hidden_size": hs,
                "max_epochs": me,
                "model_type": "base",
                "cell_type": ct,
                "pre_size": 1,
                "run_name": "clip_2",
                "device_name": dn,
                "clip_grad": 1,
                "max_norm": 1,
                "jvm_gain": 8,
                "clip_method": "norm",
            }
            run_config(cfg)


for hs in [16, 32, 48, 64, 96]:
    for ct in ["GRU", "LSTM"]:
        for dn in ["jvm", "ht1"]:
            cfg = {
                "hidden_size": hs,
                "max_epochs": me,
                "model_type": "base",
                "cell_type": ct,
                "pre_size": 1,
                "run_name": "clip_3",
                "device_name": dn,
                "clip_grad": 1,
                "max_norm": 1,
                "jvm_gain": 8,
                "clip_method": "norm",
            }
            run_config(cfg)


for hs in [16, 32, 48, 64, 96]:
    for ct in ["GRU", "LSTM"]:
        for dn in ["jvm", "ht1"]:
            cfg = {
                "hidden_size": hs,
                "max_epochs": me,
                "model_type": "base",
                "cell_type": ct,
                "pre_size": 1,
                "run_name": "autoclip_2",
                "device_name": dn,
                "clip_grad": 1,
                "max_norm": 1,
                "jvm_gain": 8,
                "clip_method": "auto",
            }
            run_config(cfg)


for hs in [16, 32, 48, 64, 96]:
    for ct in ["GRU", "LSTM"]:
        for dn in ["jvm", "ht1"]:
            cfg = {
                "hidden_size": hs,
                "max_epochs": me,
                "model_type": "base",
                "cell_type": ct,
                "pre_size": 1,
                "run_name": "autoclip_3",
                "device_name": dn,
                "clip_grad": 1,
                "max_norm": 1,
                "jvm_gain": 8,
                "clip_method": "auto",
            }
            run_config(cfg)
