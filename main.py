from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import torch

from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

data = {}


@app.on_event('startup')
def init_data():
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + "--cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml".split()
    cfg = parse_args(phase="demo")
    sys.argv = old_argv
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    model = get_model(cfg, dataset)
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()
    data["model"] = model
    return data


@app.get("/mld/")
async def function(prompt: str):
    fps = 20
    with torch.no_grad():
        batch = {"length": [100], "text": [prompt]}
        joints = data["model"](batch)
    joints = joints[0].numpy()
    return {"positions": joints.flatten().tolist(),
            "fps": fps,
            "mode": "xyz",
            "n_frames": joints.shape[0],
            "n_joints": 22}
