import dotenv
dotenv.load_dotenv()
import openai,os
openai.api_key = os.getenv("OPENAI_API_KEY")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import torch
import binascii
import numpy as np
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from ik.ik import ik
from visualize import Joints2SMPL


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
    data["j2s"] = Joints2SMPL(device=device)
    return data


@app.get("/mld_pos/")
async def mld_pos(prompt: str):
    fps = 20
    with torch.no_grad():
        batch = {"length": [100], "text": [prompt]}
        joints = data["model"](batch)
    joints = joints[0].numpy()
    return {"positions": binascii.b2a_base64(joints.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "dtype": "float32",
            "fps": fps,
            "mode": "xyz",
            "n_frames": joints.shape[0],
            "n_joints": 22}


# @app.get("/mld_quat/")
# async def mld_quat(prompt: str, normalized_offset_fp32: str = None):
#     fps = 20
#     with torch.no_grad():
#         batch = {"length": [100], "text": [prompt]}
#         joints = data["model"](batch)
#     joints = joints[0].numpy()
#     if normalized_offset_fp32 is None:
#         quat, root_pos = ik(joints)
#     else:
#         offset = np.frombuffer(binascii.a2b_base64(normalized_offset_fp32), dtype="float32").reshape(-1, 3)
#         quat, root_pos = ik(joints, offset)
#
#     return {"root_positions": binascii.b2a_base64(
#         root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
#             "rotations": binascii.b2a_base64(quat.flatten().astype(np.float32).tobytes()).decode("utf-8"),
#             "dtype": "float32",
#             "fps": fps,
#             "mode": "quaternion",
#             "n_frames": joints.shape[0],
#             "n_joints": 22}


@app.get("/mld_angle/")
async def mld_angle(prompt: str):
    fps = 20
    try:
        prompt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "translate to english without any explanation"},
                      {"role": "user", "content": prompt}],
        )["choices"][0]["message"]["content"]
    except:
        pass
    with torch.no_grad():
        batch = {"length": [100], "text": [prompt]}
        joints = data["model"](batch)
    joints = joints[0].numpy()
    rotations, root_pos = data["j2s"](joints, step_size=2e-2, num_iters=25, optimizer="lbfgs")
    return {"root_positions": binascii.b2a_base64(
        root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "rotations": binascii.b2a_base64(rotations.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "dtype": "float32",
            "fps": fps,
            "mode": "axis_angle",
            "n_frames": joints.shape[0],
            "n_joints": 24}
