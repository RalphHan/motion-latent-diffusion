import gradio as gr
import uuid

import os
import sys
import json
from vis import skeleton_render
import numpy as np
import torch

from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def init():
    sys.argv.extend("--cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml".split())
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    cfg.gradio_dir = os.path.join(cfg.FOLDER, 'gradio')
    os.makedirs(cfg.gradio_dir, exist_ok=True)
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

    return cfg, model, logger


def action(render, prompt, cfg, model, logger):
    the_uuid = str(uuid.uuid4())
    video_name = f"{cfg.gradio_dir}/{the_uuid}.mp4"
    json_name = f"{cfg.gradio_dir}/{the_uuid}.json"

    with torch.no_grad():
        batch = {"length": [100], "text": [prompt]}
        joints, quaternion, r_pos = model(batch, return_quaternion=True)
    joints = joints[0].numpy()[..., [2, 0, 1]]
    quaternion = quaternion[0].numpy()
    if joints.shape[-2] == 22:
        joints = np.concatenate([joints, joints[..., -2:, :]], axis=-2)
        quaternion = np.concatenate([quaternion, np.zeros_like(quaternion[..., -2:, :])], axis=-2)
    r_pos = r_pos[0].numpy()
    with open(json_name, "w") as f:
        json.dump({"root_positions": r_pos.tolist(),
                   "rotations": quaternion.tolist(),
                   "fps": 20}, f, indent=4)
    if render:
        skeleton_render(joints, video_name)
        return video_name, json_name
    return None, json_name


if __name__ == "__main__":
    cfg, model, logger = init()
    demo = gr.Interface(
        lambda render, prompt: action(render, prompt, cfg, model, logger),
        [gr.Checkbox(value=True, label="render"), gr.Textbox("A person is skipping rope.")],
        [gr.Video(format="mp4", autoplay=True), gr.File()],
    )
    demo.launch(server_name='0.0.0.0', server_port=7867)
