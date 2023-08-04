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

from mld.data.humanml.utils.plot_script import plot_3d_motion


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

    return cfg, model


def action(render, prompt, cfg, model):
    the_uuid = str(uuid.uuid4())
    fps = 20
    video_name = f"{cfg.gradio_dir}/{the_uuid}.mp4"
    json_name = f"{cfg.gradio_dir}/{the_uuid}.json"

    with torch.no_grad():
        batch = {"length": [100], "text": [prompt]}
        joints = model(batch)
    joints = joints[0].numpy()
    with open(json_name, "w") as f:
        json.dump({"positions": joints.flatten().tolist(),
                   "fps": fps,
                   "mode": "xyz",
                   "n_frames": joints.shape[0],
                   "n_joints": 22}, f, indent=4)
    if render != "none":
        if render == "absolute":
            skeleton_render(np.concatenate((joints, joints[:, -2:]), axis=-2)[..., [2, 0, 1]], video_name)
        elif render == "relational":
            plot_3d_motion(video_name, joints * 1.3, radius=3, title=prompt, fps=fps)
        else:
            raise ValueError(f"render {render} not supported")
        return video_name, json_name
    return None, json_name


if __name__ == "__main__":
    cfg, model = init()
    demo = gr.Interface(
        lambda render, prompt: action(render, prompt, cfg, model),
        [gr.Dropdown(choices=["none", "relational", "absolute"], value="relational", label="render"),
         gr.Textbox("A person is skipping rope.")],
        [gr.Video(format="mp4", autoplay=True), gr.File()],
    )
    demo.launch(server_name='0.0.0.0', server_port=7867)
