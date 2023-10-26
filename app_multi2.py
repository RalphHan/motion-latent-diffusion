import setproctitle
setproctitle.setproctitle("AMulti")
import gradio as gr
import uuid
from vis import skeleton_render
import numpy as np
import requests
import binascii
from mld.data.humanml.utils.plot_script import plot_3d_motion
from visualize.joints2smpl.my_smpl import MySMPL
import torch
import multiprocessing as mp


def draw(render, mid, joints, video_name, prompt, fps):
    if render == "absolute":
        skeleton_render(joints[..., [2, 0, 1]], video_name, fps=fps)
    elif render == "relational":
        plot_3d_motion(video_name, joints * 1.3, radius=3, title=mid + ":" + prompt, fps=fps)
    else:
        raise ValueError(f"render {render} not supported")


def action(render, translate, dance, random, prompt):
    the_uuid = str(uuid.uuid4())
    video_names = [f"results/gradio/{the_uuid}-{i}.mp4" for i in range(4)]
    params = {"prompt": prompt, "do_translation": translate,
              "want_number": 4}
    if dance:
        params["style"] = "dance"
    if random:
        params["regenerate"] = 1
    ret_jsons = requests.get("http://34.123.39.219:6399/angle/",
                             params=params).json()
    all_rotations = [
        np.frombuffer(binascii.a2b_base64(ret_json["rotations"]), dtype=ret_json["dtype"]).reshape(-1, 24*3)
        for ret_json in ret_jsons]
    all_root_pos = [
        np.frombuffer(binascii.a2b_base64(ret_json["root_positions"]), dtype=ret_json["dtype"]).reshape(-1, 3)
        for ret_json in ret_jsons]
    processes = []
    for mid, rotations, root_pos, video_name, fps in zip([ret_json["mid"] for ret_json in ret_jsons], all_rotations,
                                                         all_root_pos, video_names,
                                                         [ret_json["fps"] for ret_json in ret_jsons]):
        with torch.no_grad():
            pose = torch.tensor(rotations, dtype=torch.float32)
            smpl_output = smpl_model(global_orient=pose[:, :3],
                                     body_pose=pose[:, 3:],
                                     transl=torch.tensor(root_pos, dtype=torch.float32)
                                     )
            joints = smpl_output.joints.numpy()
        p = mp.Process(target=draw, args=(render, mid, joints, video_name, prompt, fps))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    return video_names


if __name__ == "__main__":
    smpl_model = MySMPL("deps/body_models/smpl", gender="neutral", ext="pkl")
    demo = gr.Interface(
        action,
        [gr.Dropdown(choices=["relational", "absolute"], value="relational", label="render"),
         gr.Checkbox(label="translate", value=True),
         gr.Checkbox(label="dance", value=False),
         gr.Checkbox(label="random", value=False),
         gr.Textbox("A person is skipping rope.")],
        [gr.Video(format="mp4", autoplay=True, label=str(i), width=225, height=225) for i in range(4)],
    )
    demo.launch(server_name='0.0.0.0', server_port=7869)
