import os
import uuid
import numpy as np
import requests
import binascii
from mld.data.humanml.utils.plot_script import plot_3d_motion
from visualize.joints2smpl.my_smpl import MySMPL
import torch
import multiprocessing as mp

n_workers=4
def action(worker_id):
    os.makedirs("tmp/video", exist_ok=True)
    os.makedirs("tmp/video_with_music", exist_ok=True)
    smpl_model = MySMPL("deps/body_models/smpl", gender="neutral", ext="pkl")
    files=sorted(os.listdir("tmp/rotations"))
    block_size = (len(files) + n_workers - 1) // n_workers
    start = worker_id * block_size
    end = start + block_size
    motions = motions[start:end]

    plot_3d_motion("tmp/", joints * 1.3, radius=3, title=mid + ":" + prompt, fps=fps)
    the_uuid = str(uuid.uuid4())
    video_names = [f"results/gradio/{the_uuid}-{i}.mp4" for i in range(4)]
    ret_jsons = requests.get("http://34.123.39.219:6399/angle/",
                             params={"prompt": prompt, "do_translation": translate, "is_dance": dance,
                                     "is_random": random,
                                     "want_number": 4}).json()
    all_rotations = [
        np.frombuffer(binascii.a2b_base64(ret_json["rotations"]), dtype=ret_json["dtype"]).reshape(-1, 24, 3)
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
    mp.set_start_method('spawn')
    processes = []
    for i in range(n_workers):
        p = mp.Process(target=action, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Done!")