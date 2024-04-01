import setproctitle

setproctitle.setproctitle("Convert")
import os
import numpy as np
from mld.data.humanml.utils.plot_script import plot_3d_motion
from visualize.joints2smpl.my_smpl import MySMPL
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import random
import pickle as pk

n_workers = 4


def action(worker_id, files):
    os.makedirs("/home/user/priorMDM/dataset/inter-human/video", exist_ok=True)
    smpl_model = MySMPL("deps/body_models/smpl", gender="neutral", ext="pkl")

    block_size = (len(files) + n_workers - 1) // n_workers
    start = worker_id * block_size
    end = start + block_size
    files = files[start:end]
    for file in tqdm(files):
        if os.path.exists("/home/user/priorMDM/dataset/inter-human/video/" + file.replace(".pkl", ".mp4")):
            continue
        with open("/home/user/priorMDM/dataset/inter-human/motions/" + file, "rb") as f:
            motion = pk.load(f)
        two_joints = []
        with torch.no_grad():
            for person_name in ['person1', 'person2']:
                person = motion[person_name]
                smpl_output = smpl_model(global_orient=torch.from_numpy(person['root_orient']),
                                         body_pose=torch.from_numpy(np.concatenate(
                                             (person['pose_body'], np.zeros_like(person['pose_body'][:, :6])), axis=1)),
                                         transl=torch.from_numpy(person['trans']),
                                         betas=torch.from_numpy(
                                             np.tile(person['betas'][None], (person['trans'].shape[0], 1))),
                                         )
                joints = smpl_output.joints.numpy()[..., [0, 2, 1]]
                two_joints.append(joints)
        two_joints = np.concatenate(two_joints, axis=1)
        plot_3d_motion("/home/user/priorMDM/dataset/inter-human/video/" + file.replace(".pkl", ".mp4"),
                       two_joints * 1.3,
                       radius=3, title=file.strip(".npy"),
                       fps=motion['mocap_framerate'])


if __name__ == "__main__":
    mp.set_start_method('spawn')
    files = os.listdir("/home/user/priorMDM/dataset/inter-human/motions")
    random.shuffle(files)
    processes = []
    for i in range(n_workers):
        p = mp.Process(target=action, args=(i, files))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("Done!")
