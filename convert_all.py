import setproctitle
setproctitle.setproctitle("Convert")
import os
import numpy as np
from mld.data.humanml.utils.plot_script import plot_3d_motion
from visualize.joints2smpl.my_smpl import MySMPL
import torch
import multiprocessing as mp
import mld.utils.rotation_conversions as geometry
from moviepy.editor import VideoFileClip, AudioFileClip

n_workers=4
def action(worker_id):
    os.makedirs("tmp/video", exist_ok=True)
    os.makedirs("tmp/video_with_music", exist_ok=True)
    smpl_model = MySMPL("deps/body_models/smpl", gender="neutral", ext="pkl")
    files=sorted(os.listdir("tmp/rotations"))
    block_size = (len(files) + n_workers - 1) // n_workers
    start = worker_id * block_size
    end = start + block_size
    files = files[start:end]
    for file in files:
        with torch.no_grad():
            quat=torch.tensor(np.load("tmp/rotations/" + file),dtype=torch.float32)
            pose=geometry.quaternion_to_axis_angle(quat[...,[3,0,1,2]])
            root_position=torch.tensor(np.load("tmp/root_positions/" + file),dtype=torch.float32)
            smpl_output = smpl_model(global_orient=pose[:, :3],
                                     body_pose=pose[:, 3:],
                                     transl=root_position
                                     )
            joints = smpl_output.joints.numpy()
        plot_3d_motion(f"tmp/video/{file.replace('.npy','.mp4')}", joints * 1.3, radius=3, title=file.strip(".npy"), fps=30)
        video = VideoFileClip(f"tmp/video/{file.replace('.npy','.mp4')}")
        audio = AudioFileClip(f"tmp/music/{file.replace('.npy','.wav')}")
        video_with_audio = video.set_audio(audio)
        video_with_audio.write_videofile(f"tmp/video_with_music/{file.replace('.npy','.mp4')}", audio_codec='aac')

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