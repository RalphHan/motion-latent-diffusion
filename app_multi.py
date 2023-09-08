import gradio as gr
import uuid
from vis import skeleton_render
import numpy as np
import requests
import binascii
from mld.data.humanml.utils.plot_script import plot_3d_motion
import multiprocessing as mp


def draw(render, joints, video_name, prompt, fps):
    if render == "absolute":
        skeleton_render(np.concatenate((joints, joints[:, -2:]), axis=-2)[..., [2, 0, 1]], video_name)
    elif render == "relational":
        plot_3d_motion(video_name, joints * 1.3, radius=3, title=prompt, fps=fps)
    else:
        raise ValueError(f"render {render} not supported")


def action(render, translate, search, refine, prompt):
    the_uuid = str(uuid.uuid4())
    video_names = [f"results/gradio/{the_uuid}-{i}.mp4" for i in range(4)]
    fps = 20
    ret_jsons = requests.get("http://0.0.0.0:8019/position/",
                             params={"prompt": prompt, "do_translation": translate, "do_search": search,
                                     "do_refine": refine, "want_number": 4}).json()
    all_joints = [np.frombuffer(binascii.a2b_base64(ret_json["positions"]), dtype=ret_json["dtype"]).reshape(-1, 22, 3)
                  for ret_json in ret_jsons]
    processes = []
    for joints, video_name in zip(all_joints, video_names):
        p = mp.Process(target=draw, args=(render, joints, video_name, prompt, fps))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    return video_names


if __name__ == "__main__":
    demo = gr.Interface(
        action,
        [gr.Dropdown(choices=["relational", "absolute"], value="relational", label="render"),
         gr.Checkbox(label="translate", value=True),
         gr.Checkbox(label="search", value=True),
         gr.Checkbox(label="refine", value=True),
         gr.Textbox("A person is skipping rope.")],
        [gr.Video(format="mp4", autoplay=True, label=str(i), width=225, height=225) for i in range(4)],
    )
    demo.launch(server_name='0.0.0.0', server_port=7867)
