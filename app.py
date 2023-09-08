import gradio as gr
import uuid
import json
from vis import skeleton_render
import numpy as np
import requests
import binascii
from mld.data.humanml.utils.plot_script import plot_3d_motion


def action(render, translate, search, refine, prompt):
    the_uuid = str(uuid.uuid4())
    fps = 20
    video_name = f"results/gradio/{the_uuid}.mp4"
    json_name = f"results/gradio/{the_uuid}.json"
    # ret_json = json.loads(requests.get("http://0.0.0.0:8019/mld_pos/", params={"prompt": prompt}).text)
    ret_json = requests.get("http://0.0.0.0:8019/position/",
                            params={"prompt": prompt, "do_translation": translate, "do_search": search,
                                    "do_refine": refine}).json()[0]
    joints = np.frombuffer(binascii.a2b_base64(ret_json["positions"]), dtype=ret_json["dtype"]).reshape(-1, 22, 3)
    with open(json_name, "w") as f:
        json.dump(ret_json, f, indent=4)
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
    demo = gr.Interface(
        action,
        [gr.Dropdown(choices=["none", "relational", "absolute"], value="relational", label="render"),
         gr.Checkbox(label="translate", value=True),
         gr.Checkbox(label="search", value=True),
         gr.Checkbox(label="refine", value=True),
         gr.Textbox("A person is skipping rope.")],
        [gr.Video(format="mp4", autoplay=True), gr.File()],
    )
    demo.launch(server_name='0.0.0.0', server_port=7867)
