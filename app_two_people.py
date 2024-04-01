# Copyright 2023 The SeedV Lab (Beijing SeedV Technology Co., Ltd.)
# All Rights Reserved.

import gradio as gr
import requests
import dotenv

dotenv.load_dotenv()
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")
from opensearchpy import OpenSearch
from functools import partial


def get_cli():
    host = os.getenv("OPENSEARCH_SERVER")
    auth = (os.getenv("OPENSEARCH_USER", 'admin'), os.getenv("OPENSEARCH_PASSWORD", 'admin'))
    client = OpenSearch(
        hosts=[host],
        http_compress=bool(int(os.getenv("HTTP_COMPRESS", "1"))),
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        request_timeout=10,
        timeout=10,
    )
    return client


def get_openai_embedding(query):
    for i in range(5):
        try:
            ret = openai.Embedding.create(
                input=query,
                model="text-embedding-ada-002",
                timeout=10,
                request_timeout=10,
            )
            text_embeddings = ret['data'][0]['embedding']
            print("Usage: " + str(ret["usage"].to_dict()).replace("\n", ""))
            return text_embeddings
        except Exception:
            pass
    print(f"Error 508, failed to call openai ada")


def movie2path(movie):
    return "/mnt/extdisk/Download/InternVId-FLT_1/" + movie["youtube_id"] + "_" + movie["start_timestamp"] + "_" + \
        movie["end_timestamp"] + ".mp4"


def text2movies(query, duration_lower_bound):
    assert query
    text_embedding = get_openai_embedding(query)
    should_list = []
    should_list.append({
        "function_score": {
            "query": {
                "knn": {
                    "text_vector": {
                        "vector": text_embedding,
                        "k": 180,
                    }
                }
            },
            "functions": [
                {
                    "script_score": {
                        "script": {
                            "source": "1.0"
                        }
                    }
                }
            ],
            "boost_mode": "multiply"
        }
    })
    should_list.append({
        "function_score": {
            "query": {
                "match": {
                    "text": {
                        "query": query,
                    }
                }
            },
            "functions": [
                {
                    "script_score": {
                        "script": {
                            "source": f"0.01*Math.log(1 + _score)"
                        }
                    }
                }
            ],
            "boost_mode": "replace"
        }
    })
    must_list = []
    if duration_lower_bound > 0:
        must_list.append({"range": {
            "time": {
                "gte": duration_lower_bound
            }
        }})
    if must_list:
        must = {"must": must_list}
    else:
        must = {}
    query_body = {
        "_source": ["motion_id", "text"],
        "size": 180,
        "query": {
            'bool': {
                **must,
                'should': should_list
            }
        }
    }
    with get_cli() as client:
        hits = client.search(
            index="two-people-index",
            body=query_body
        )
    all_videos = []
    for hit in hits['hits']['hits']:
        all_videos.append(hit['_source'])
    page = 0
    movies = []
    pickles = []
    for video in all_videos[:9]:
        if os.path.exists("/home/user/priorMDM/dataset/inter-human/video/" + video["motion_id"] + ".mp4"):
            movies.append(
                gr.update(value="/home/user/priorMDM/dataset/inter-human/video/" + video["motion_id"] + ".mp4",
                          label=video["text"][:100], visible=True))
            pickles.append(
                gr.update(value="/home/user/priorMDM/dataset/inter-human/motions/" + video["motion_id"] + ".pkl",
                          visible=True))
        else:
            movies.append(gr.update(value=None, label=None, visible=True))
            pickles.append(gr.update(value=None, visible=True))
    return movies + pickles + [page, all_videos]


def set_page(mode, page, all_videos):
    if mode == "prev":
        page = max(0, page - 1)
    else:
        page = min(len(all_videos) // 9 - 1, page + 1)
    movies = []
    pickles = []
    for video in all_videos[page * 9: (page + 1) * 9]:
        if os.path.exists("/home/user/priorMDM/dataset/inter-human/video/" + video["motion_id"] + ".mp4"):
            movies.append(
                gr.update(value="/home/user/priorMDM/dataset/inter-human/video/" + video["motion_id"] + ".mp4",
                          label=video["text"][:100], visible=True))
            pickles.append(
                gr.update(value="/home/user/priorMDM/dataset/inter-human/motions/" + video["motion_id"] + ".pkl",
                          visible=True))
        else:
            movies.append(gr.update(value=None, label=None, visible=True))
            pickles.append(gr.update(value=None, visible=True))
    return movies + pickles + [page]


def main():
    css = "#model-3d-out {height: 400px;} #plot-out {height: 450px;}"
    title = "Two People Motion"
    movies = []
    pickles = []
    with gr.Blocks(title=title, css=css) as demo:
        gr.Markdown('# ' + title)
        page = gr.State(0)
        all_videos = gr.State([])
        with gr.Column():
            query = gr.Text(placeholder="running", label="Query")
            with gr.Row():
                duration_lower_bound = gr.Slider(minimum=0, maximum=20, step=1, value=0, label="Duration Lower Bound")
                btn = gr.Button("Search", variant="primary")
                btn_prev_page = gr.Button("Previous Page", variant="secondary")
                btn_next_page = gr.Button("Next Page", variant="secondary")
        for i in range(3):
            with gr.Row():
                for j in range(3):
                    with gr.Column():
                        movies.append(gr.Video(format="mp4", autoplay=True, visible=False))
                        pickles.append(gr.File(visible=False))

        btn.click(fn=text2movies,
                  inputs=[query, duration_lower_bound],
                  outputs=movies + pickles + [page, all_videos])
        btn_prev_page.click(fn=partial(set_page, "prev"),
                            inputs=[page, all_videos],
                            outputs=movies + pickles + [page])
        btn_next_page.click(fn=partial(set_page, "next"),
                            inputs=[page, all_videos],
                            outputs=movies + pickles + [page])

    demo.launch(server_name="0.0.0.0", server_port=6405)


if __name__ == "__main__":
    main()
