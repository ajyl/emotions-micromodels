"""
Serve featurizer.
"""

import os
import json
import base64
import numpy as np
from tqdm import tqdm
from flask import Flask, request
from emotions.featurizer import load_encoder

MI_DATA_DIR = "./data/HighLowQualityCounseling/json"


def init_cache(data_dir=MI_DATA_DIR):
    """
    Initialize MI dialogue data.
    """
    print("Initializing cache...")
    cache = {}
    for filename in tqdm(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(data_dir, filename)) as file_p:
            data = json.load(file_p)

        for utt_obj in data:
            utterance = utt_obj["utterance"]
            cache[utterance] = encoder.encode(utterance)

    breakpoint()
    return cache


encoder = load_encoder()
# cache = init_cache()
cache = {}

app = Flask(__name__)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode("ascii")
        else:
            return super(MyEncoder, self).default(obj)


@app.route("/encode", methods=["POST"])
def encode():
    query = request.json["query"]
    result = cache.get(query, encoder.encode(query))
    if query not in cache:
        cache[query] = result
    return json.dumps(result, cls=MyEncoder)


@app.route("/ping", methods=["GET"])
def ping():
    return {"ready": True}


if __name__ == "__main__":
    app.run()
