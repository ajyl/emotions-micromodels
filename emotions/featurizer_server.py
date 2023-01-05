"""
Serve featurizer.
"""

import os
import json
import base64
import pickle
import numpy as np
from tqdm import tqdm
from flask import Flask, request
from emotions.featurizer import load_encoder
from interpret.glassbox.ebm.ebm import EBMExplanation

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
            cache[utterance] = encoder.encode_utterance(utterance)

    breakpoint()
    return cache


MM_HOME = os.environ.get("MM_HOME")
MODELS_DIR = os.path.join(MM_HOME, "emotions/models")
ed_classifier = os.path.join(MODELS_DIR, "ed_classifier.pkl")
emp_er_classifier = os.path.join(MODELS_DIR, "emp_er.pkl")
emp_exp_classifier = os.path.join(MODELS_DIR, "emp_exp.pkl")
emp_int_classifier = os.path.join(MODELS_DIR, "emp_int.pkl")
pair_path = os.path.join(MODELS_DIR, "pair.pth")

encoder = load_encoder(
    ed_path=ed_classifier,
    emp_er_path=emp_er_classifier,
    emp_exp_path=emp_exp_classifier,
    emp_int_path=emp_int_classifier,
    pair_path=pair_path,
    device="cuda:0",
)


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
        elif isinstance(obj, EBMExplanation):
            return pickle.dumps(obj)
        else:
            return super(MyEncoder, self).default(obj)


@app.route("/encode_utterance", methods=["POST"])
def encode_utterance():
    query = request.json["query"]
    result = cache.get(query, encoder.encode_utterance(query))
    result["emotion"]["explanations"]["global"] = pickle.dumps(
        result["emotion"]["explanations"]["global"]
    )
    if query not in cache:
        cache[query] = result
    return json.dumps(result, cls=MyEncoder)


@app.route("/encode", methods=["POST"])
def encode():
    prompt = request.json.get("prompt")
    response = request.json["response"]
    result = encoder.encode(response, prompt)
    return json.dumps(result, cls=MyEncoder)


@app.route("/ping", methods=["GET"])
def ping():
    return {"ready": True}


if __name__ == "__main__":
    app.run()
