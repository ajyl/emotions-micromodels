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
from interpret.glassbox.ebm.ebm import EBMExplanation
from emotions.backend.featurizer import load_encoder
from emotions.server.components.dialogue_dropdown import init_anno_mi_data
from emotions.constants import THERAPIST, PATIENT


MM_HOME = os.environ.get("MM_HOME")
MODELS_DIR = os.path.join(MM_HOME, "emotions/models")
ed_classifier = os.path.join(MODELS_DIR, "ed_classifier.pkl")
emp_er_classifier = os.path.join(MODELS_DIR, "emp_er.pkl")
emp_exp_classifier = os.path.join(MODELS_DIR, "emp_exp.pkl")
emp_int_classifier = os.path.join(MODELS_DIR, "emp_int.pkl")
epitome_er_classifier = os.path.join(MODELS_DIR, "EPITOME_ER.pth")
epitome_exp_classifier = os.path.join(MODELS_DIR, "EPITOME_EXP.pth")
epitome_int_classifier = os.path.join(MODELS_DIR, "EPITOME_INT.pth")

cache_filepath = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "cache.json"
)

pair_path = os.path.join(MODELS_DIR, "pair.pth")

encoder = load_encoder(
    ed_path=ed_classifier,
    emp_er_path=emp_er_classifier,
    emp_exp_path=emp_exp_classifier,
    emp_int_path=emp_int_classifier,
    epitome_er_path=epitome_er_classifier,
    epitome_exp_path=epitome_exp_classifier,
    epitome_int_path=epitome_int_classifier,
    pair_path=pair_path,
    device="cpu",
)



def init_cache(cache_filepath=None):
    print("Initializing cache...")
    cache = {}
    if cache_filepath and os.path.isfile(cache_filepath):
        with open(cache_filepath, "r") as file_p:
            cache = json.load(file_p)
        return cache

    mi_data = init_anno_mi_data()
    for dialogue in tqdm(mi_data.values()):
        for idx, utt_obj in enumerate(dialogue):
            if utt_obj["speaker"] == THERAPIST and idx > 0:
                prompt = dialogue[idx-1]["utterance"]
                response = utt_obj["utterance"]

                cache[utt_obj["utterance"]] = encoder.encode(response, prompt)
            else:
                cache[utt_obj["utterance"]] = encoder.encode_utterance(utt_obj["utterance"])

    return cache


cache = init_cache(cache_filepath)

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


@app.route("/explain", methods=["GET", "POST"])
def explain():
    return json.dumps(encoder.get_explain(), cls=MyEncoder)


@app.route("/ping", methods=["GET"])
def ping():
    return {"ready": True}


if __name__ == "__main__":
    app.run(port=8090)
