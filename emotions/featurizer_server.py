"""
Serve featurizer.
"""

import json
import base64
import numpy as np
from flask import Flask, request
from emotions.featurizer import load_encoder


encoder = load_encoder()
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
    result = encoder.encode(query)
    return json.dumps(result, cls=MyEncoder)


@app.route("/ping", methods=["GET"])
def ping():
    return {"ready": True}


if __name__ == "__main__":
    app.run()
