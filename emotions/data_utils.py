"""
Utility functions for data.
"""

import os
import json
import csv
from collections import defaultdict
from nltk import tokenize

MM_HOME = os.environ.get("MM_HOME")
DATA_HOME = os.path.join(MM_HOME, "emotions/data")
EMP_DATA_FILE = os.path.join(DATA_HOME, "epitome.json")


def load_emp_data(filepath=EMP_DATA_FILE):
    """
    Load Epitome data, including sentence-tokenizing the input text and
    splitting the data into train, val, and test splits.
    """
    with open(filepath, "r") as file_p:
        data = json.load(file_p)

    for _, data_obj in data.items():
        data_obj["seeker_tokenized"] = tokenize.sent_tokenize(
            data_obj["seeker_post"]
        )
        data_obj["response_tokenized"] = [
            x
            for x in tokenize.sent_tokenize(data_obj["response_post"])
            if len(x) > 3
        ]
    return list(data.values())


def load_ed_data(file_path):
    """
    Load Empathetic Dialogue data
    """
    data = defaultdict(list)
    with open(file_path, "r") as file_p:
        reader = csv.DictReader(file_p)

        for row in reader:
            prompt = row["prompt"].replace("_comma_", ",")
            if prompt not in data[row["context"]]:
                data[row["context"]].append(prompt)
    return data

def reformat_emp_data(data):
    """Reformat data to list of queries"""
    query_groups = [x["response_tokenized"] for x in data]
    reformatted = []
    groupings = {}
    query_idx = 0
    for idx, queries in enumerate(query_groups):
        for _ in queries:
            groupings[query_idx] = idx
            query_idx += 1
        reformatted.extend(queries)
    return reformatted, groupings


