"""
Micromodel Configurations
"""

import os
import json

EMP_TASKS = ["emotional_reactions", "explorations", "interpretations"]
EMP_MMS = [
    "empathy_emotional_reactions",
    "empathy_explorations",
    "empathy_interpretations",
]


INCLUDE_EMPATHY = True
BUILD_EMPATHY_MM = True

INCLUDE_ED_EMOTIONS = True
BUILD_ED_EMOTION_MMS = True

INCLUDE_GO_EMOTIONS = False
BUILD_GO_EMOTION_MMS = False

ED_EMOTIONS = [
    "afraid",
    "angry",
    "annoyed",
    "anticipating",
    "anxious",
    "apprehensive",
    "ashamed",
    "caring",
    "confident",
    "content",
    "devastated",
    "disappointed",
    "disgusted",
    "embarrassed",
    "excited",
    "faithful",
    "furious",
    "grateful",
    "guilty",
    "hopeful",
    "impressed",
    "jealous",
    "joyful",
    "lonely",
    "nostalgic",
    "prepared",
    "proud",
    "sad",
    "sentimental",
    "surprised",
    "terrified",
    "trusting",
]

ED_EMOTIONS_EKMAN = [
    "angry", # anger
    "disgusted", # disgust
    # fear
    "joyful", # joy
    # neutral
    "sad", # sadness
    "surprised", # surprise
]



GO_EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
]


mm_base_path = os.environ.get("MM_HOME")
mm_model_dir = os.path.join(mm_base_path, "models")
ed_seed_dir = os.path.join(mm_base_path, "emotions/seeds")


def load_mm_config(json_path):
    """
    Load bert-query config from json file.
    """
    with open(json_path, "r") as file_p:
        data = json.load(file_p)
    return data


MM_CONFIG = []


EMP_CONFIGS = [
    # EMPATHY
    {
        "name": "empathy_emotional_reactions",
        "model_type": "bert_query",
        "model_path": os.path.join(
            mm_model_dir, "empathy_emotional_reactions"
        ),
        "setup_args": {},
        "build": BUILD_EMPATHY_MM,
    },
    {
        "name": "empathy_explorations",
        "model_type": "bert_query",
        "model_path": os.path.join(mm_model_dir, "empathy_explorations"),
        "setup_args": {},
        "build": BUILD_EMPATHY_MM,
    },
    {
        "name": "empathy_interpretations",
        "model_type": "bert_query",
        "model_path": os.path.join(mm_model_dir, "empathy_interpretations"),
        "setup_args": {},
        "build": BUILD_EMPATHY_MM,
    },
]

EMP_CONFIGS_MERGED = [
    {
        "name": "empathy_%s" % task,
        "model_type": "bert_query",
        "model_path": os.path.join(
            mm_model_dir, "empathy_%s" % task
        ),
        "setup_args": {
            "infer_config": {
                "segment_config": {"window_size": 10, "step_size": 4},
            },
            "seed": [],
        },
        "build": BUILD_EMPATHY_MM,
    }
    for task in EMP_TASKS
]


def add_go_emotions(configs):
    """ Add GoEmotion micromodels """
    for emotion in GO_EMOTIONS:
        filepath = os.path.join(mm_model_dir, "go_%s.json" % emotion)
        setup_args = load_mm_config(filepath)
        setup_args["threshold"] = 0.6
        setup_args["infer_config"] = {
            "segment_config": {"window_size": 999, "step_size": 10}
        }
        config = {
            "name": "go_emotion_%s" % emotion,
            "model_type": "bert_query",
            "setup_args": setup_args,
            "model_path": os.path.join(mm_model_dir, "go_%s" % emotion),
            "build": BUILD_GO_EMOTION_MMS
        }
        configs.append(config)

def add_ed_emotions(configs):
    """ Add emotion micromodels """
    for emotion in ED_EMOTIONS:
        filepath = os.path.join(ed_seed_dir, "ed_%s.json" % emotion)
        setup_args = load_mm_config(filepath)
        setup_args["threshold"] = 0.6
        setup_args["infer_config"] = {
            "segment_config": {"window_size": 999, "step_size": 10}
        }
        config = {
            "name": "emotion_%s" % emotion,
            "model_type": "bert_query",
            "setup_args": setup_args,
            "model_path": os.path.join(ed_seed_dir, "%s" % emotion),
            "build": BUILD_ED_EMOTION_MMS
        }
        configs.append(config)


# EMPATHY
if INCLUDE_EMPATHY:
    MM_CONFIG.extend(EMP_CONFIGS)

# ED EMOTIONs
if INCLUDE_ED_EMOTIONS:
    add_ed_emotions(MM_CONFIG)

# GO EMOTIONS
if INCLUDE_GO_EMOTIONS:
    add_go_emotions(MM_CONFIG)
