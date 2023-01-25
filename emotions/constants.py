"""
Constant variables used by app.
"""
import os

MM_HOME = os.environ.get("MM_HOME")
DATA_HOME = os.path.join(MM_HOME, "emotions/data")


FEATURIZER_SERVER = "http://127.0.0.1:8090"
MITI_THRESHOLD = 0.5
EMOTION_THRESHOLD = 0.6
EMPATHY_THRESHOLD = 0.5
COG_DIST_THRESHOLD = 0.7

THERAPIST = "therapist"
PATIENT = "patient"
