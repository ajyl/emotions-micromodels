"""
Format AnnoMI data.
"""

import os
import csv
import json
from collections import defaultdict
from emotions.constants import DATA_HOME

ANNO_MI_DIR = os.path.join(DATA_HOME, "AnnoMI")
DATA_FILE = os.path.join(ANNO_MI_DIR, "dataset.csv")


def main():
    """ Driver """
    data = defaultdict(list)
    with open(DATA_FILE, "r") as file_p:
        reader = csv.DictReader(file_p)

        for row in reader:

            transcript_label = "%s_%s" % (
                row["mi_quality"],
                row["transcript_id"],
            )
            data[transcript_label].append(
                {
                    "utt_id": row["utterance_id"],
                    "speaker": "therapist" if row["interlocutor"] == "therapist" else "patient",
                    "utterance": row["utterance_text"],
                    "therapist_behavior": row["main_therapist_behaviour"],
                    "client_talk_type": row["client_talk_type"],
                }
            )

    output_filepath = os.path.join(ANNO_MI_DIR, "anno_mi.json")
    with open(output_filepath, "w") as file_p:
        json.dump(data, file_p, indent=4)


if __name__ == "__main__":
    main()
