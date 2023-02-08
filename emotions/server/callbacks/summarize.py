"""
Callback method to generate summary.
"""

from dash import callback, Input, Output, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc

from emotions.server.components import summary as summary_component
from emotions.backend.search import evaluate
from emotions.config import (
    EMOTIONS,
    COG_DISTS,
    EMPATHY_COMMUNICATION_MECHANISMS,
    MITI_CODES,
)
from emotions.constants import THERAPIST, PATIENT


@callback(
    Output(summary_component, "children"),
    [Input("conversation-encoding", "data")],
)
def summarize(mm_data):
    """
    Return summary of dialogue.
    """

    # parsed_query = ["or"]
    # parsed_query.extend(
    #    [[">=", "miti_%s" % miti_code, 0.8] for miti_code in MITI_CODES]
    # )
    # parsed_query.extend([
    #    [">=", "custom_%s" % emotion, 0.8] for emotion in EMOTIONS
    # ])
    # parsed_query.extend([
    #    [">=", "emotion_%s" % emotion, 0.8] for emotion in EMOTIONS
    # ])
    # parsed_query.extend([
    #    [">=", "empathy_%s" % empathy, 0.8]
    #    for empathy in EMPATHY_COMMUNICATION_MECHANISMS
    # ])
    # parsed_query.extend([
    #    [">=", "epitome_%s" % empathy, 0.8]
    #    for empathy in EMPATHY_COMMUNICATION_MECHANISMS
    # ])
    # parsed_query.extend([
    #    [">=", "cog_dist_%s" % cog_dist, 0.8] for cog_dist in COG_DISTS
    # ])

    emotion_queries = [
        [">=", "custom_%s" % emotion, 0.8] for emotion in EMOTIONS
    ]
    emotion_queries.extend(
        [[">=", "emotion_%s" % emotion, 0.8] for emotion in EMOTIONS]
    )
    emotion_queries = (emotion_queries, None)

    empathy_queries = [
        [">=", "empathy_%s" % empathy, 0.8]
        for empathy in EMPATHY_COMMUNICATION_MECHANISMS
    ]
    empathy_queries.extend(
        [
            [">=", "epitome_%s" % empathy, 0.8]
            for empathy in EMPATHY_COMMUNICATION_MECHANISMS
        ]
    )
    empathy_queries = (empathy_queries, THERAPIST)

    queries = {
        "miti": ([
            [">=", "miti_%s" % miti_code, 0.8] for miti_code in MITI_CODES
        ], THERAPIST),
        "emotions": emotion_queries,
        "empathy": empathy_queries,
        "cog_dist": ([
            [">=", "cog_dist_%s" % cog_dist, 0.8] for cog_dist in COG_DISTS
        ], PATIENT),
    }

    results = {}
    summaries = []
    for mm_type, query_obj in queries.items():
        speaker = query_obj[1]
        query = ["or"]
        query.extend(query_obj[0])

        if speaker:
            query = ["and", query, ["==", "speaker", speaker]]

        print(query)

        results[mm_type] = sorted(evaluate(query, mm_data))

        summary = "%s: %d / %d" % (mm_type.upper(), len(results[mm_type]), len(mm_data))
        summaries.append(
            dbc.Button(
                summary,
                id={"type": "summary_box"},
                n_clicks=0,
                color="green",
            )
        )

    return summaries
