"""
Callback method to generate summary.
"""

from dash import callback, Input, Output, State, ALL, ctx, no_update, html
import dash_bootstrap_components as dbc

from emotions.server.components import summary as summary_component
from emotions.server.backend.search import evaluate
from emotions.config import (
    EMOTIONS,
    COG_DISTS,
    EMPATHY_COMMUNICATION_MECHANISMS,
    MITI_CODES,
)
from emotions.constants import THERAPIST, PATIENT


def get_default_query():
    """
    Default query when dialogue is first loaded.
    """
    miti_queries = [
        [">=", "miti_%s" % miti_code, 0.8] for miti_code in MITI_CODES
    ]
    miti_queries.append([">=", "pair", 0.8])
    miti_queries = (miti_queries, THERAPIST)

    emotion_queries = [
        [">=", "fasttext_emotion_%s" % emotion, 0.8] for emotion in EMOTIONS
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
        "miti": miti_queries,
        "emotions": emotion_queries,
        "empathy": empathy_queries,
        "cog_dist": (
            [[">=", "cog_dist_%s" % cog_dist, 0.8] for cog_dist in COG_DISTS],
            PATIENT,
        ),
    }
    return queries


@callback(
    [
        Output(summary_component, "children"),
        Output("miti-summary-idxs", "data"),
        Output("emotions-summary-idxs", "data"),
        Output("empathy-summary-idxs", "data"),
        Output("cog-dist-summary-idxs", "data"),
    ],
    [Input("conversation-encoding", "data")],
)
def summarize(mm_data):
    """
    Return summary of dialogue.
    """
    queries = get_default_query()
    results = {}
    summaries = []
    idx = 0
    for mm_type, query_obj in queries.items():
        speaker = query_obj[1]
        query = ["or"]
        query.extend(query_obj[0])

        if speaker:
            query = ["and", query, ["==", "speaker", speaker]]

        results[mm_type] = sorted(evaluate(query, mm_data))

        # summary = "%s: %d / %d" % (
        #    mm_type.upper(),
        #    len(results[mm_type]),
        #    len(mm_data),
        # )
        summary_str_map = {
            "miti": "MITI codes",
            "cog_dist": "cognitive distortions",
        }
        summary = "%d out of %d utterances " % (
            len(results[mm_type]),
            len(mm_data),
        )

        summaries.append(
            html.Tr(
                [
                    html.Td(children=mm_type.upper()),
                    html.Td(
                        html.Div(
                            [
                                dbc.Button(
                                    summary,
                                    id={"type": mm_type, "index": idx},
                                ),
                                html.Span("demonstrate %s." % summary_str_map.get(mm_type, mm_type))
                            ]
                        )
                    ),
                ]
            )
        )
        idx += 1

    return (
        html.Div(
            [
                html.Thead(html.Tr([html.Th("Clinical Skill"), html.Th("")])),
                html.Tbody(summaries),
            ]
        ),
        results["miti"],
        results["emotions"],
        results["empathy"],
        results["cog_dist"],
    )


@callback(
    Output("dialogue-textbox-card", "type"),
    [
        Input({"type": "miti", "index": ALL}, "n_clicks"),
        Input({"type": "emotions", "index": ALL}, "n_clicks"),
        Input({"type": "empathy", "index": ALL}, "n_clicks"),
        Input({"type": "cog_dist", "index": ALL}, "n_clicks"),
    ],
    [
        State("miti-summary-idxs", "data"),
        State("emotions-summary-idxs", "data"),
        State("empathy-summary-idxs", "data"),
        State("cog-dist-summary-idxs", "data"),
    ],
)
def query_summary(
    miti_clicks,
    emotions_clicks,
    empathy_clicks,
    cog_dist_clicks,
    miti_idxs,
    emotions_idxs,
    empathy_idxs,
    cog_dist_idxs,
):
    from dash.exceptions import PreventUpdate

    raise PreventUpdate
    # triggered_id = ctx.triggered_id
    print(triggered_id)

    # breakpoint()
    if triggered_id["type"] == "miti":
        pass
    elif triggered_id["type"] == "emotions":
        pass
    elif triggered_id["type"] == "empathy":
        pass
    elif triggered_id["type"] == "cog-dist":
        pass

    # breakpoint()
