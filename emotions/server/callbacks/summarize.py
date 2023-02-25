"""
Callback method to generate summary.
"""

from dash import callback, Input, Output, State, ALL, ctx, no_update, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from emotions.server.components import search_bar, summary as summary_component
from emotions.server.backend.search import evaluate, parse_query
from emotions.config import (
    EMOTIONS,
    PHQ9,
    OTHER,
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
        "phq9": (
            [[">=", "phq9_%s" % phq9, 0.8] for phq9 in PHQ9],
            PATIENT,
        ),
        "other": (
            [[">=", "other_%s" % behavior, 0.8] for behavior in OTHER],
            PATIENT,
        )
    }
    return queries


@callback(
    [
        Output(summary_component, "children"),
        Output("miti-summary-idxs", "data"),
        Output("emotions-summary-idxs", "data"),
        Output("empathy-summary-idxs", "data"),
        Output("phq9-summary-idxs", "data"),
        Output("other-summary-idxs", "data"),
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
            "phq9": "PHQ9 behaviors",
            "other": "other mental-health related behaviors"
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
                                html.Span(
                                    "demonstrate %s."
                                    % summary_str_map.get(mm_type, mm_type)
                                ),
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
        results["phq9"],
        results["other"],
    )


@callback(
    Output({"type": "dialogue-textbox-card", "index": ALL}, "style"),
    [
        Input({"type": "miti", "index": ALL}, "n_clicks"),
        Input({"type": "emotions", "index": ALL}, "n_clicks"),
        Input({"type": "empathy", "index": ALL}, "n_clicks"),
        Input({"type": "phq9", "index": ALL}, "n_clicks"),
        Input({"type": "other", "index": ALL}, "n_clicks"),
        Input(search_bar, "value"),
        Input("conversation-encoding", "data"),
    ],
    [
        State("miti-summary-idxs", "data"),
        State("emotions-summary-idxs", "data"),
        State("empathy-summary-idxs", "data"),
        State("phq9-summary-idxs", "data"),
        State("other-summary-idxs", "data"),
        State({"type": "dialogue-textbox-card", "index": ALL}, "style"),
    ],
    prevent_initial_call=True,
)
def query_summary(
    miti_clicks,
    emotions_clicks,
    empathy_clicks,
    phq9_clicks,
    other_clicks,
    query,
    mm_data,
    miti_idxs,
    emotions_idxs,
    empathy_idxs,
    phq9_idxs,
    other_idxs,
    styles,
):
    _sum = sum(
        [
            x
            for x in miti_clicks
            + emotions_clicks
            + empathy_clicks
            + phq9_clicks
            + other_clicks
            if x is not None
        ]
    )
    if _sum <= 0 and (query is None or len(query) < 1):
        raise PreventUpdate

    triggered_id = ctx.triggered_id

    if triggered_id == "conversation-encoding":
        raise PreventUpdate

    query_idxs = []
    if triggered_id == "search_bar":
        parsed_query = parse_query(query)
        query_idxs = evaluate(parsed_query, mm_data)

    else:

        if triggered_id["type"] == "miti":
            query_idxs = miti_idxs
        elif triggered_id["type"] == "emotions":
            query_idxs = emotions_idxs
        elif triggered_id["type"] == "empathy":
            query_idxs = empathy_idxs
        elif triggered_id["type"] == "phq9":
            query_idxs = phq9_idxs
        elif triggered_id["type"] == "other":
            query_idxs = other_idxs

    for _idx in range(len(styles)):
        styles[_idx]["opacity"] = 1

    non_idxs = [idx for idx in range(len(styles)) if idx not in query_idxs]
    for _idx in non_idxs:
        styles[_idx]["opacity"] = 0.5

    return styles
