"""
Utility function to create micromodel component.
"""

import pandas as pd
import plotly.express as px
from dash import dcc
from emotions.server.utils import (
    MM_TYPES,
    get_mm_color,
    COLOR_SCHEME,
    get_legend_name,
)
from emotions.constants import PATIENT


def build_micromodel_component(micromodel_results, utterance, speaker):
    """
    Return bar graph.
    micromodel_results: response_obj["micromodels"]
    """
    client_mms = ["emotion", "custom", "cog_dist"]
    height = 1800
    if speaker == PATIENT:
        height = 1500
    mms = list(micromodel_results.keys())
    sorted_mms = []
    for mm_type in MM_TYPES[::-1]:
        sorted_mms.extend([mm for mm in mms if mm.startswith(mm_type)])

    data = []
    for mm in sorted_mms:
        if speaker == PATIENT and not any(
            mm.startswith(_client_mm) for _client_mm in client_mms
        ):
            continue
        mm_result = micromodel_results[mm]
        data.append(
            (
                mm,
                utterance,
                max(mm_result["max_score"], 0),
                mm_result["segment"],
                get_mm_color(mm),
                mm_result.get("top_k_scores", [[None]])[0][0],
            )
        )

    data = pd.DataFrame(
        data=data,
        columns=[
            "Micromodel",
            "query",
            "score",
            "segment",
            "color",
            "similar_seed",
        ],
    )
    micromodel_fig = px.bar(
        data_frame=data,
        x="score",
        y="Micromodel",
        color="color",
        hover_name="Micromodel",
        hover_data=["Micromodel", "score", "similar_seed"],
        custom_data=["segment", "query"],
        orientation="h",
        height=height,
        color_discrete_sequence=COLOR_SCHEME,
    )
    micromodel_fig.update_coloraxes(showscale=False)
    micromodel_fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            title="",
            traceorder="reversed",
        ),
    )
    micromodel_fig.update_xaxes(range=[0, 1], dtick=0.1)
    micromodel_fig.for_each_trace(lambda t: t.update(name=get_legend_name(t)))

    return dcc.Graph(
        id="micromodel-results",
        figure=micromodel_fig,
        style={"display": "block"},
    )
