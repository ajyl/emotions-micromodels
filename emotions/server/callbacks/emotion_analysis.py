"""
Utility functions to build emotion analysis component.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
from emotions.server.init_server import EMOTION_EXPL


def build_emotion_analysis_component(emotions, expl_dropdown):
    """
    emotions: List[List[predictions], List[scores]]
    """
    emotion_classifications = emotions[0]
    emotion_scores = emotions[1]

    # Emotions - Explanations
    global_exp = EMOTION_EXPL
    visual_idx = None
    if expl_dropdown in global_exp.feature_names:
        visual_idx = global_exp.feature_names.index(expl_dropdown)

    global_fig = global_exp.visualize(visual_idx)
    global_fig.update_layout(legend_title="Predictions:")

    if visual_idx is not None:
        global_fig["data"] = global_fig["data"][:-1]
        orig_idxs = {_idx: x for _idx, x in enumerate(global_fig["data"])}
        orig_order = [x["name"] for x in global_fig["data"]]
        global_fig["data"] = [
            orig_idxs[orig_order.index(emotion)]
            for emotion in emotion_classifications
        ]
        global_fig["layout"].pop("xaxis")
        global_fig["layout"].pop("yaxis")

    component = html.Div(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Emotion"),
                        html.Th("Confidence Score"),
                    ]
                ),
                style={"hidden": True},
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                id="emotion_1",
                                children=emotion_classifications[0],
                            ),
                            html.Td(
                                id="emotion_score_1",
                                children=round(emotion_scores[0], 3),
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(
                                id="emotion_2",
                                children=emotion_classifications[1],
                            ),
                            html.Td(
                                id="emotion_score_2",
                                children=round(emotion_scores[1], 3),
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(
                                id="emotion_3",
                                children=emotion_classifications[2],
                            ),
                            html.Td(
                                id="emotion_score_3",
                                children=round(emotion_scores[2], 3),
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )
    return component
