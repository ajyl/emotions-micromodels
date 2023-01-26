"""
Utility functions to create explanation component.
"""

from dash import no_update
from emotions.server.init_server import EMOTION_EXPL


def build_explanation_component(explanation_dropdown, emotion_classifications):
    """
    Build explanation component.
    """
    visual_idx = None
    if explanation_dropdown in EMOTION_EXPL.feature_names:
        visual_idx = EMOTION_EXPL.feature_names.index(explanation_dropdown)

    global_fig = EMOTION_EXPL.visualize(visual_idx)
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


def update_global_exp(explanation_dropdown, emotion_classifications):
    """
    Update global explanation figure.
    """
    visual_idx = None
    if explanation_dropdown in EMOTION_EXPL.feature_names:
        visual_idx = EMOTION_EXPL.feature_names.index(explanation_dropdown)

    global_fig = EMOTION_EXPL.visualize(visual_idx)
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

    return [
        no_update,
        no_update,
        no_update,
        no_update,
        global_fig,
        no_update,
        no_update,
        no_update,
        no_update,
    ]
