"""
Search callbacks.
"""
from dash import callback, Input, Output, State, ALL, ctx, no_update
from dash.exceptions import PreventUpdate
from emotions.server.components.dialogue_dropdown import textbox
from emotions.server.components import (
    search_results_display,
    search_result_idxs,
    search_bar,
)


def build_search_result_component(
    utterance, speaker, idx, prev_utt=None, prev_speaker=None, prev_idx=None
):
    """
    Search result
    """
    result_textbox = textbox(utterance, speaker, idx)
    if prev_utt and prev_speaker and prev_idx is not None and prev_idx >= 0:
        prev_textbox = textbox(prev_utt, prev_speaker, prev_idx)

        return [prev_textbox, result_textbox]

    return [result_textbox]


@callback(
    [
        Output(search_results_display, "children"),
        Output(search_result_idxs, "data"),
    ],
    [
        Input(search_bar, "value"),
        Input("conversation-encoding", "data"),
    ],
    prevent_initial_call=True
)
def search(query, data):
    query_result_idxs = [0, 2, 4]

    curr_idx = 0
    result_idx = query_result_idxs[curr_idx]
    prev_idx = None
    if result_idx > 0:
        prev_idx = result_idx - 1

    curr_result = data[result_idx]
    utterance = curr_result["utterance"]
    speaker = curr_result["speaker"]

    prev_utt = None
    prev_speaker = None
    if prev_idx is not None:
        prev_utt = data[prev_idx]["utterance"]
        prev_speaker = data[prev_idx]["speaker"]

    result_component = build_search_result_component(
        utterance, speaker, result_idx, prev_utt, prev_speaker, prev_idx
    )

    return [result_component, query_result_idxs]
