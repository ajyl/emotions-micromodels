"""
Search callbacks.
"""
from dash import callback, Input, Output, State, ALL, ctx, no_update
from dash.exceptions import PreventUpdate
from emotions.server.backend.search import evaluate, parse_query
from emotions.server.components.dialogue_dropdown import textbox
from emotions.server.components import (
    search_results_display,
    search_result_idxs,
    current_search_idx,
    current_search_idx_ptr,
    search_bar,
)
from emotions.server.callbacks.summarize import get_default_query


def build_search_result_component(data, result_idx):
    """
    Search result
    """
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

    result_textbox = textbox(
        utterance, speaker, result_idx, "search_result_textbox"
    )
    if prev_utt and prev_speaker and prev_idx is not None and prev_idx >= 0:
        prev_textbox = textbox(
            prev_utt, prev_speaker, prev_idx, "search_result_textbox"
        )

        return [prev_textbox, result_textbox]

    return [result_textbox]


@callback(
    [
        Output(search_results_display, "children"),
        Output(search_result_idxs, "data"),
        Output(current_search_idx_ptr, "data"),
        Output(current_search_idx, "data"),
        Output("search-result-prev", "style"),
        Output("search-result-next", "style"),
    ],
    [
        Input(search_bar, "value"),
        Input("search-result-prev", "n_clicks"),
        Input("search-result-next", "n_clicks"),
        Input(search_result_idxs, "data"),
        Input(current_search_idx_ptr, "data"),
        Input("conversation-encoding", "data"),
    ],
    #[
    #    State("conversation-encoding", "data"),
    #],
    prevent_initial_call=True,
)
def search(
    query,
    prev_button,
    next_button,
    search_result_idxs,
    curr_idx,
    data,
):
    triggered_id = ctx.triggered_id
    parsed_query = None
    next_arrow = {
        "display": "block",
        "height": "10%",
        "width": "8%",
        "horizontalAlign": "center",
    }
    prev_arrow = {
        "display": "block",
        "height": "10%",
        "width": "8%",
        "horizontalAlign": "center",
    }

    if triggered_id in ["search-result-prev", "search-result-next"]:
        assert curr_idx is not None
        assert search_result_idxs is not None

        if triggered_id == "search-result-next":
            if curr_idx + 1 >= len(search_result_idxs):
                return [
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                ]
            curr_idx += 1
        if triggered_id == "search-result-prev":
            if curr_idx == 0:
                return [no_update, no_update, no_update, no_update]
            curr_idx -= 1

        result_idx = search_result_idxs[curr_idx]
        result_component = build_search_result_component(data, result_idx)
        # n_clicks[result_idx] += 1

        if curr_idx >= len(search_result_idxs) - 1:
            next_arrow["display"] = "none"

        if curr_idx <= 0:
            prev_arrow["display"] = "none"

        return [
            result_component,
            search_result_idxs,
            curr_idx,
            result_idx,
            prev_arrow,
            next_arrow,
        ]

    if query is None or len(query) < 1:
        parsed_query = get_default_query()
        query_result_idxs = set()
        for mm_type, query_obj in parsed_query.items():
            speaker = query_obj[1]
            query = ["or"]
            query.extend(query_obj[0])
            if speaker:
                query = ["and", query, ["==", "speaker", speaker]]

            _results = evaluate(query, data)
            query_result_idxs = query_result_idxs.union(_results)
        query_result_idxs = sorted(list(query_result_idxs))

    else:
        parsed_query = parse_query(query)
        # query_result_idxs = evaluate(parsed_query, data)
        query_result_idxs = [1, 4, 8]

    if len(query_result_idxs) < 1:
        return [no_update] * 6

    curr_idx = 0
    result_idx = query_result_idxs[curr_idx]
    result_component = build_search_result_component(data, result_idx)

    if curr_idx >= len(query_result_idxs) - 1:
        next_arrow["display"] = "none"

    if curr_idx <= 0:
        prev_arrow["display"] = "none"

    return [
        result_component,
        query_result_idxs,
        curr_idx,
        result_idx,
        prev_arrow,
        next_arrow,
    ]
