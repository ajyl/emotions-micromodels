"""
Search results component.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from emotions.constants import THERAPIST, PATIENT


search_results_display = html.Div(
    id="search-result-display-component",
)
search_result_idxs = dcc.Store(id="search-results-storage")

search_result_component = html.Div(
    [
        search_result_idxs,
        search_results_display,
        html.Div(
            [
                # Left-Arrow
                html.Button(
                    id="search-result-prev",
                    children=[
                        html.Img(
                            src=r"assets/left_arrow.png",
                            alt="image",
                            style={"height": "70%", "width": "70%"},
                        )
                    ],
                    style={
                        "height": "10%",
                        "width": "8%",
                        "margin-left": "42%",
                    },
                ),
                # Right-Arrow
                html.Button(
                    id="search-result-next",
                    children=[
                        html.Img(
                            src=r"assets/right_arrow.png",
                            alt="image",
                            style={"height": "70%", "width": "70%"},
                        )
                    ],
                    style={
                        "height": "10%",
                        "width": "8%",
                        "horizontalAlign": "center",
                    },
                ),
            ],
            style={"horizontalAlign": "center"},
        ),
    ],
    id="search-result-component",
)
