"""
Components for Emotion Classification.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from emotions.config import EMOTIONS


def get_emotion_mms():
    return ["emotion_%s" % emotion for emotion in EMOTIONS] + [
        "custom_%s" % emotion for emotion in EMOTIONS
    ]


emotion_analysis = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Emotion Classification Results"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [html.Th("Emotion"), html.Th("Confidence Score")]
                        ),
                        style={"hidden": True},
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(id="emotion_1"),
                                    html.Td(id="emotion_score_1"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td(id="emotion_2"),
                                    html.Td(id="emotion_score_2"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td(id="emotion_3"),
                                    html.Td(id="emotion_score_3"),
                                ]
                            ),
                        ]
                    ),
                ],
                style={"display": "none"},
                id="emotion-classification-table",
            ),
            html.Br(),
            html.H6("Explanations:"),
            dbc.Card(
                [
                    dcc.Dropdown(
                        id="global-explanation-feature-dropdown",
                        options=["Overall"] + get_emotion_mms(),
                        multi=False,
                        value="Overall",
                    ),
                    dcc.Graph(
                        id="global-explanation",
                        style={"display": "none"},
                    ),
                    dcc.Store(id="global-explanation-storage"),
                ],
            ),
        ]
    )
)
