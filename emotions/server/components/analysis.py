"""
Dash Components
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from emotions.config import ED_EMOTIONS


def get_emotion_mms():
    return ["emotion_%s" % emotion for emotion in ED_EMOTIONS] + [
        "custom_%s" % emotion for emotion in ED_EMOTIONS
    ]


utterance_component = dbc.Card(
    [
        dcc.Store(id="annotated-utterance-storage"),
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Emotions", tab_id="utterance-tab-1"),
                    dbc.Tab(label="MITI Codes", tab_id="utterance-tab-2"),
                    dbc.Tab(label="Empathy", tab_id="utterance-tab-3"),
                ],
                id="utterance-tabs",
                active_tab="utterance-tab-1",
            )
        ),
        dbc.CardBody(
            [
                html.H5(id="speaker"),
                html.Br(),
                html.Div(id="utterance"),
            ],
            id="utterance-card",
        ),
    ]
)


micromodel_component = dbc.Card(
    [
        dcc.Graph(id="micromodel-results", style={"display": "none"}),
        dcc.Store(id="emotion-classification-storage"),
    ]
)


emotion_analysis = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Emotion Analysis"),
            html.Table(
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
                ],
            ),
        ]
    )
)


empathy_analysis = dbc.Card(
    [
        html.H4("Empathy Analysis"),
        html.Pre(id="empathy"),
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(""),
                            html.Th("Emotional Reaction"),
                            html.Th("Interpretation"),
                            html.Th("Exploration"),
                        ],
                    ),
                    style={"hidden": True},
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    "Micromodels",
                                    style={"font-weight": "bold"},
                                ),
                                html.Td(id="emotional_reaction_mm"),
                                html.Td(id="interpretation_mm"),
                                html.Td(id="exploration_mm"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(
                                    "Epitome",
                                    style={"font-weight": "bold"},
                                ),
                                html.Td(id="emotional_reaction_epitome"),
                                html.Td(id="interpretation_epitome"),
                                html.Td(id="exploration_epitome"),
                            ]
                        ),
                    ]
                ),
            ],
            # style={"display": "none"},
            id="empathy-table",
        ),
    ]
)

pair_analysis = dbc.Card([html.H4("MITI Analysis"), html.Pre(id="pair")])


analysis = html.Div(
    children=[
        html.Div(
            [
                utterance_component,
                html.Br(),
                micromodel_component,
                html.Br(),
                emotion_analysis,
                empathy_analysis,
                pair_analysis,
            ],
            style={"display": "none"},
            id="analysis",
        ),
    ],
)
