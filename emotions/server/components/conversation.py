"""
Conversation component.
"""

from dash import html, dcc

conversation = html.Div(
    [
        html.Div(
            style={
                "max-width": "800px",
                "height": "70vh",
                "margin": 0,
                "overflow-y": "auto",
            },
            id="display-conversation",
        ),
        dcc.Store(id="conversation-encoding")
    ]
)
