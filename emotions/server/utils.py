"""
Utility functions.
"""

from dash import html
import plotly.express as px

COLOR_SCHEME = px.colors.qualitative.Set2


MM_TYPES = [
    "miti_",
    "empathy_",
    "epitome_",
    "custom_",
    "emotion_",
    "cog_dist_",
]


def get_mm_color(mm_name):
    """
    Get color for mm.
    """
    mm_prefixes = MM_TYPES[::-1]
    for idx, prefix in enumerate(mm_prefixes):
        if mm_name.startswith(prefix):
            return COLOR_SCHEME[idx]
    raise ValueError("Unknown MM %s!" % mm_name)


def entname(name):
    return html.Span(
        name,
        style={
            "font-size": "0.8em",
            "font-weight": "bold",
            "line-height": "1",
            "border-radius": "0.35em",
            "text-transform": "uppercase",
            "vertical-align": "middle",
            "margin-left": "0.5rem",
            "margin-right": "0.5rem",
        },
    )


def entbox(children, color):
    return html.Mark(
        children,
        style={
            "background": color,
            "padding": "0.45em 0.6em",
            "margin": "0 0.25em",
            "line-height": "1",
            "border-radius": "0.35em",
        },
    )


def entity(text, entity_name, color):
    assert isinstance(text, str)
    text = [entname(entity_name)] + [text]
    return entbox(text, color)


