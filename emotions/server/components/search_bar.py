"""
Search bar component.
"""

from dash import html, dcc

search_bar = dcc.Input(
    id="search_bar",
    type="text",
    debounce=True,
    placeholder="Enter query."
)

search_bar_component = html.Div([search_bar])
