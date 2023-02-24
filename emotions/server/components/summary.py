"""
Summary component.
"""

from dash import html, dcc


summary = html.Div()
miti_query_idxs = dcc.Store(id="miti-summary-idxs")
emotions_query_idxs = dcc.Store(id="emotions-summary-idxs")
empathy_query_idxs = dcc.Store(id="empathy-summary-idxs")
cog_dist_query_idxs = dcc.Store(id="cog-dist-summary-idxs")
