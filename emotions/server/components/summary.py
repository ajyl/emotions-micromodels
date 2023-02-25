"""
Summary component.
"""

from dash import html, dcc


summary = html.Div()
miti_query_idxs = dcc.Store(id="miti-summary-idxs")
emotions_query_idxs = dcc.Store(id="emotions-summary-idxs")
empathy_query_idxs = dcc.Store(id="empathy-summary-idxs")
phq9_query_idxs = dcc.Store(id="phq9-summary-idxs")
other_query_idxs = dcc.Store(id="other-summary-idxs")
