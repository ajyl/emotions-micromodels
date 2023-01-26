"""
Dash Server
"""

from emotions.server.init_server import app, server
from emotions.server.callbacks.encode import encode
from emotions.server.callbacks.search import search
from emotions.server.callbacks.dialogue import display_dialogue


if __name__ == "__main__":
    app.run_server(debug=True)
