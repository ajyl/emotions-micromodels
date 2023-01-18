"""
Dash Server
"""

from emotions.server.init_server import app, server
from emotions.server.callbacks.encode import encode

if __name__ == "__main__":
    app.run_server(debug=True)
