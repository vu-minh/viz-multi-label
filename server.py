import os
import ssl
import dash
from flask import send_from_directory
from flask_caching import Cache
import dash_bootstrap_components as dbc


# avoid certificate-verify SSL error in urllib
ssl._create_default_https_context = ssl._create_unverified_context

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# prevent errors when callbacks are loaded (from import) before layout
app.config['suppress_callback_exceptions'] = True

# for running app with `gunicorn`, should define `server` instance:
server = app.server

# serving css locally
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# Redis cache
CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'localhost:6379')
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)
