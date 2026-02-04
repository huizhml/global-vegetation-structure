import requests
from flask import Response
from dash import Dash, html

app = Dash(__name__)

@app.server.route("/widget")
def proxy_widget():
    resp = requests.get("http://localhost:8866")
    return Response(resp.content, resp.status_code, resp.headers.items())

app.layout = html.Iframe(src="/widget", style={"width": "100%", "height": "800px"})

if __name__ == "__main__":
    app.run(debug=True, port=8050)