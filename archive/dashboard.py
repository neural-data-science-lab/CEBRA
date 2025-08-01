import dash
from dash import dcc, html, Input, Output
import re
from pathlib import Path
import flask

output_root = Path(r"C:\Users\bayer\MPI\embeddings\archive\results")  # Path where .html files are saved

def get_available_embeddings(output_root):
    subject_dirs = list(output_root.glob("sub-*"))
    subject_config_map = {}

    for subject_dir in subject_dirs:
        subject_key = subject_dir.name
        html_files = subject_dir.glob("*.html")
        subject_config_map[subject_key] = []
        for html_file in html_files:
            match = re.search(rf"{subject_key}_(.+?)_embedding\.html", html_file.name)
            if match:
                config_str = match.group(1)
                subject_config_map[subject_key].append((config_str, html_file))
    return subject_config_map

def make_scaled_iframe(src_url, scale=0.7, orig_width=700, orig_height=500):
    # orig_width/height = the actual size of your HTML plot content
    # scale = how much to shrink the iframe content (0.7 means 70% size)
    return html.Div(
        html.Iframe(
            src=src_url,
            style={
                'width': f'{orig_width}px',
                'height': f'{orig_height}px',
                'border': 'none',
                'transform': f'scale({scale})',
                'transform-origin': 'top left',
                'pointer-events': 'auto',  # Keep interactivity (zoom, hover)
                'display': 'block',
            },
            sandbox="allow-scripts allow-same-origin"
        ),
        style={
            'width': f'{int(orig_width * scale)}px',  # Container shrinks to scaled size
            'height': f'{int(orig_height * scale)}px',
            'overflow': 'hidden',                      # Hide scrollbars
            'margin': '2px',
            'boxSizing': 'border-box',
        }
    )

def launch_dashboard():
    app = dash.Dash(__name__)
    server = app.server  # expose Flask app for serving files

    subject_config_map = get_available_embeddings(output_root)
    subjects = sorted(subject_config_map.keys())

    # Collect all config_strs from available files
    all_configs = sorted(set(
        config for configs in subject_config_map.values() for config, _ in configs
    ))

    @server.route("/plots/<subject>/<filename>")
    def serve_html_plot(subject, filename):
        html_path = output_root / subject / filename
        if html_path.exists():
            return flask.send_file(html_path)
        else:
            return "File not found", 404

    app.layout = html.Div([
        html.H1("Live EEG CEBRA Embeddings Dashboard"),

        html.Div([
            html.Label("Filter by Subject:"),
            dcc.Dropdown(
                id='subject-dropdown',
                options=[{'label': s, 'value': s} for s in subjects],
                placeholder="Select a subject"
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Filter by Configuration:"),
            dcc.Dropdown(
                id='config-dropdown',
                options=[{'label': c, 'value': c} for c in all_configs],
                placeholder="Select a configuration"
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Button("Refresh", id="refresh-button", n_clicks=0),

        html.Div(id='plots-container')
    ], style={'padding': '20px'})

    @app.callback(
        Output('plots-container', 'children'),
        Input('subject-dropdown', 'value'),
        Input('config-dropdown', 'value'),
        Input("refresh-button", "n_clicks")
    )
    def update_iframes(selected_subject, selected_config, n_clicks):
        subject_config_map = get_available_embeddings(output_root)
        iframe_elements = []

        if selected_subject and selected_config:
            return html.Div("Please select either Subject OR Configuration, not both.")

        elif selected_subject:
            configs = subject_config_map.get(selected_subject, [])
            for config_str, filepath in configs:
                src_url = f"/plots/{selected_subject}/{filepath.name}"
                iframe_elements.append(make_scaled_iframe(src_url, scale=0.7))

        elif selected_config:
            for subject, configs in subject_config_map.items():
                for config_str, filepath in configs:
                    if config_str == selected_config:
                        src_url = f"/plots/{subject}/{filepath.name}"
                        iframe_elements.append(make_scaled_iframe(src_url, scale=0.7))

        else:
            return html.Div("Please select a Subject or Configuration to display plots.")

        if not iframe_elements:
            return html.Div("No HTML files found.")

        return html.Div(
            iframe_elements,
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'flex-start',
                'gap': '3px',
            }
        )

    app.run(debug=False, port=8050)

if __name__ == "__main__":
    launch_dashboard()
