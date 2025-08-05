import dash
from dash import dcc, html, Input, Output, State
import re
from pathlib import Path
import flask

output_root = Path(r"C:\Users\bayer\MPI\embeddings\archive\results")

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

def expand_subject_range(start, end):
    prefix = re.match(r"([a-zA-Z\-]+)", start).group(1)
    start_num = int(re.search(r"(\d+)", start).group(1))
    end_num = int(re.search(r"(\d+)", end).group(1))
    return [f"{prefix}{str(i).zfill(len(str(start_num)))}" for i in range(start_num, end_num + 1)]

def parse_range_input(input_str):
    if not input_str:
        return []
    entries = [e.strip() for e in input_str.split(";") if e.strip()]
    result = []
    for entry in entries:
        if ":" in entry:
            start, end = entry.split(":")
            expanded = expand_subject_range(start.strip(), end.strip())
            result.extend(expanded)
        else:
            result.append(entry)
    return result

def make_scaled_iframe(src_url, scale=0.7, orig_width=700, orig_height=500):
    return html.Div(
        html.Iframe(
            src=src_url,
            style={
                'width': f'{orig_width}px',
                'height': f'{orig_height}px',
                'border': 'none',
                'transform': f'scale({scale})',
                'transform-origin': 'top left',
                'pointer-events': 'auto',
                'display': 'block',
            },
            sandbox="allow-scripts allow-same-origin"
        ),
        style={
            'width': f'{int(orig_width * scale)}px',
            'height': f'{int(orig_height * scale)}px',
            'overflow': 'hidden',
            'margin': '2px',
            'boxSizing': 'border-box',
        }
    )

def launch_dashboard():
    app = dash.Dash(__name__)
    server = app.server

    # Load the data once here and store globally to avoid re-reading files each callback
    subject_config_map = get_available_embeddings(output_root)
    subjects = sorted(subject_config_map.keys())
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
            html.Label("Select Subjects (dropdown, multi-select):"),
            dcc.Dropdown(
                id='subject-dropdown',
                options=[{'label': s, 'value': s} for s in subjects],
                multi=True,
                placeholder="Select subjects"
            ),
            html.Br(),
            html.Label("Or enter subject ranges (e.g. sub-001:sub-005; sub-010):"),
            dcc.Input(
                id='subject-range-input',
                type='text',
                placeholder="Enter subject ranges separated by ;",
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

        html.Div([
            html.Label("Select Configurations (dropdown, multi-select):"),
            dcc.Dropdown(
                id='config-dropdown',
                options=[{'label': c, 'value': c} for c in all_configs],
                multi=True,
                placeholder="Select configurations"
            ),
            html.Br(),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Checklist(
                id='va-checkbox',
                options=[{'label': 'Preselect all VA configurations', 'value': 'VA'}],
                value=[],
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'margin': '15px 0'}),

        html.Button("Refresh", id="refresh-button", n_clicks=0),

        # Start with empty container
        html.Div(id='plots-container', style={'marginTop': '20px'})
    ], style={'padding': '20px'})

    @app.callback(
        Output('plots-container', 'children'),
        Input("refresh-button", "n_clicks"),
        State('subject-dropdown', 'value'),
        State('subject-range-input', 'value'),
        State('config-dropdown', 'value'),
        State('va-checkbox', 'value'),
    )
    def update_iframes(n_clicks, subject_dropdown_vals, subject_range_str, config_dropdown_vals, va_checkbox_values):
        if n_clicks == 0:
            # No refresh yet, do not load anything
            return html.Div("Please configure filters and press Refresh to load plots.")

        # Use cached subject_config_map loaded once (do not reload from disk)
        iframe_elements = []

        selected_subjects = set(subject_dropdown_vals or [])
        selected_subjects.update(parse_range_input(subject_range_str))

        selected_configs = set(config_dropdown_vals or [])

        if not selected_subjects:
            selected_subjects = set(subject_config_map.keys())

        if not selected_configs:
            all_configs_set = set(
                config for configs in subject_config_map.values() for config, _ in configs
            )
            selected_configs = all_configs_set

        if 'VA' in (va_checkbox_values or []):
            va_entries = []
            for subject, configs in subject_config_map.items():
                for config_str, filepath in configs:
                    if "VA_" in filepath.name:
                        va_entries.append((subject, config_str, filepath))

            if not va_entries:
                return html.Div("No VA configurations found.")

            for subject, config_str, filepath in va_entries:
                if subject in selected_subjects and config_str in selected_configs:
                    src_url = f"/plots/{subject}/{filepath.name}"
                    iframe_elements.append(make_scaled_iframe(src_url, scale=0.7))

        else:
            for subject in sorted(selected_subjects):
                configs = subject_config_map.get(subject, [])
                for config_str, filepath in configs:
                    if config_str in selected_configs:
                        src_url = f"/plots/{subject}/{filepath.name}"
                        iframe_elements.append(make_scaled_iframe(src_url, scale=0.7))

        if not iframe_elements:
            return html.Div("No HTML files found for selected filters.")

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
