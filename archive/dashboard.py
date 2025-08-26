import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import re
from pathlib import Path
import flask
import numpy as np
import base64
from collections import defaultdict
import functools

# --- CONFIGURATION ---
# IMPORTANT: Adjust these paths to your local setup.
output_root = Path(r"C:\Users\bayer\MPI\embeddings\archive\results")
eeg_data_root = Path(r"E:\Cris_Work\preproc")

# --- UTILITY FUNCTIONS ---
# These functions remain largely the same, but with minor adjustments for clarity.

def parse_config_string(filename):
    """Parse T{t_start}-{t_end}_B{band}_CH{channels_label} from filename"""
    pattern = r'T(\d+|None)-(\d+|None)_B(?:\(([^)]+)\)|(None))_CH([^\.]+)'
    match = re.search(pattern, filename)
    
    if match:
        t_start, t_end, b_tuple, b_none, channels = match.groups()
        if b_tuple:
            band = f"({b_tuple})"
        elif b_none:
            band = b_none
        else:
            band = ""
        return {
            'T': f"{t_start}-{t_end}",
            'B': band,
            'CH': channels,
            'full_config': f"T{t_start}-{t_end}_B{band}_CH{channels}"
        }
    return None

def get_available_data(output_root):
    """Get all available data and parse configurations"""
    data_map = {
        'html_files': defaultdict(list),
        'overview_exploration': [],
        'subject_exploration': defaultdict(list),
        'config_options': {'T': set(), 'B': set(), 'CH': set()}
    }
    
    subject_dirs = list(output_root.glob("sub-*"))
    
    for subject_dir in subject_dirs:
        subject_key = subject_dir.name
                
        html_files = list(subject_dir.glob("*.html"))
        for html_file in html_files:
            config_info = parse_config_string(html_file.name)
            if config_info:
                data_map['html_files'][subject_key].append((config_info, html_file))
                data_map['config_options']['T'].add(config_info['T'])
                data_map['config_options']['B'].add(config_info['B'])
                data_map['config_options']['CH'].add(config_info['CH'])
    
    overview_dir = output_root / "overview_exploration"
    if overview_dir.exists():
        data_map['overview_exploration'] = list(overview_dir.glob("*.png"))
    
    exploration_dir = output_root / "exploration"
    if exploration_dir.exists():
        subject_dirs = exploration_dir.glob("sub-*")
        for subject_dir in subject_dirs:
            subject_key = subject_dir.name
            png_files = list(subject_dir.glob("*.png"))
            for png_file in png_files:
                config_info = parse_config_string(png_file.name)
                if config_info:
                    data_map['subject_exploration'][subject_key].append((config_info, png_file))
                else:
                    data_map['subject_exploration'][subject_key].append((None, png_file))
    
    for key in data_map['config_options']:
        data_map['config_options'][key] = sorted(list(data_map['config_options'][key]))
    
    return data_map

def encode_image_to_base64(image_path):
    """Convert image to base64 for display"""
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def create_image_component(image_path, scale=0.7, max_width=400):
    """Create HTML component for displaying images"""
    encoded_image = encode_image_to_base64(image_path)
    if not encoded_image:
        return html.Div(f"Error loading {image_path.name}")
    
    return html.Div([
        html.H5(image_path.stem, style={'margin': '5px 0', 'fontSize': '12px', 'textAlign': 'center'}),
        html.Img(
            src=encoded_image,
            style={
                'maxWidth': f'{max_width}px',
                'height': 'auto',
                'border': '1px solid #ddd',
                'borderRadius': '4px'
            }
        )
    ], style={
        'margin': '5px',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'textAlign': 'center'
    })

def create_html_iframe_component(src_url, subject, config, html_only_mode=False):
    """Create HTML component for displaying HTML plots with adaptive sizing"""
    if html_only_mode:
        # Smaller size for grid layout when only showing HTML plots
        width = '400px'
        height = '350px'
        title_style = {'fontSize': '11px', 'margin': '5px 0', 'textAlign': 'center', 'color': '#2c3e50'}
        container_style = {
            'margin': '8px',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'textAlign': 'center',
            'border': '1px solid #ddd',
            'borderRadius': '8px',
            'padding': '5px',
            'backgroundColor': '#fff'
        }
    else:
        # Larger size for mixed view
        width = '600px'
        height = '500px'
        title_style = {'fontSize': '13px', 'margin': '10px 0', 'textAlign': 'center', 'color': '#2c3e50'}
        container_style = {
            'margin': '10px',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'textAlign': 'center'
        }
    
    return html.Div([
        html.Div(f"{subject} - {config}", style=title_style),
        html.Iframe(
            src=src_url,
            style={
                'width': width,
                'height': height,
                'border': '1px solid #ddd',
                'borderRadius': '4px'
            }
        )
    ], style=container_style)

def configs_match(selected_T, selected_B, selected_CH, config_info):
    return (
        (not selected_T or config_info['T'] in selected_T) and
        (not selected_B or config_info['B'] in selected_B) and
        (not selected_CH or config_info['CH'] in selected_CH)
    )

# --- MAIN DASHBOARD APP ---
def launch_dashboard():
    app = dash.Dash(__name__)
    server = app.server
    
    # Load data ONCE when the app starts
    print("Pre-loading data map...")
    data_map = get_available_data(output_root)
    subjects = sorted(set(
        list(data_map['html_files'].keys()) + 
        list(data_map['subject_exploration'].keys())
    ))
    print("Data pre-loading complete.")
    
    @server.route("/plots/<subject>/<filename>")
    def serve_html_plot(subject, filename):
        html_path = output_root / subject / filename
        if html_path.exists():
            return flask.send_file(html_path)
        else:
            return "File not found", 404
    
    app.layout = html.Div([
        html.H1("EEG Analysis Dashboard", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        html.Div([
            html.Div([
                html.Label("View Mode:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Checklist(
                    id='view-mode-checks',
                    options=[
                        {'label': 'Exploration', 'value': 'exploration'},
                        {'label': 'Overview Exploration', 'value': 'overview'}
                    ],
                    value=['exploration'],
                    inline=True,
                    style={'marginBottom': '10px'}
                )
            ], style={'width': '60%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Checklist(
                    id='special-options',
                    options=[{'label': 'Include HTML files', 'value': 'html'}],
                    value=['html'],
                    inline=True
                )
            ], style={'width': '35%', 'display': 'inline-block', 'textAlign': 'right'})
        ], style={'margin': '20px 0', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'}),
        
        html.Div([
            html.H3("Subject Selection", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '5px'}),
            html.Div([
                html.Div([
                    html.Label("Select Subjects:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Checklist(
                        id='select-all-subjects',
                        options=[{'label': 'All', 'value': 'all'}],
                        value=[],
                        inline=True,
                        style={'marginBottom': '5px'}
                    ),
                    dcc.Dropdown(
                        id='subject-dropdown',
                        options=[{'label': s, 'value': s} for s in subjects],
                        multi=True,
                        placeholder="Select subjects or use range below"
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            ])
        ], style={'margin': '20px 0', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}),
        
        html.Div([
            html.H3("Configuration Selection", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '5px'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Time (T):", style={'fontWeight': 'bold', 'display': 'inline-block', 'marginRight': '10px'}),
                        dcc.Checklist(
                            id='select-all-T',
                            options=[{'label': 'All', 'value': 'all'}],
                            value=[],
                            inline=True,
                            style={'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='T-dropdown',
                        options=[{'label': t, 'value': t} for t in data_map['config_options']['T']],
                        multi=True,
                        placeholder="Select time ranges"
                    )
                ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Div([
                        html.Label("Band (B):", style={'fontWeight': 'bold', 'display': 'inline-block', 'marginRight': '10px'}),
                        dcc.Checklist(
                            id='select-all-B',
                            options=[{'label': 'All', 'value': 'all'}],
                            value=[],
                            inline=True,
                            style={'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='B-dropdown',
                        options=[{'label': b, 'value': b} for b in data_map['config_options']['B']],
                        multi=True,
                        placeholder="Select bands"
                    )
                ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Div([
                        html.Label("Channels (CH):", style={'fontWeight': 'bold', 'display': 'inline-block', 'marginRight': '10px'}),
                        dcc.Checklist(
                            id='select-all-CH',
                            options=[{'label': 'All', 'value': 'all'}],
                            value=[],
                            inline=True,
                            style={'display': 'inline-block'}
                        )
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='CH-dropdown',
                        options=[{'label': ch, 'value': ch} for ch in data_map['config_options']['CH']],
                        multi=True,
                        placeholder="Select channels"
                    )
                ], style={'width': '32%', 'display': 'inline-block'})
            ])
        ], style={'margin': '20px 0', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}),
        
        html.Div([
            html.Button("🔄 Refresh Dashboard", id="refresh-button", n_clicks=0,
                        style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                               'padding': '12px 25px', 'borderRadius': '6px', 'fontSize': '16px',
                               'cursor': 'pointer', 'marginRight': '15px'}),
            html.Span(id="status-message", style={'marginLeft': '10px', 'fontStyle': 'italic', 'color': '#7f8c8d'})
        ], style={'margin': '30px 0', 'textAlign': 'center'}),
        
        html.Div(id='content-container', style={'marginTop': '20px'})
    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'width': '100%', 'margin': '0 auto'})
    
    @app.callback(
        [Output('T-dropdown', 'value'),
         Output('B-dropdown', 'value'),
         Output('CH-dropdown', 'value'),
         Output('subject-dropdown', 'value')],
        [Input('select-all-T', 'value'),
         Input('select-all-B', 'value'),
         Input('select-all-CH', 'value'),
         Input('select-all-subjects', 'value')],
        [State('T-dropdown', 'value'),
         State('B-dropdown', 'value'),
         State('CH-dropdown', 'value'),
         State('subject-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_select_all(all_T, all_B, all_CH, all_subjects, current_T, current_B, current_CH, current_subjects):
        ctx = callback_context
        if not ctx.triggered:
            return current_T or [], current_B or [], current_CH or [], current_subjects or []
        
        new_T = current_T or []
        new_B = current_B or []
        new_CH = current_CH or []
        new_subjects = current_subjects or []
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'select-all-T':
            if 'all' in (all_T or []):
                new_T = data_map['config_options']['T']
            else:
                new_T = []
                
        elif triggered_id == 'select-all-B':
            if 'all' in (all_B or []):
                new_B = data_map['config_options']['B']
            else:
                new_B = []
                
        elif triggered_id == 'select-all-CH':
            if 'all' in (all_CH or []):
                new_CH = data_map['config_options']['CH']
            else:
                new_CH = []
                
        elif triggered_id == 'select-all-subjects':
            if 'all' in (all_subjects or []):
                new_subjects = subjects
            else:
                new_subjects = []

        return new_T, new_B, new_CH, new_subjects
    
    @app.callback(
        [Output('content-container', 'children'),
         Output('status-message', 'children')],
        [Input("refresh-button", "n_clicks")],
        [State('view-mode-checks', 'value'),
         State('special-options', 'value'),
         State('subject-dropdown', 'value'),
         State('T-dropdown', 'value'),
         State('B-dropdown', 'value'),
         State('CH-dropdown', 'value')],
        prevent_initial_call=False
    )
    def update_dashboard(n_clicks, view_modes, special_options, selected_subjects_list,
                         selected_T, selected_B, selected_CH):
        
        ctx = callback_context
        
        if not n_clicks:
            return html.Div(
                "Configure your filters above and press 'Refresh Dashboard' to load content.",
                style={'textAlign': 'center', 'padding': '50px', 'color': '#7f8c8d'}
            ), ""
        
        selected_subjects = set(selected_subjects_list or [])
        if not selected_subjects:
            selected_subjects = set(subjects)
        
        components = []
        status_parts = []
        
        # Determine layout mode based on what's being displayed
        show_exploration = 'exploration' in (view_modes or [])
        show_html = 'html' in (special_options or [])
        html_only_mode = show_html and not show_exploration
        
        if 'overview' in (view_modes or []) and data_map['overview_exploration']:
            components.append(html.H2("Overview Exploration", 
                                      style={'borderBottom': '3px solid #3498db', 'paddingBottom': '10px', 'marginTop': '30px'}))
            overview_components = [create_image_component(png_path, max_width=500) for png_path in data_map['overview_exploration']]
            components.append(html.Div(overview_components, style={'margin': '20px 0', 'textAlign': 'center'}))
            status_parts.append(f"{len(data_map['overview_exploration'])} overview plots")
        
        html_count = 0
        exploration_count = 0
        
        # If HTML-only mode, collect all HTML plots first for grid layout
        if html_only_mode:
            all_html_components = []
            
            for subject in sorted(selected_subjects):
                if subject in data_map['html_files']:
                    for config_info, html_path in data_map['html_files'][subject]:
                        if configs_match(selected_T, selected_B, selected_CH, config_info):
                            src_url = f"/plots/{subject}/{html_path.name}"
                            html_component = create_html_iframe_component(
                                src_url, subject, config_info['full_config'], html_only_mode=True
                            )
                            all_html_components.append(html_component)
                            html_count += 1
            
            if all_html_components:
                components.append(html.H2("HTML Embeddings Overview", 
                                         style={'borderBottom': '3px solid #3498db', 'paddingBottom': '10px', 'marginTop': '30px'}))
                components.append(html.Div(
                    all_html_components,
                    style={
                        'display': 'flex',
                        'flexWrap': 'wrap',
                        'justifyContent': 'flex-start',
                        'alignItems': 'flex-start',
                        'margin': '20px 0',
                        'gap': '10px'
                    }
                ))
        else:
            # Original subject-by-subject layout for mixed view
            for subject in sorted(selected_subjects):
                subject_configs = set()
                            
                if 'html' in (special_options or []) and subject in data_map['html_files']:
                    for config_info, _ in data_map['html_files'][subject]:
                        if configs_match(selected_T, selected_B, selected_CH, config_info):
                            subject_configs.add(config_info['full_config'])
                
                if subject_configs:
                    components.append(html.H2(f"{subject}", 
                                             style={'marginTop': '40px', 'color': '#2c3e50', 'borderBottom': '2px solid #34495e', 'paddingBottom': '8px'}))
                    
                    for config in sorted(subject_configs):
                        config_components = []
                        
                        if 'html' in (special_options or []) and subject in data_map['html_files']:
                            for config_info, html_path in data_map['html_files'][subject]:
                                if config_info['full_config'] == config:
                                    src_url = f"/plots/{subject}/{html_path.name}"
                                    html_component = create_html_iframe_component(
                                        src_url, subject, config_info['full_config'], html_only_mode=False
                                    )
                                    config_components.append(html_component)
                                    html_count += 1
                                    break
                        
                        if 'exploration' in (view_modes or []) and subject in data_map['subject_exploration']:
                            exploration_for_config = []
                            for config_info, png_path in data_map['subject_exploration'][subject]:
                                if config_info and config_info['full_config'] == config:
                                    exploration_for_config.append(create_image_component(png_path, max_width=300))
                                    exploration_count += 1
                            
                            if exploration_for_config:
                                config_components.append(html.Div(
                                    exploration_for_config,
                                    style={'display': 'inline-block', 'verticalAlign': 'top', 'margin': '10px'}
                                ))
                        
                        if config_components:
                            components.append(html.Div([
                                html.H4(f"{config}", style={'color': '#8e44ad', 'marginBottom': '15px'}),
                                html.Div(config_components, style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start'})
                            ], style={'marginBottom': '30px', 'padding': '15px', 'backgroundColor': '#fdfdfd', 'borderRadius': '8px', 'border': '1px solid #e0e0e0'}))
        
        if not components:
            return html.Div(
                "No data found for the selected filters. Please adjust your selection and try again.",
                style={'textAlign': 'center', 'padding': '50px', 'color': '#e74c3c'}
            ), "No data found"
        
        status_parts.extend([
            f"{html_count} HTML plots", 
            f"{exploration_count} exploration plots"
        ])

        status_msg = f"Loaded: {', '.join(status_parts)}"
        
        return html.Div(components), status_msg
    
    app.run(debug=False, port=8050, host='0.0.0.0')

if __name__ == "__main__":
    launch_dashboard()