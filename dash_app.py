import math
import threading
import random
import time
from datetime import datetime
import sqlite3
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
import base64
import io
import uuid
import numpy as np
import plotly.express as px

# Global vars
sail_polar_data = None
data_ingestion_running = False
interpolator = None
twa_range = None
tws_range = None
plot_mode = "default"
current_session_id = None

# Define values for the "dashboard"
last_displayed_values = {
    'vmg': None,
    'speed': None,
    'net_speed': None,
    'wind_angle': None,
    'wind_speed': None,
    'last_display_time': None
}

current_changes = {
    'vmg': 0,
    'speed': 0,
    'net_speed': 0,
    'wind_angle': 0,
    'wind_speed': 0
}

previous_values = {
    'vmg': [],
    'speed': [],
    'net_speed': [],
    'wind_angle': [],
    'wind_speed': []
}

def initialize_database():
    conn = sqlite3.connect('sailing_data.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time DATETIME,
                        end_time DATETIME NULL)''')
    
    # Create sensor_data table with session support
    cursor.execute('''CREATE TABLE IF NOT EXISTS sensor_data (
                        timestamp DATETIME,
                        session_id TEXT,
                        speed REAL,
                        theoretical_speed REAL,
                        wind_speed REAL,
                        wind_angle REAL,
                        latitude REAL,
                        longitude REAL,
                        vmg REAL,
                        PRIMARY KEY (timestamp, session_id),
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id))''')
    conn.commit()
    return conn

def start_new_session():
    global current_session_id
    current_session_id = str(uuid.uuid4())
    conn = initialize_database()
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO sessions (session_id, start_time)
                     VALUES (?, ?)''', (current_session_id, datetime.now()))
    conn.commit()
    conn.close()
    return current_session_id

def end_current_session():
    global current_session_id
    if current_session_id:
        conn = initialize_database()
        cursor = conn.cursor()
        cursor.execute('''UPDATE sessions 
                         SET end_time = ?
                         WHERE session_id = ?''', (datetime.now(), current_session_id))
        conn.commit()
        conn.close()
        current_session_id = None

def get_sessions():
    conn = initialize_database()
    cursor = conn.cursor()
    cursor.execute('''SELECT session_id, start_time, end_time 
                     FROM sessions 
                     ORDER BY start_time DESC''')
    sessions = cursor.fetchall()
    conn.close()
    
    return [{'label': f"Session {idx + 1}: {start_time}", 
             'value': session_id} 
            for idx, (session_id, start_time, end_time) in enumerate(sessions)]

# Mock sensor data generator
def mock_sensor_data():
    while True:
        speed = random.uniform(0, 20)
        yield speed
        time.sleep(1)

# Load sail polar data from a CSV file
def load_sail_polar_data(contents):
    global interpolator, twa_range, tws_range
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';', index_col=0)
    twa = data.index.values
    tws = data.columns.astype(float).values
    boat_speeds = data.values
    interpolator = RegularGridInterpolator((twa, tws), boat_speeds, method='linear')
    twa_range = (min(twa), max(twa))
    tws_range = (min(tws), max(tws))
    return data

# Simulate wind conditions (wind speed and wind angle)
def simulate_wind_conditions():
    wind_speed = random.uniform(5, 25)
    wind_angle = random.uniform(-180, 180)
    return wind_speed, wind_angle

# Add function to simulate GPS coordinates (after simulate_wind_conditions function)
def simulate_gps_coordinates():
    latitude = random.uniform(60.15, 60.17)
    longitude = random.uniform(24.95, 24.97)
    return latitude, longitude

# Limit a value to a given range
def limit(value, min_value, max_value):
    return max(min_value, min(value, max_value))

# Calculate theoretical speed using interpolated sail polar data
def calculate_theoretical_speed(wind_speed, wind_angle):
    wind_angle = abs(wind_angle)
    global interpolator, twa_range, tws_range
    if interpolator is None:
        print("Error: Interpolator is not initialized.")
        return None
    # Limit the wind angle and wind speed to the valid ranges
    wind_angle_limited = limit(wind_angle, twa_range[0], twa_range[1])
    wind_speed_limited = limit(wind_speed, tws_range[0], tws_range[1])
    # Warn if the values were limited
    if wind_angle != wind_angle_limited or wind_speed != wind_speed_limited:
        print(f"Warning: limited wind angle from {wind_angle} to {wind_angle_limited} and wind speed from {wind_speed} to {wind_speed_limited}")
    # Use the interpolator to get the theoretical speed for the given wind conditions
    try:
        theoretical_speed = interpolator((wind_angle_limited, wind_speed_limited))
        print(f"Calculated theoretical speed: {theoretical_speed}")
        return theoretical_speed
    except Exception as e:
        print(f"Error calculating theoretical speed: {e}")
        return None
    
def calculate_vmg(boat_speed, wind_angle):
    angle_360 = wind_angle if wind_angle >= 0 else wind_angle + 360
    signed_vmg = boat_speed * math.cos(math.radians(180 - angle_360))
    return signed_vmg, abs(signed_vmg)

# Store data in the database
def store_data(conn, timestamp, speed, theoretical_speed, wind_speed, wind_angle, latitude, longitude):
    global current_session_id
    cursor = conn.cursor()
    try:
        if theoretical_speed is None:
            theoretical_speed = 0.0
        theoretical_speed = float(theoretical_speed)
        signed_vmg, abs_vmg = calculate_vmg(speed, wind_angle)
        cursor.execute('''INSERT INTO sensor_data 
                         (timestamp, session_id, speed, theoretical_speed, wind_speed, wind_angle, latitude, longitude, vmg)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                      (timestamp, current_session_id, speed, theoretical_speed, 
                       wind_speed, wind_angle, latitude, longitude, signed_vmg))
        conn.commit()
    except Exception as e:
        print(f"Error storing data: {e}")

# Data ingestion and processing thread
def data_ingestion_thread():
    global data_ingestion_running
    conn = initialize_database()
    sensor_data = mock_sensor_data()
    while data_ingestion_running:
        speed = next(sensor_data)
        wind_speed, wind_angle = simulate_wind_conditions()
        latitude, longitude = simulate_gps_coordinates()
        theoretical_speed = calculate_theoretical_speed(wind_speed, wind_angle)
        if theoretical_speed is None:
            theoretical_speed = 0.0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        store_data(conn, timestamp, speed, theoretical_speed, wind_speed, wind_angle, latitude, longitude)
    conn.close()

# Dash app for real-time plotting
def create_dash_app():
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='live-graph', style={'height': '45vh'}),
            ], style={'width': '100%'}),
            html.Div([
                dcc.Graph(id='map-view', style={'height': '45vh'}),
            ], style={'width': '100%'}),
        ]),
        dcc.Interval(
            id='interval-component',
            interval=1000,
            n_intervals=0,
            disabled=False
        ),
        dcc.Store(id='relayout-store', data=None),
        dcc.Store(id='ingestion-status-store', data=False),
        dcc.Checklist(
            id='plot-toggles',
            options=[
                {'label': 'Actual Speed', 'value': 'actual'},
                {'label': 'Theoretical Speed', 'value': 'theoretical'},
                {'label': 'Wind Speed', 'value': 'wind'},
                {'label': 'VMG', 'value': 'vmg'},
            ],
            value=['actual', 'theoretical'],  # Default selected values
            inline=True,
            style={'margin': '10px'}
        ),
        dcc.Dropdown(
            id='session-selector',
            options=[],
            placeholder='Select a session to view',
            style={'width': '300px', 'margin': '10px'}
        ),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a File to Upload Sail Polar Data')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Button('Start/Stop Data Ingestion', id='start-stop-button', n_clicks=0),
        html.Button('Switch to Net Value Plot', id='plot-mode-button', n_clicks=0),
        html.Div(id='dashboard-readout', style={'border': '2px solid black', 'padding': '10px', 'margin': '10px'})
    ])

    @app.callback(
    [Output('live-graph', 'figure'),
     Output('map-view', 'figure'),
     Output('relayout-store', 'data'),
     Output('interval-component', 'disabled'),
     Output('session-selector', 'options')],
    [Input('interval-component', 'n_intervals'),
     Input('plot-mode-button', 'n_clicks'),
     Input('start-stop-button', 'n_clicks'),
     Input('session-selector', 'value'),
     Input('plot-toggles', 'value')],
    [State('live-graph', 'relayoutData'),
     State('relayout-store', 'data'),
     State('ingestion-status-store', 'data')]
    )
    def update_graphs(n, n_clicks_plot, n_clicks_ingestion, selected_session,
                     active_plots, relayout_data, stored_relayout_data, ingestion_status):
        conn = initialize_database()
        
        session_to_display = selected_session if selected_session else current_session_id
        
        if session_to_display:
            df = pd.read_sql_query(
                """SELECT timestamp, speed, theoretical_speed, wind_speed, 
                          latitude, longitude, vmg 
                   FROM sensor_data 
                   WHERE session_id = ? 
                   ORDER BY timestamp ASC""",
                conn, params=(session_to_display,))
        else:
            df = pd.DataFrame(columns=['timestamp', 'speed', 'theoretical_speed', 
                                     'wind_speed', 'latitude', 'longitude', 'vmg'])
        
        conn.close()

        plot_mode = "net_value" if n_clicks_plot % 2 == 1 else "default"
        fig_line = go.Figure()

        if plot_mode == "default":
            if 'actual' in active_plots:
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'], 
                    y=df['speed'],
                    name='Actual Speed',
                    mode='lines+markers'
                ))

            if 'theoretical' in active_plots:
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['theoretical_speed'],
                    name='Theoretical Speed',
                    mode='lines+markers'
                ))

            if 'wind' in active_plots:
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['wind_speed'],
                    name='Wind Speed',
                    mode='lines+markers'
                ))

            if 'vmg' in active_plots:
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['vmg'],
                    name='VMG',
                    mode='lines+markers'
                ))
        else:
            net_values = df['speed'] - df['theoretical_speed']
            fig_line.add_trace(go.Scatter(
                x=df['timestamp'],
                y=net_values,
                name='Net Value (Actual - Theoretical)',
                mode='lines+markers'
            ))

        # Create the map view
        fig_map = go.Figure()
        
        if not df.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=df['latitude'],
                lon=df['longitude'],
                mode='lines+markers',
                marker=dict(size=8),
                text=df['timestamp'],
                name='Boat Track'
            ))

        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=df['latitude'].mean() if not df.empty else 60.16,
                    lon=df['longitude'].mean() if not df.empty else 24.96
                ),
                zoom=13
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True
        )

        rangeslider_visible = not data_ingestion_running

        fig_line.update_layout(
            title_text="Real-Time Sailing Performance Analysis",
            xaxis=dict(
                rangeslider=dict(visible=rangeslider_visible),
                type="date"
            ),
            yaxis=dict(title="Speed (knots)")
        )

        if relayout_data and 'xaxis.range[0]' in relayout_data:
            stored_relayout_data = relayout_data
        elif stored_relayout_data and 'xaxis.range[0]' in stored_relayout_data:
            fig_line.update_xaxes(range=[
                stored_relayout_data['xaxis.range[0]'],
                stored_relayout_data['xaxis.range[1]']
            ])

        interval_disabled = not data_ingestion_running
        session_options = get_sessions()

        return fig_line, fig_map, stored_relayout_data, interval_disabled, session_options

    @app.callback(
        Output('dashboard-readout', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_dashboard(n):
        global last_displayed_values, previous_values, current_changes
        current_time = datetime.now()
        # Query the latest sensor data row for the current session
        conn = initialize_database()
        if current_session_id:
            df = pd.read_sql_query(
                """SELECT speed, theoretical_speed, wind_speed, wind_angle, vmg, timestamp
                    FROM sensor_data 
                    WHERE session_id=? 
                    ORDER BY timestamp DESC 
                    LIMIT 1""",
                conn, params=(current_session_id,))
        else:
            df = pd.DataFrame()
        conn.close()

        if df.empty:
            return "No data available yet."

        latest = df.iloc[0]
        
        # Store current values
        current_values = {
            'vmg': latest['vmg'],
            'speed': latest['speed'],
            'net_speed': latest['speed'] - latest['theoretical_speed'],
            'wind_angle': latest['wind_angle'],
            'wind_speed': latest['wind_speed']
        }
        
        # Add current values to previous values lists
        for key, value in current_values.items():
            previous_values[key].append(value)
            # Keep only last 3 seconds of data
            if len(previous_values[key]) > 3:
                previous_values[key].pop(0)

        # Check if it's time to update changes (every 3 seconds)
        should_update_changes = (
            last_displayed_values['last_display_time'] is None or
            (current_time - last_displayed_values['last_display_time']).total_seconds() >= 3
        )

        if should_update_changes:
            # Calculate new changes
            for key in current_values.keys():
                if last_displayed_values[key] is not None:
                    # Calculate both instant and average changes
                    instant_change = current_values[key] - last_displayed_values[key]
                    if len(previous_values[key]) >= 3:
                        avg_change = current_values[key] - sum(previous_values[key][:-1]) / len(previous_values[key][:-1])
                        current_changes[key] = avg_change
                    else:
                        current_changes[key] = instant_change
                else:
                    current_changes[key] = 0

            # Update last displayed values for next change calculation
            last_displayed_values = current_values.copy()
            last_displayed_values['last_display_time'] = current_time

        def get_color_style(change):
            if abs(change) < 0.01:  # Threshold for considering no significant change
                return {'color': 'black'}
            return {'color': 'green' if change > 0 else 'red'}

        def format_change(value, change):
            change_text = f"{abs(change):+.2f}" if abs(change) >= 0.01 else "±0.00"
            return html.Div([
                html.Span(f"{value:.2f}", style={'fontSize': '1.5em'}),
                html.Br(),
                html.Span(change_text, style=get_color_style(change))
            ])

        # VMG direction
        direction = "upwind" if current_values['vmg'] < 0 else "downwind"
        
        # Create dashboard cards with changes
        dashboard = html.Div([
            html.Div([
                html.H4("Actual Speed"),
                format_change(current_values['speed'], current_changes['speed']),
                html.Span("knots")
            ], style={
                'flex': '1', 'textAlign': 'center', 'padding': '10px', 
                'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'
            }),
            html.Div([
                html.H4("Speed potential"),
                format_change(current_values['net_speed'], current_changes['net_speed']),
                html.Span("knots")
            ], style={
                'flex': '1', 'textAlign': 'center', 'padding': '10px', 
                'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'
            }),
            html.Div([
                html.H4("VMG"),
                format_change(abs(current_values['vmg']), current_changes['vmg']),
                html.Div([
                    html.Span(f"knots ({direction})")
                ])
            ], style={
                'flex': '1', 'textAlign': 'center', 'padding': '10px', 
                'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'
            }),
            html.Div([
                html.H4("Wind Angle"),
                format_change(current_values['wind_angle'], current_changes['wind_angle']),
                html.Span("degrees")
            ], style={
                'flex': '1', 'textAlign': 'center', 'padding': '10px', 
                'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'
            }),
            html.Div([
                html.H4("Wind Speed"),
                format_change(current_values['wind_speed'], current_changes['wind_speed']),
                html.Span("knots")
            ], style={
                'flex': '1', 'textAlign': 'center', 'padding': '10px', 
                'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'
            }),
        ], style={'display': 'flex', 'flexWrap': 'wrap'})

        return dashboard

    @app.callback(
        Output('upload-data', 'children'),
        Input('upload-data', 'contents'),
        prevent_initial_call=True
    )
    def upload_sail_polar_data(contents):
        global sail_polar_data
        sail_polar_data = load_sail_polar_data(contents)
        return "Sail Polar Data Uploaded"

    @app.callback(
        [Output('start-stop-button', 'children'),
        Output('ingestion-status-store', 'data'),
        Output('session-selector', 'value')],
        Input('start-stop-button', 'n_clicks'),
        State('ingestion-status-store', 'data'),
        prevent_initial_call=True
    )
    def toggle_data_ingestion_callback(n_clicks, ingestion_status):
        global data_ingestion_running
        
        if not data_ingestion_running:
            if sail_polar_data is None:
                print("Error: Please upload sail polar data first.")
                return "Start Data Ingestion", False, None
            # Starting new session
            start_new_session()
            data_ingestion_running = True
            threading.Thread(target=data_ingestion_thread, daemon=True).start()
            return "Stop Data Ingestion", True, None
        else:
            # Stopping current session
            data_ingestion_running = False
            end_current_session()
            return "Start Data Ingestion", False, None
    
    return app

# Run the Dash app
if __name__ == "__main__":
    app = create_dash_app()
    app.run_server(debug=False)