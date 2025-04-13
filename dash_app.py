import math
import threading
import time
from datetime import datetime
import sqlite3
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from scipy.interpolate import RectBivariateSpline
import base64
import io
import uuid
import json
from collections import deque
import os

# Global vars
sail_polar_data = None
data_ingestion_running = False
interpolator = None
twa_range = None
tws_range = None
plot_mode = "default"
current_session_id = None
tracking_data = deque()

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

# Some hardcoded values for Mark and Start Line to create a realistic scenario for the used tracking data
race_marks = {
    'Mark1': (-41.28390911038982, 174.80283147197125)
}

start_line = {
    'pin': (-41.28451670508809, 174.79731541291872),
    'committee': (-41.28541711877493, 174.79731947185735)
}

current_mark = 'Mark1'

# Add timing decorator for function performance measurement
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f" {func.__name__} took {execution_time:.2f}ms")
        return result
    return wrapper

def initialize_database():
    conn = sqlite3.connect('sailing_data.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time DATETIME,
                        end_time DATETIME NULL)''')
    
    # Create sensor_data table with wind_direction
    cursor.execute('''CREATE TABLE IF NOT EXISTS sensor_data (
                        timestamp DATETIME,
                        session_id TEXT,
                        speed REAL,
                        theoretical_speed REAL,
                        wind_speed REAL,
                        wind_angle REAL,
                        wind_direction REAL,
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
    
    return [{'label': f"Race {idx + 1}: {start_time}", 
             'value': session_id} 
            for idx, (session_id, start_time, end_time) in enumerate(sessions)]

def compute_destination(lat, lon, bearing_degrees, distance_nm):
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_degrees)
    
    # Earth radius in km; 1 NM = 1.852 km
    R_km = 6371.0
    distance_km = distance_nm * 1.852
    
    # Great-circle formula
    lat2 = math.asin(
        math.sin(lat_rad) * math.cos(distance_km / R_km) +
        math.cos(lat_rad) * math.sin(distance_km / R_km) * math.cos(bearing_rad)
    )
    lon2 = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_km / R_km) * math.cos(lat_rad),
        math.cos(distance_km / R_km) - math.sin(lat_rad) * math.sin(lat2)
    )
    
    return (math.degrees(lat2), math.degrees(lon2))

def load_tracking_data(file_path):
    global tracking_data
    with open(file_path, 'r') as f:
        data = json.load(f)
        tracking_data = deque(data['dframes'])

def real_sensor_data():
    global tracking_data
    while tracking_data:
        frame = tracking_data.popleft()
        data = {
            'latitude': frame[0],  
            'longitude': frame[1],
            'speed': frame[3],
            'wind_speed': frame[7],
            'wind_angle': frame[8],
            'wind_direction': frame[11]
        }
        yield data
        tracking_data.append(frame)
        time.sleep(1)

def load_sail_polar_data(contents):
    global interpolator, twa_range, tws_range
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';', index_col=0)
    twa = data.index.values
    tws = data.columns.astype(float).values
    boat_speeds = data.values
    interpolator = RectBivariateSpline(twa, tws, boat_speeds)
    twa_range = (min(twa), max(twa))
    tws_range = (min(tws), max(tws))
    return data

def get_sail_polar_files():
    directory = 'data/sail_polars'
    files = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.endswith('.csv'):
                files.append(f)
    return files

def load_sail_polar_data_from_file(filename):
    global interpolator, twa_range, tws_range
    filepath = os.path.join('data/sail_polars', filename)
    data = pd.read_csv(filepath, sep=';', index_col=0)
    twa = data.index.values
    tws = data.columns.astype(float).values
    boat_speeds = data.values
    interpolator = RectBivariateSpline(twa, tws, boat_speeds)
    twa_range = (min(twa), max(twa))
    tws_range = (min(tws), max(tws))
    return data

def limit(value, min_value, max_value):
    return max(min_value, min(value, max_value))

@timing_decorator
def calculate_theoretical_speed(wind_speed, wind_angle, actual_speed=None):
    wind_angle = abs(wind_angle)
    global interpolator, twa_range, tws_range
    if interpolator is None:
        print("Error: Interpolator is not initialized.")
        return None
        
    wind_angle_limited = limit(wind_angle, twa_range[0], twa_range[1])
    wind_speed_limited = limit(wind_speed, tws_range[0], tws_range[1])
    
    if wind_angle != wind_angle_limited or wind_speed != wind_speed_limited:
        print(f"Warning: limited wind angle from {wind_angle} to {wind_angle_limited} and wind speed from {wind_speed} to {wind_speed_limited}")
    
    try:
        interpolation_start = time.perf_counter()
        theoretical_speed = float(interpolator(wind_angle_limited, wind_speed_limited)[0, 0])
        interpolation_time = (time.perf_counter() - interpolation_start) * 1000
        print(f" Interpolation calculation took {interpolation_time:.2f}ms")
        
        if actual_speed is not None and theoretical_speed < actual_speed:
            theoretical_speed = actual_speed
        return theoretical_speed
    except Exception as e:
        print(f"Error calculating theoretical speed: {e}")
        return None

def calculate_vmg(boat_speed, wind_angle):
    angle_360 = wind_angle if wind_angle >= 0 else wind_angle + 360
    signed_vmg = boat_speed * math.cos(math.radians(180 - angle_360))
    return signed_vmg, abs(signed_vmg)

@timing_decorator
def store_data(conn, timestamp, speed, theoretical_speed, wind_speed, wind_angle, latitude, longitude, wind_direction):
    global current_session_id
    cursor = conn.cursor()
    try:
        if theoretical_speed is None:
            theoretical_speed = 0.0
        theoretical_speed = float(theoretical_speed)
        
        vmg_start = time.perf_counter()
        signed_vmg, abs_vmg = calculate_vmg(speed, wind_angle)
        vmg_time = (time.perf_counter() - vmg_start) * 1000
        print(f" VMG calculation took {vmg_time:.2f}ms")
        
        db_start = time.perf_counter()
        cursor.execute('''INSERT INTO sensor_data 
                         (timestamp, session_id, speed, theoretical_speed, wind_speed, wind_angle, 
                          latitude, longitude, vmg, wind_direction)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                      (timestamp, current_session_id, speed, theoretical_speed, 
                       wind_speed, wind_angle, latitude, longitude, signed_vmg, wind_direction))
        conn.commit()
        db_time = (time.perf_counter() - db_start) * 1000
        print(f"Database insert took {db_time:.2f}ms")
        
    except Exception as e:
        print(f"Error storing data: {e}")

@timing_decorator
def data_ingestion_thread():
    global data_ingestion_running
    conn = initialize_database()
    sensor_data = real_sensor_data()
    total_samples = 0
    total_processing_time = 0
    
    while data_ingestion_running:
        cycle_start = time.perf_counter()
        
        data = next(sensor_data)
        sensor_time = (time.perf_counter() - cycle_start) * 1000
        print(f"Sensor data fetch took {sensor_time:.2f}ms")
        
        theo_start = time.perf_counter()
        theoretical_speed = calculate_theoretical_speed(
            data['wind_speed'], 
            data['wind_angle'],
            actual_speed=data['speed']
        )
        theo_time = (time.perf_counter() - theo_start) * 1000
        print(f"Theoretical speed calculation took {theo_time:.2f}ms")
        
        if theoretical_speed is None:
            theoretical_speed = 0.0
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        store_start = time.perf_counter()
        store_data(conn, timestamp, data['speed'], theoretical_speed, 
                  data['wind_speed'], data['wind_angle'], 
                  data['latitude'], data['longitude'],
                  data['wind_direction'])
        store_time = (time.perf_counter() - store_start) * 1000
        print(f"Data storage took {store_time:.2f}ms")
        
        cycle_time = (time.perf_counter() - cycle_start) * 1000
        total_processing_time += cycle_time
        total_samples += 1
        avg_processing_time = total_processing_time / total_samples
        
        # Run Latency tests
        print(f"Complete cycle took {cycle_time:.2f}ms")
        print(f"Average processing time: {avg_processing_time:.2f}ms over {total_samples} samples")
        print("----------------------------------------")
    
    conn.close()

def calculate_wind_shift(angles):
    if len(angles) < 2:
        return 0
    shifts = []
    for i in range(1, len(angles)):
        shift = angles[i] - angles[i-1]
        if shift > 180:
            shift -= 360
        elif shift < -180:
            shift += 360
        shifts.append(shift)
    return sum(shifts) / len(shifts)

def analyze_wind_trend(speeds, directions, timespan):
    if not speeds or not directions:
        return {
            'speed_trend': 0,
            'direction_trend': 0,
            'avg_speed': 0,
            'avg_direction': 0
        }
    
    avg_speed = sum(speeds) / len(speeds)
    speed_trend = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
    
    direction_changes = []
    for i in range(1, len(directions)):
        change = directions[i] - directions[i-1]
        if change > 180:
            change -= 360
        elif change < -180:
            change += 360
        direction_changes.append(change)
    
    direction_trend = sum(direction_changes) / len(direction_changes) if direction_changes else 0
    avg_direction = directions[-1]
    
    return {
        'speed_trend': speed_trend,
        'direction_trend': direction_trend,
        'avg_speed': avg_speed,
        'avg_direction': avg_direction
    }

def create_mark_input():
    return html.Div([
        html.H4("Race Course Setup", style={'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.H5("Start Line", style={'marginBottom': '10px'}),
                html.Div([
                    html.Div(f"Pin End: ({start_line['pin'][0]:.6f}, {start_line['pin'][1]:.6f})"),
                    html.Div(f"Committee: ({start_line['committee'][0]:.6f}, {start_line['committee'][1]:.6f})")
                ])
            ], style={'flex': '1', 'marginRight': '20px'}),
            
            html.Div([
                html.H5("Course Marks", style={'marginBottom': '10px'}),
                html.Div([
                    html.Div(f"Mark1: ({race_marks['Mark1'][0]:.6f}, {race_marks['Mark1'][1]:.6f})")
                ], id='marks-list', style={'marginTop': '10px'})
            ], style={'flex': '1'}),
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        html.Div([
            html.H5("Current Mark"),
            dcc.Dropdown(
                id='current-mark-selector',
                options=[{'label': name, 'value': name} for name in race_marks.keys()],
                value='Mark1',
                placeholder='Select current mark'
            )
        ], style={'width': '200px'})
    ], style={
        'border': '2px solid black',
        'padding': '15px',
        'margin': '10px',
        'backgroundColor': '#f8f9fa'
    })

def get_latest_wind_direction():
    conn = initialize_database()
    if current_session_id:
        df = pd.read_sql_query(
            """SELECT wind_angle FROM sensor_data 
               WHERE session_id = ? 
               ORDER BY timestamp DESC LIMIT 1""",
            conn, params=(current_session_id,))
        conn.close()
        if not df.empty:
            return df['wind_angle'].iloc[0]
    return None

def get_latest_position():
    conn = initialize_database()
    if current_session_id:
        df = pd.read_sql_query(
            """SELECT latitude, longitude FROM sensor_data 
               WHERE session_id = ? 
               ORDER BY timestamp DESC LIMIT 1""",
            conn, params=(current_session_id,))
        conn.close()
        if not df.empty:
            return {'lat': df['latitude'].iloc[0], 'lon': df['longitude'].iloc[0]}
    return None

def calculate_distance_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(y, x))
    bearing = (bearing + 360) % 360
    
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    nm = 3440.065 * c  # Earth radius in NM
    
    return nm, bearing

def analyze_start_line():
    if not start_line['pin'] or not start_line['committee']:
        return None
    
    pin_lat, pin_lon = start_line['pin']
    com_lat, com_lon = start_line['committee']
    
    line_length, line_bearing = calculate_distance_bearing(pin_lat, pin_lon, com_lat, com_lon)
    line_length_meters = line_length * 1852
    
    wind_direction = get_latest_wind_direction()
    if wind_direction is None:
        return None
        
    line_bias = (wind_direction - line_bearing + 360) % 360
    if line_bias > 180:
        line_bias = line_bias - 360
    
    return {
        'line_length': line_length_meters,
        'line_bearing': line_bearing,
        'bias_angle': line_bias,
        'favored_end': 'Pin' if line_bias > 0 else 'Committee' if line_bias < 0 else 'Even',
        'bias_advantage': abs(line_bias)
    }

def analyze_mark_rounding():
    global current_mark, race_marks
    
    if not current_mark or current_mark not in race_marks:
        return None
    
    boat_position = get_latest_position()
    if not boat_position:
        return None
    
    mark_lat, mark_lon = race_marks[current_mark]
    distance, bearing = calculate_distance_bearing(
        boat_position['lat'], boat_position['lon'],
        mark_lat, mark_lon
    )
    
    wind_direction = get_latest_wind_direction()
    if wind_direction is None:
        return None
    
    layline_port = (wind_direction - 42) % 360
    layline_stbd = (wind_direction + 42) % 360
    
    return {
        'distance_to_mark': distance,
        'bearing_to_mark': bearing,
        'on_layline': min(abs(bearing - layline_port), abs(bearing - layline_stbd)) < 5,
        'suggested_tack': 'Port' if abs(bearing - layline_port) < abs(bearing - layline_stbd) else 'Starboard',
        'layline_port': layline_port,
        'layline_stbd': layline_stbd
    }

def analyze_performance():
    conn = initialize_database()
    if not current_session_id:
        return None
        
    df = pd.read_sql_query(
        """SELECT speed, theoretical_speed, wind_speed, wind_angle, vmg
           FROM sensor_data 
           WHERE session_id = ? 
           ORDER BY timestamp DESC LIMIT 1""",
        conn, params=(current_session_id,))
    conn.close()

    if df.empty:
        return None
        
    latest = df.iloc[0]
    speed_efficiency = (latest['speed'] / latest['theoretical_speed'] * 100) if latest['theoretical_speed'] > 0 else 0
    return {
        'speed_efficiency': speed_efficiency,
    }

def calculate_time_to_mark():
    global current_mark, race_marks
    
    if not current_mark or current_mark not in race_marks:
        return None
    
    conn = initialize_database()
    if current_session_id:
        df = pd.read_sql_query(
            """SELECT speed, latitude, longitude, wind_direction, wind_angle
               FROM sensor_data 
               WHERE session_id = ? 
               ORDER BY timestamp DESC LIMIT 1""",
            conn, params=(current_session_id,))
        conn.close()
        
        if df.empty:
            return None
            
        current_speed = df['speed'].iloc[0]
        current_pos = {'lat': df['latitude'].iloc[0], 'lon': df['longitude'].iloc[0]}
        wind_direction = df['wind_direction'].iloc[0]
        
        if current_speed <= 0:
            return None
            
        mark_lat, mark_lon = race_marks[current_mark]
        distance, bearing_to_mark = calculate_distance_bearing(
            current_pos['lat'], current_pos['lon'],
            mark_lat, mark_lon
        )
        
        wind_angle_to_mark = abs((wind_direction - bearing_to_mark + 360) % 360)
        
        if wind_angle_to_mark < 45:
            tacking_angle = 42
            tacking_distance = distance / math.cos(math.radians(tacking_angle))
            upwind_speed = current_speed * 0.85
            time_to_mark = (tacking_distance * 60) / upwind_speed
            return time_to_mark
        else:
            time_to_mark = (distance * 60) / current_speed
            return time_to_mark
            
    return None

def calculate_changes(values):
    if len(values) < 2:
        return []
    changes = []
    for i in range(1, len(values)):
        if isinstance(values[i], (int, float)) and isinstance(values[i-1], (int, float)):
            changes.append(values[i] - values[i-1])
    return changes

def calculate_sma(data, window=3):
    if not data or len(data) == 0:
        return []
    sma = []
    for i in range(len(data)):
        if i < window - 1:
            sma.append(sum(data[:i+1]) / (i+1))
        else:
            sma.append(sum(data[i-window+1:i+1]) / window)
    return sma

def calculate_ema(data, alpha=0.3):
    if not data:
        return []
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

def format_direction(degrees):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(degrees / 45) % 8
    return directions[index]

def create_trend_graph(data, metric_name, color='blue', long_term_alpha=0.1, height=50):
    short_term = calculate_ema(data, alpha=0.3)
    traces = [{
        'x': list(range(len(short_term))),
        'y': short_term,
        'mode': 'lines',
        'line': {'color': color},
        'name': 'Short-term'
    }]
    long_term = calculate_ema(data, alpha=long_term_alpha)
    traces.append({
        'x': list(range(len(long_term))),
        'y': long_term,
        'mode': 'lines',
        'line': {'color': 'gray', 'dash': 'dash'},
        'name': 'Long-term'
    })
    
    return dcc.Graph(
        figure={
            'data': traces,
            'layout': {
                'height': height,
                'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                'showlegend': False
            }
        },
        config={'displayModeBar': False},
        style={'width': '100px', 'height': '50px', 'display': 'inline-block'}
    )

def format_metric_with_trend(value, change, trend_data, metric_name, long_term_alpha=0.1):
    return html.Div([
        html.Div([
            html.Span(f"{value:.2f}", style={'fontSize': '1.5em'}),
            html.Br(),
            html.Span(f"{change:+.2f}" if abs(change) >= 0.01 else "±0.00",
                     style={'color': 'green' if change > 0 else 'red' if change < 0 else 'black'})
        ], style={'display': 'inline-block', 'marginRight': '10px'}),
        create_trend_graph(trend_data, metric_name, long_term_alpha=long_term_alpha)
    ], style={'display': 'flex', 'alignItems': 'center'})

def create_dash_app():
    app = dash.Dash(__name__)
    
    # Load tracking data at startup
    load_tracking_data('data/Tracking_example_data_1.json')
    
    app.layout = html.Div([
        # Graphs
        html.Div([
            html.Div([
                dcc.Graph(id='map-view', style={'height': '45vh'}),
            ], id='map-view-container', style={'display': 'none', 'width': '50%'}),
            html.Div([
                dcc.Graph(id='live-graph', style={'height': '45vh'}),
            ], id='live-graph-container', style={'display': 'none', 'width': '50%'}),
        ],style={'display': 'flex'}),
        # Intervals: primary (3 sec) and fast (1 sec)
        dcc.Interval(
            id='interval-component',
            interval=3000,
            n_intervals=0,
            disabled=False
        ),
        dcc.Interval(
            id='fast-interval-component',
            interval=1000,
            n_intervals=0,
            disabled=False
        ),
        dcc.Store(id='relayout-store', data=None),
        dcc.Store(id='ingestion-status-store', data=False),
        dcc.Checklist(
            id='plot-toggles',
            options=[
                {'label': 'Speed', 'value': 'actual'},
                {'label': 'Theoretical Speed', 'value': 'theoretical'},
                {'label': 'Wind Speed', 'value': 'wind'},
                {'label': 'VMG', 'value': 'vmg'},
                {'label': 'Wind Direction', 'value': 'wind_direction'},
            ],
            value=['actual', 'theoretical'],
            inline=True,
            style={'margin': '10px'}
        ),
        dcc.Dropdown(
            id='session-selector',
            options=[],
            placeholder='Select a session to view',
            style={'width': '300px', 'margin': '10px'}
        ),
        # Sail Polar Data Dropdown
        dcc.Dropdown(
            id='sail-polar-dropdown',
            options=[{'label': f, 'value': f} for f in get_sail_polar_files()],
            placeholder='Select Sail Polar Data',
            style={'width': '300px', 'margin': '10px'}
        ),
        html.Div(id='sail-polar-status'),
        html.Button('Start/Stop Data Ingestion', id='start-stop-button', n_clicks=0),
        html.Button('Switch to Net Value Plot', id='plot-mode-button', n_clicks=0),
        html.Div([
        html.Label("Long-term Trend Range (EMA alpha):"),
        dcc.Slider(
            id='long-term-range-slider',
            min=0.05,
            max=0.5,
            step=0.05,
            value=0.1,  # default value
            marks={i/100: f"{i/100:.2f}" for i in range(5, 51, 5)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '300px', 'margin': '10px'}),
        html.Div(id='dashboard-readout', style={'border': '2px solid black', 'padding': '10px', 'margin': '10px'}),
        html.Div([
            html.Div([
                html.Div(id='race-analysis-display')
            ], style={
                'border': '2px solid black',
                'padding': '15px',
                'margin': '10px',
                'backgroundColor': '#f8f9fa'
            }),
            create_mark_input(),
        ])
    ])
    @app.callback(
    [Output('map-view-container', 'style'),
     Output('live-graph-container', 'style')],
    [Input('ingestion-status-store', 'data')]
    )
    def update_graph_visibility(ingestion_status):
        # If ingestion is running, show the graphs; otherwise, hide them.
        if ingestion_status:
            return {'display': 'block'}, {'display': 'block'}
        else:
            return {'display': 'none'}, {'display': 'none'}

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
                          wind_direction, latitude, longitude, vmg 
                   FROM sensor_data 
                   WHERE session_id = ? 
                   ORDER BY timestamp ASC""",
                conn, params=(session_to_display,))
        else:
            df = pd.DataFrame(columns=['timestamp', 'speed', 'theoretical_speed', 
                                       'wind_speed', 'wind_direction', 'latitude', 'longitude', 'vmg'])
        
        conn.close()

        plot_mode_local = "net_value" if n_clicks_plot % 2 == 1 else "default"
        fig_line = go.Figure()

        if plot_mode_local == "default":
            # Add actual speed and its trend
            if 'actual' in active_plots:
                # Calculate EMA for actual speed
                speed_ema = calculate_ema(df['speed'].tolist(), alpha=0.1)
                
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'], 
                    y=df['speed'],
                    name='SOG',
                    mode='lines',
                    line={'color': 'blue'}
                ))
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=speed_ema,
                    name='SOG Trend',
                    mode='lines',
                    line={'color': 'blue', 'dash': 'dot'},
                    opacity=0.7
                ))

            # Add theoretical speed and its trend
            if 'theoretical' in active_plots:
                # Calculate EMA for theoretical speed
                theo_ema = calculate_ema(df['theoretical_speed'].tolist(), alpha=0.1)
                
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['theoretical_speed'],
                    name='Theoretical Speed',
                    mode='lines',
                    line={'color': 'red'}
                ))
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=theo_ema,
                    name='Theoretical Speed Trend',
                    mode='lines',
                    line={'color': 'red', 'dash': 'dot'},
                    opacity=0.7
                ))

            # Add wind speed and its trend
            if 'wind' in active_plots:
                # Calculate EMA for wind speed
                wind_ema = calculate_ema(df['wind_speed'].tolist(), alpha=0.1)
                
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['wind_speed'],
                    name='Wind Speed',
                    mode='lines',
                    line={'color': 'green'}
                ))
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=wind_ema,
                    name='Wind Speed Trend',
                    mode='lines',
                    line={'color': 'green', 'dash': 'dot'},
                    opacity=0.7
                ))

            # Add VMG and its trend
            if 'vmg' in active_plots:
                # Calculate EMA for VMG
                vmg_ema = calculate_ema(df['vmg'].tolist(), alpha=0.1)
                
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['vmg'],
                    name='VMG',
                    mode='lines',
                    line={'color': 'purple'}
                ))
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=vmg_ema,
                    name='VMG Trend',
                    mode='lines',
                    line={'color': 'purple', 'dash': 'dot'},
                    opacity=0.7
                ))

            # Add wind direction and its trend
            if 'wind_direction' in active_plots:
                # Calculate EMA for wind direction
                wind_dir_ema = calculate_ema(df['wind_direction'].tolist(), alpha=0.1)
                
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['wind_direction'],
                    name='Wind Direction',
                    mode='lines',
                    line={'color': 'orange'},
                    yaxis='y2'
                ))
                fig_line.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=wind_dir_ema,
                    name='Wind Direction Trend',
                    mode='lines',
                    line={'color': 'orange', 'dash': 'dot'},
                    opacity=0.7,
                    yaxis='y2'
                ))

            fig_line.update_layout(
                yaxis2=dict(
                    title='Wind Direction',
                    overlaying='y',
                    side='right',
                    range=[0, 360]
                ),
                yaxis=dict(title='SOG'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

        else:
            # Net value plot mode
            net_values = df['speed'] - df['theoretical_speed']
            net_ema = calculate_ema(net_values.tolist(), alpha=0.1)
            
            fig_line.add_trace(go.Scatter(
                x=df['timestamp'],
                y=net_values,
                                name='Speed Potential (SOG - Theoretical)', 
                mode='lines',
                line={'color': 'blue'}
            ))
            fig_line.add_trace(go.Scatter(
                x=df['timestamp'],
                y=net_ema,
                name='Speed Potential Trend',
                mode='lines',
                line={'color': 'blue', 'dash': 'dot'},
                opacity=0.7
            ))

        start_analysis = analyze_start_line() or {}
        mark_analysis = analyze_mark_rounding() or {}
        
        # Determine favored end color
        favored_end = start_analysis.get('favored_end', 'Even')
        start_line_color = 'gray'
        if favored_end == 'Pin':
            start_line_color = 'red'
        elif favored_end == 'Committee':
            start_line_color = 'blue'

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
            
            # Color-coded start line
            if start_line['pin'] and start_line['committee']:
                fig_map.add_trace(go.Scattermapbox(
                    lat=[start_line['pin'][0], start_line['committee'][0]],
                    lon=[start_line['pin'][1], start_line['committee'][1]],
                    mode='lines+markers',
                    marker=dict(size=10, color=start_line_color),
                    line=dict(width=3, color=start_line_color),
                    name='Start Line'
                ))
            
            # Plot each mark
            for mark_name, (m_lat, m_lon) in race_marks.items():
                fig_map.add_trace(go.Scattermapbox(
                    lat=[m_lat],
                    lon=[m_lon],
                    mode='markers+text',
                    marker=dict(size=12, color='red'),
                    text=[mark_name],
                    textposition="top center",
                    name=f'Mark: {mark_name}'
                ))
            
            # The boat's latest position
            boat_lat = df['latitude'].iloc[-1]
            boat_lon = df['longitude'].iloc[-1]
            
            # Draw laylines if available
            layline_port = mark_analysis.get('layline_port', None)
            layline_stbd = mark_analysis.get('layline_stbd', None)
            if layline_port is not None and layline_stbd is not None:
                port_dest = compute_destination(boat_lat, boat_lon, layline_port, 1.0)
                stbd_dest = compute_destination(boat_lat, boat_lon, layline_stbd, 1.0)
                
                fig_map.add_trace(go.Scattermapbox(
                    lat=[boat_lat, port_dest[0]],
                    lon=[boat_lon, port_dest[1]],
                    mode='lines',
                    line=dict(width=2, color='rgba(255, 165, 0, 0.6)'),
                    name='Port Layline'
                ))
                fig_map.add_trace(go.Scattermapbox(
                    lat=[boat_lat, stbd_dest[0]],
                    lon=[boat_lon, stbd_dest[1]],
                    mode='lines',
                    line=dict(width=2, color='rgba(255, 165, 0, 0.6)'),
                    name='Starboard Layline'
                ))
            
            # Center the map on the boat's latest position
            fig_map.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=boat_lat, lon=boat_lon),
                    zoom=14
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
        else:
            # If no data, show a default map
            fig_map.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=0, lon=0),
                    zoom=2
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )

        return fig_line, fig_map, stored_relayout_data, not data_ingestion_running, get_sessions()

    @app.callback(
        Output('dashboard-readout', 'children'),
        [Input('fast-interval-component', 'n_intervals'),
        Input('long-term-range-slider', 'value'),
        Input('session-selector', 'value')]
    )
    def update_dashboard(n_1sec, long_term_alpha, selected_session):
        callback_start = time.perf_counter()
        
        session_to_display = selected_session if selected_session else current_session_id
        
        if not session_to_display:
            callback_time = (time.perf_counter() - callback_start) * 1000
            print(f"Dashboard update callback (no data) took {callback_time:.2f}ms")
            return "No data available yet."
        
        try:
            db_start = time.perf_counter()
            conn = initialize_database()
            latest_df = pd.read_sql_query(
                """SELECT speed, theoretical_speed, wind_speed, wind_direction, 
                          vmg, timestamp
                   FROM sensor_data 
                   WHERE session_id = ? 
                   ORDER BY timestamp DESC LIMIT 1""",
                conn, params=(session_to_display,))
            
            hist_df = pd.read_sql_query(
                """SELECT speed, theoretical_speed, wind_speed, wind_direction, 
                          vmg, timestamp
                   FROM sensor_data 
                   WHERE session_id = ? 
                   ORDER BY timestamp DESC LIMIT 30""",
                conn, params=(session_to_display,))
            conn.close()
            db_time = (time.perf_counter() - db_start) * 1000
            print(f" Dashboard database queries took {db_time:.2f}ms")
            
            if latest_df.empty or hist_df.empty:
                callback_time = (time.perf_counter() - callback_start) * 1000
                print(f" Dashboard update callback (empty data) took {callback_time:.2f}ms")
                return "No data available yet."
            
            processing_start = time.perf_counter()
            hist_df = hist_df.iloc[::-1]  # Chronological order
            
            # Calculate EMAs and changes
            ma_calculations_start = time.perf_counter()
            raw_speed = hist_df['speed'].tolist()
            ma_speed = calculate_ema(raw_speed, alpha=0.3)
            change_speed = ma_speed[-1] - (ma_speed[-2] if len(ma_speed) > 1 else 0)
            
            raw_net_speed = (hist_df['speed'] - hist_df['theoretical_speed']).tolist()
            ma_net_speed = calculate_ema(raw_net_speed, alpha=0.3)
            change_net_speed = ma_net_speed[-1] - (ma_net_speed[-2] if len(ma_net_speed) > 1 else 0)
            
            raw_vmg = hist_df['vmg'].abs().tolist()
            ma_vmg = calculate_ema(raw_vmg, alpha=0.3)
            change_vmg = ma_vmg[-1] - (ma_vmg[-2] if len(ma_vmg) > 1 else 0)
            
            raw_wind_direction = hist_df['wind_direction'].tolist()
            ma_wind_direction = calculate_ema(raw_wind_direction, alpha=0.3)
            change_wind_direction = ma_wind_direction[-1] - (ma_wind_direction[-2] if len(ma_wind_direction) > 1 else 0)
            
            raw_wind_speed = hist_df['wind_speed'].tolist()
            ma_wind_speed = calculate_ema(raw_wind_speed, alpha=0.3)
            change_wind_speed = ma_wind_speed[-1] - (ma_wind_speed[-2] if len(ma_wind_speed) > 1 else 0)
            
            ma_trends = {
                'speed': ma_speed,
                'net_speed': ma_net_speed,
                'vmg': ma_vmg,
                'wind_direction': ma_wind_direction,
                'wind_speed': ma_wind_speed
            }
            
            ma_changes = {
                'speed': change_speed,
                'net_speed': change_net_speed,
                'vmg': change_vmg,
                'wind_direction': change_wind_direction,
                'wind_speed': change_wind_speed
            }
            
            latest = latest_df.iloc[0]
            current_values = {
                'speed': float(latest['speed'] or 0),
                'net_speed': float((latest['speed'] or 0) - (latest['theoretical_speed'] or 0)),
                'vmg': abs(float(latest['vmg'] or 0)),
                'wind_direction': float(latest['wind_direction'] or 0),
                'wind_speed': float(latest['wind_speed'] or 0)
            }
            
            ma_time = (time.perf_counter() - ma_calculations_start) * 1000
            print(f" Moving average calculations took {ma_time:.2f}ms")
            
            # Generate dashboard HTML
            render_start = time.perf_counter()
            dashboard = html.Div([
                html.Div([
                    html.H4("SOG"),
                    format_metric_with_trend(
                        current_values['speed'],
                        ma_changes['speed'],
                        ma_trends['speed'],
                        'speed',
                        long_term_alpha=long_term_alpha
                    ),
                    html.Span("knots")
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px',
                          'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                html.Div([
                    html.H4("Speed potential"),
                    format_metric_with_trend(
                        current_values['net_speed'],
                        ma_changes['net_speed'],
                        ma_trends['net_speed'],
                        'net_speed',
                        long_term_alpha=long_term_alpha
                    ),
                    html.Span("knots")
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px',
                          'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                html.Div([
                    html.H4("VMG"),
                    format_metric_with_trend(
                        current_values['vmg'],
                        ma_changes['vmg'],
                        ma_trends['vmg'],
                        'vmg',
                        long_term_alpha=long_term_alpha
                    ),
                    html.Span("knots")
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px',
                          'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                html.Div([
                    html.H4("Wind Direction"),
                    format_metric_with_trend(
                        current_values['wind_direction'],
                        ma_changes['wind_direction'],
                        ma_trends['wind_direction'],
                        'wind_direction',
                        long_term_alpha=long_term_alpha
                    ),
                    html.Div([
                        html.Span(f"({format_direction(current_values['wind_direction'])})", 
                                  style={'fontSize': '0.9em', 'color': 'gray'})
                    ])
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px',
                          'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                html.Div([
                    html.H4("Wind Speed"),
                    format_metric_with_trend(
                        current_values['wind_speed'],
                        ma_changes['wind_speed'],
                        ma_trends['wind_speed'],
                        'wind_speed',
                        long_term_alpha=long_term_alpha
                    ),
                    html.Span("knots")
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px',
                          'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
            
            render_time = (time.perf_counter() - render_start) * 1000
            print(f" Dashboard HTML rendering took {render_time:.2f}ms")
            
            callback_time = (time.perf_counter() - callback_start) * 1000
            print(f" Complete dashboard update took {callback_time:.2f}ms")
            print("----------------------------------------")
            
            return dashboard
        
        except Exception as e:
            callback_time = (time.perf_counter() - callback_start) * 1000
            print(f" Dashboard update failed after {callback_time:.2f}ms: {e}")
            return "Error updating dashboard"

    @app.callback(
        Output('sail-polar-status', 'children'),
        Input('sail-polar-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_sail_polar(selected_file):
        global sail_polar_data
        if selected_file:
            sail_polar_data = load_sail_polar_data_from_file(selected_file)
            return f"Sail Polar Data '{selected_file[:-4]}' Loaded"
        return "Select a Sail Polar File"

    @app.callback(
        [Output('start-stop-button', 'children'),
        Output('ingestion-status-store', 'data'),
        Output('session-selector', 'value')],
        Input('start-stop-button', 'n_clicks'),
        State('ingestion-status-store', 'data'),
        prevent_initial_call=True
    )
    def toggle_data_ingestion_callback(n_clicks, ingestion_status):
        global data_ingestion_running, current_session_id
        if not data_ingestion_running:
            if sail_polar_data is None:
                print("Error: Please select sail polar data first.")
                return "Start Data Ingestion", False, None
            current_session_id = start_new_session()
            print("Session started, current_session_id =", current_session_id)
            data_ingestion_running = True
            threading.Thread(target=data_ingestion_thread, daemon=True).start()
            return "Stop Data Ingestion", True, None
        else:
            data_ingestion_running = False
            end_current_session()
            print("Session ended. current_session_id reset.")
            return "Start Data Ingestion", False, None
    
    @app.callback(
        Output('wind-trend-display', 'children'),
        [Input('interval-component', 'n_intervals'),
         Input('wind-trend-timespan', 'value')]
    )
    def update_wind_trend(n, timespan):
        if not current_session_id:
            return "No data available"
            
        conn = initialize_database()
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(seconds=timespan)
        
        df = pd.read_sql_query(
            """SELECT wind_speed, wind_direction, timestamp
               FROM sensor_data 
               WHERE session_id = ? 
               AND timestamp BETWEEN ? AND ?
               ORDER BY timestamp ASC""",
            conn,
            params=(current_session_id, 
                    start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    end_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.close()
        
        if df.empty:
            return "Insufficient data for analysis"
        
        trend = analyze_wind_trend(
            df['wind_speed'].tolist(),
            df['wind_direction'].tolist(),
            timespan
        )
        
        def get_trend_arrow(value):
            if abs(value) < 0.5:
                return "→"
            return "↗" if value > 0 else "↘"
        
        def get_trend_text(value):
            if abs(value) < 0.5:
                return "steady"
            return "increasing" if value > 0 else "decreasing"
        
        def get_shift_text(value):
            if abs(value) < 0.5:
                return "steady"
            return "shifting right" if value > 0 else "shifting left"
        
        def format_direction(degrees):
            directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            index = round(degrees / 45) % 8
            return directions[index]
        
        return html.Div([
            html.Div([
                html.H5("Wind Speed", style={'marginBottom': '5px'}),
                html.Div([
                    html.Span(f"{trend['avg_speed']:.1f} knots ", style={'fontSize': '1.2em'}),
                    html.Span(get_trend_arrow(trend['speed_trend']), 
                              style={'fontSize': '1.5em', 
                                     'color': 'green' if trend['speed_trend'] > 0 else 'red'})
                ]),
                html.Div(get_trend_text(trend['speed_trend']),
                         style={'fontSize': '0.9em', 'color': 'gray'})
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.H5("Wind Direction"),
                html.Div([
                    html.Span(
                        f"{trend['avg_direction']:.1f}° ({format_direction(trend['avg_direction'])}) ", 
                        style={'fontSize': '1.2em'}
                    ),
                    html.Span(get_trend_arrow(trend['direction_trend']), 
                              style={'fontSize': '1.5em',
                                     'color': 'orange'})
                ]),
                html.Div(get_shift_text(trend['direction_trend']),
                         style={'fontSize': '0.9em', 'color': 'gray'})
            ])
        ])

    @app.callback(
        [Output('marks-list', 'children'),
         Output('current-mark-selector', 'options')],
        [Input('add-mark', 'n_clicks')],
        [State('mark-name', 'value'),
         State('mark-lat', 'value'),
         State('mark-lon', 'value')]
    )
    def add_race_mark(n_clicks, name, lat, lon):
        if n_clicks is None or not all([name, lat, lon]):
            marks_list = [
                html.Div(f"{name}: ({lat:.6f}, {lon:.6f})")
                for name, (lat, lon) in race_marks.items()
            ]
            options = [{'label': name, 'value': name} for name in race_marks.keys()]
            return marks_list, options
        
        race_marks[name] = (lat, lon)
        marks_list = [
            html.Div(f"{name}: ({lat:.6f}, {lon:.6f})")
            for name, (lat, lon) in race_marks.items()
        ]
        options = [{'label': name, 'value': name} for name in race_marks.keys()]
        
        return marks_list, options

    @app.callback(
        Output('race-analysis-display', 'children'),
        [Input('interval-component', 'n_intervals'),
        Input('current-mark-selector', 'value')]
    )
    def update_race_analysis(n, selected_mark):
        global current_mark
        current_mark = selected_mark

        # Retrieve analysis data
        start_analysis = analyze_start_line() or {}
        mark_analysis = analyze_mark_rounding() or {}
        performance = analyze_performance() or {}
        time_to_mark = calculate_time_to_mark()

        # Extract Start Line values
        line_length = start_analysis.get('line_length', 0.0)
        line_bearing = start_analysis.get('line_bearing', 0.0)
        favored_end = start_analysis.get('favored_end', 'n/a')
        bias_advantage = start_analysis.get('bias_advantage', 0.0)

        # Extract Mark Analysis values
        distance_to_mark = mark_analysis.get('distance_to_mark', 0.0)
        bearing_to_mark = mark_analysis.get('bearing_to_mark', 0.0)
        on_layline = mark_analysis.get('on_layline', False)
        suggested_tack = mark_analysis.get('suggested_tack', 'n/a')

        # Extract Performance values
        speed_efficiency = performance.get('speed_efficiency', 0.0)

        # Shared box style
        box_style = {
            'flex': '1',
            'textAlign': 'center',
            'padding': '10px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'margin': '5px'
        }

        # Start Line Analysis boxes
        line_length_box = html.Div([
            html.H4("Line Length"),
            html.Div(f"{line_length:.1f} m")
        ], style=box_style)

        line_bearing_box = html.Div([
            html.H4("Line Bearing"),
            html.Div(f"{line_bearing:.1f}°")
        ], style=box_style)

        favored_end_box = html.Div([
            html.H4("Favored End"),
            html.Div(f"{favored_end}")
        ], style=box_style)

        bias_advantage_box = html.Div([
            html.H4("Bias Advantage"),
            html.Div(f"{bias_advantage:.1f}°")
        ], style=box_style)

        # Mark Analysis boxes
        distance_to_mark_box = html.Div([
            html.H4("Distance to Mark"),
            html.Div(f"{distance_to_mark:.2f} NM")
        ], style=box_style)

        bearing_to_mark_box = html.Div([
            html.H4("Bearing to Mark"),
            html.Div(f"{bearing_to_mark:.1f}°")
        ], style=box_style)

        tack_layline_box = html.Div([
            html.H4("Tack / Layline"),
            html.Div("On Layline" if on_layline else f"{suggested_tack}")
        ], style=box_style)

        time_to_mark_box = html.Div([
            html.H4("Est. Time to Mark"),
            html.Div(f"{time_to_mark:.1f} min" if time_to_mark else "n/a")
        ], style=box_style)

        return html.Div([
            # Start Line Analysis
            html.H3("Start Line Analysis", style={'marginTop': '20px'}),
            html.Div([
                line_length_box,
                line_bearing_box,
                favored_end_box,
                bias_advantage_box
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}),

            # Mark Analysis
            html.H3("Mark Analysis", style={'marginTop': '20px'}),
            html.Div([
                distance_to_mark_box,
                bearing_to_mark_box,
                tack_layline_box,
                time_to_mark_box
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'})
        ])

    @app.callback(
        Output('set-start-line', 'children'),
        [Input('set-start-line', 'n_clicks')],
        [State('pin-lat', 'value'),
         State('pin-lon', 'value'),
         State('committee-lat', 'value'),
         State('committee-lon', 'value')]
    )
    def set_start_line(n_clicks, pin_lat, pin_lon, com_lat, com_lon):
        if n_clicks is None or not all([pin_lat, pin_lon, com_lat, com_lon]):
            return "Set Start Line"
        
        global start_line
        start_line['pin'] = (pin_lat, pin_lon)
        start_line['committee'] = (com_lat, com_lon)
        
        return "Start Line Set"
    
    return app

if __name__ == "__main__":
    app = create_dash_app()
    app.run_server(debug=False)