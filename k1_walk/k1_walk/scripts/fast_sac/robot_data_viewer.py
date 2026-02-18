"""
Streamlit app for visualizing logged robot data.

Usage:
    streamlit run robot_data_viewer.py
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import glob

st.set_page_config(page_title="Robot Data Viewer", layout="wide")

st.title("🤖 Robot Data Viewer")
st.markdown("Visualize robot posture, foot trajectories, and command tracking data")


@st.cache_data
def load_pickle_data(filepath):
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


@st.cache_data
def load_csv_data(csv_dir, env_id):
    """Load data from CSV file."""
    csv_path = Path(csv_dir) / f"robot_data_env_{env_id}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None


def normalize_angle_deg(angle_deg):
    """Normalize angle to [-180, 180] range."""
    return ((angle_deg + 180) % 360) - 180


def plot_robot_orientation(data, env_idx, time_range):
    """Plot roll, pitch, yaw over time."""
    start_idx, end_idx = time_range
    euler = data['base_euler'][start_idx:end_idx, env_idx, :]
    steps = np.arange(start_idx, end_idx)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Roll', 'Pitch', 'Yaw'),
        vertical_spacing=0.08
    )

    # Normalize angles to [-180, 180] range
    roll_deg = normalize_angle_deg(np.rad2deg(euler[:, 0]))
    pitch_deg = normalize_angle_deg(np.rad2deg(euler[:, 1]))
    yaw_deg = normalize_angle_deg(np.rad2deg(euler[:, 2]))

    # Roll
    fig.add_trace(
        go.Scatter(x=steps, y=roll_deg, name='Roll', line=dict(color='red')),
        row=1, col=1
    )

    # Pitch
    fig.add_trace(
        go.Scatter(x=steps, y=pitch_deg, name='Pitch', line=dict(color='green')),
        row=2, col=1
    )

    # Yaw
    fig.add_trace(
        go.Scatter(x=steps, y=yaw_deg, name='Yaw', line=dict(color='blue')),
        row=3, col=1
    )

    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_yaxes(title_text="Roll (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Pitch (deg)", row=2, col=1)
    fig.update_yaxes(title_text="Yaw (deg)", row=3, col=1)

    fig.update_layout(height=600, showlegend=False, title_text="Robot Orientation")

    return fig


def plot_base_position(data, env_idx, time_range):
    """Plot base position over time."""
    start_idx, end_idx = time_range
    pos = data['base_pos'][start_idx:end_idx, env_idx, :]
    steps = np.arange(start_idx, end_idx)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=steps, y=pos[:, 0], name='X', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=steps, y=pos[:, 1], name='Y', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=steps, y=pos[:, 2], name='Z', line=dict(color='blue')))

    fig.update_layout(
        title="Base Position",
        xaxis_title="Step",
        yaxis_title="Position (m)",
        height=400
    )

    return fig


def plot_foot_trajectories_3d(data, env_idx, time_range):
    """Plot 3D foot trajectories."""
    start_idx, end_idx = time_range
    left_foot = data['left_foot_pos'][start_idx:end_idx, env_idx, :]
    right_foot = data['right_foot_pos'][start_idx:end_idx, env_idx, :]

    fig = go.Figure()

    # Left foot trajectory
    fig.add_trace(go.Scatter3d(
        x=left_foot[:, 0], y=left_foot[:, 1], z=left_foot[:, 2],
        mode='lines',
        name='Left Foot',
        line=dict(color='blue', width=3)
    ))

    # Right foot trajectory
    fig.add_trace(go.Scatter3d(
        x=right_foot[:, 0], y=right_foot[:, 1], z=right_foot[:, 2],
        mode='lines',
        name='Right Foot',
        line=dict(color='red', width=3)
    ))

    # Start markers
    fig.add_trace(go.Scatter3d(
        x=[left_foot[0, 0]], y=[left_foot[0, 1]], z=[left_foot[0, 2]],
        mode='markers',
        name='Left Start',
        marker=dict(color='blue', size=5, symbol='circle')
    ))

    fig.add_trace(go.Scatter3d(
        x=[right_foot[0, 0]], y=[right_foot[0, 1]], z=[right_foot[0, 2]],
        mode='markers',
        name='Right Start',
        marker=dict(color='red', size=5, symbol='circle')
    ))

    fig.update_layout(
        title="Foot Trajectories (3D)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        ),
        height=600
    )

    return fig


def plot_foot_height(data, env_idx, time_range):
    """Plot foot height over time."""
    start_idx, end_idx = time_range
    left_foot = data['left_foot_pos'][start_idx:end_idx, env_idx, 2]
    right_foot = data['right_foot_pos'][start_idx:end_idx, env_idx, 2]
    steps = np.arange(start_idx, end_idx)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=steps, y=left_foot, name='Left Foot', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=steps, y=right_foot, name='Right Foot', line=dict(color='red')))

    fig.update_layout(
        title="Foot Height",
        xaxis_title="Step",
        yaxis_title="Height (m)",
        height=400
    )

    return fig


def plot_foot_xy_trajectory(data, env_idx, time_range):
    """Plot foot XY trajectory (top-down view)."""
    start_idx, end_idx = time_range
    left_foot = data['left_foot_pos'][start_idx:end_idx, env_idx, :]
    right_foot = data['right_foot_pos'][start_idx:end_idx, env_idx, :]
    base_pos = data['base_pos'][start_idx:end_idx, env_idx, :]

    fig = go.Figure()

    # Base trajectory
    fig.add_trace(go.Scatter(
        x=base_pos[:, 0], y=base_pos[:, 1],
        mode='lines',
        name='Base',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # Left foot trajectory
    fig.add_trace(go.Scatter(
        x=left_foot[:, 0], y=left_foot[:, 1],
        mode='lines+markers',
        name='Left Foot',
        line=dict(color='blue', width=2),
        marker=dict(size=3)
    ))

    # Right foot trajectory
    fig.add_trace(go.Scatter(
        x=right_foot[:, 0], y=right_foot[:, 1],
        mode='lines+markers',
        name='Right Foot',
        line=dict(color='red', width=2),
        marker=dict(size=3)
    ))

    fig.update_layout(
        title="Foot Trajectory (Top View)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig


def plot_velocity_tracking(data, env_idx, time_range):
    """Plot commanded vs actual velocities."""
    start_idx, end_idx = time_range
    steps = np.arange(start_idx, end_idx)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Linear Velocity X', 'Linear Velocity Y', 'Angular Velocity Z'),
        vertical_spacing=0.08
    )

    # Linear X
    fig.add_trace(
        go.Scatter(x=steps, y=data['cmd_lin_vel_x'][start_idx:end_idx, env_idx],
                  name='Cmd X', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=steps, y=data['actual_lin_vel_x'][start_idx:end_idx, env_idx],
                  name='Actual X', line=dict(color='red')),
        row=1, col=1
    )

    # Linear Y
    fig.add_trace(
        go.Scatter(x=steps, y=data['cmd_lin_vel_y'][start_idx:end_idx, env_idx],
                  name='Cmd Y', line=dict(color='green', dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=steps, y=data['actual_lin_vel_y'][start_idx:end_idx, env_idx],
                  name='Actual Y', line=dict(color='green')),
        row=2, col=1
    )

    # Angular Z
    fig.add_trace(
        go.Scatter(x=steps, y=data['cmd_ang_vel_z'][start_idx:end_idx, env_idx],
                  name='Cmd Z', line=dict(color='blue', dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=steps, y=data['actual_ang_vel_z'][start_idx:end_idx, env_idx],
                  name='Actual Z', line=dict(color='blue')),
        row=3, col=1
    )

    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_yaxes(title_text="Vel (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Vel (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Vel (rad/s)", row=3, col=1)

    fig.update_layout(height=700, title_text="Velocity Tracking (Command vs Actual)")

    return fig


def plot_tracking_error(data, env_idx, time_range):
    """Plot tracking errors."""
    start_idx, end_idx = time_range
    steps = np.arange(start_idx, end_idx)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Linear Velocity Error', 'Angular Velocity Error'),
        vertical_spacing=0.15
    )

    # Linear velocity error
    fig.add_trace(
        go.Scatter(x=steps, y=data['lin_vel_error'][start_idx:end_idx, env_idx],
                  name='Lin Vel Error', line=dict(color='red'), fill='tozeroy'),
        row=1, col=1
    )

    # Angular velocity error
    fig.add_trace(
        go.Scatter(x=steps, y=data['ang_vel_error'][start_idx:end_idx, env_idx],
                  name='Ang Vel Error', line=dict(color='blue'), fill='tozeroy'),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="Error (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Error (rad/s)", row=2, col=1)

    fig.update_layout(height=500, showlegend=False, title_text="Tracking Errors")

    return fig


# Sidebar for file selection and parameters
st.sidebar.header("📁 Data Selection")

data_source = st.sidebar.radio("Data Source", ["Pickle File", "CSV Directory"])

data = None
num_envs = 0

if data_source == "Pickle File":
    # Find all pickle files in review/play_data
    pickle_files = glob.glob("review/play_data/*.pkl")

    if pickle_files:
        selected_file = st.sidebar.selectbox("Select Pickle File", pickle_files)

        if selected_file:
            loaded_data = load_pickle_data(selected_file)
            data = loaded_data['data']
            num_envs = len(loaded_data['env_ids'])
            st.sidebar.success(f"Loaded {loaded_data['num_steps']} steps")
    else:
        st.sidebar.warning("No pickle files found in review/play_data/")

else:  # CSV Directory
    csv_dirs = glob.glob("review/play_data/csv_*")

    if csv_dirs:
        selected_dir = st.sidebar.selectbox("Select CSV Directory", csv_dirs)
        env_id = st.sidebar.number_input("Environment ID", min_value=0, value=0)

        df = load_csv_data(selected_dir, env_id)

        if df is not None:
            # Convert CSV data to the same format as pickle
            data = {
                'base_pos': df[['base_pos_x', 'base_pos_y', 'base_pos_z']].values[:, None, :],
                'base_euler': df[['base_roll', 'base_pitch', 'base_yaw']].values[:, None, :],
                'left_foot_pos': df[['left_foot_x', 'left_foot_y', 'left_foot_z']].values[:, None, :],
                'right_foot_pos': df[['right_foot_x', 'right_foot_y', 'right_foot_z']].values[:, None, :],
                'cmd_lin_vel_x': df['cmd_lin_vel_x'].values[:, None],
                'cmd_lin_vel_y': df['cmd_lin_vel_y'].values[:, None],
                'cmd_ang_vel_z': df['cmd_ang_vel_z'].values[:, None],
                'actual_lin_vel_x': df['actual_lin_vel_x'].values[:, None],
                'actual_lin_vel_y': df['actual_lin_vel_y'].values[:, None],
                'actual_ang_vel_z': df['actual_ang_vel_z'].values[:, None],
                'lin_vel_error': df['lin_vel_error'].values[:, None],
                'ang_vel_error': df['ang_vel_error'].values[:, None],
            }
            num_envs = 1
            st.sidebar.success(f"Loaded {len(df)} steps")
    else:
        st.sidebar.warning("No CSV directories found in review/play_data/")

if data is not None:
    st.sidebar.header("⚙️ Parameters")

    # Environment selection (for pickle data)
    if data_source == "Pickle File":
        env_idx = st.sidebar.number_input("Environment Index", min_value=0, max_value=num_envs-1, value=0)
    else:
        env_idx = 0

    # Time range selection
    max_steps = data['base_pos'].shape[0]
    time_range = st.sidebar.slider(
        "Time Range (steps)",
        min_value=0,
        max_value=max_steps-1,
        value=(0, min(1000, max_steps-1))
    )

    # Display options
    st.sidebar.header("📊 Display Options")
    show_orientation = st.sidebar.checkbox("Robot Orientation", value=True)
    show_base_pos = st.sidebar.checkbox("Base Position", value=True)
    show_foot_3d = st.sidebar.checkbox("Foot Trajectories 3D", value=True)
    show_foot_xy = st.sidebar.checkbox("Foot Trajectories XY", value=True)
    show_foot_height = st.sidebar.checkbox("Foot Height", value=True)
    show_velocity = st.sidebar.checkbox("Velocity Tracking", value=True)
    show_error = st.sidebar.checkbox("Tracking Error", value=True)

    # Main content area
    st.header("📈 Visualizations")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🤖 Robot State", "👣 Foot Trajectories", "🎯 Command Tracking"])

    with tab1:
        if show_orientation:
            st.plotly_chart(plot_robot_orientation(data, env_idx, time_range), use_container_width=True)

        if show_base_pos:
            st.plotly_chart(plot_base_position(data, env_idx, time_range), use_container_width=True)

    with tab2:
        if show_foot_3d:
            st.plotly_chart(plot_foot_trajectories_3d(data, env_idx, time_range), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            if show_foot_xy:
                st.plotly_chart(plot_foot_xy_trajectory(data, env_idx, time_range), use_container_width=True)

        with col2:
            if show_foot_height:
                st.plotly_chart(plot_foot_height(data, env_idx, time_range), use_container_width=True)

    with tab3:
        if show_velocity:
            st.plotly_chart(plot_velocity_tracking(data, env_idx, time_range), use_container_width=True)

        if show_error:
            st.plotly_chart(plot_tracking_error(data, env_idx, time_range), use_container_width=True)

    # Statistics summary
    st.header("📊 Statistics Summary")

    start_idx, end_idx = time_range

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_pitch = np.mean(np.abs(data['base_euler'][start_idx:end_idx, env_idx, 1]))
        st.metric("Avg |Pitch|", f"{np.rad2deg(avg_pitch):.2f}°")

    with col2:
        avg_roll = np.mean(np.abs(data['base_euler'][start_idx:end_idx, env_idx, 0]))
        st.metric("Avg |Roll|", f"{np.rad2deg(avg_roll):.2f}°")

    with col3:
        avg_lin_error = np.mean(data['lin_vel_error'][start_idx:end_idx, env_idx])
        st.metric("Avg Lin Vel Error", f"{avg_lin_error:.3f} m/s")

    with col4:
        avg_ang_error = np.mean(data['ang_vel_error'][start_idx:end_idx, env_idx])
        st.metric("Avg Ang Vel Error", f"{avg_ang_error:.3f} rad/s")

else:
    st.info("👈 Please select a data file from the sidebar to start visualization")

    st.markdown("""
    ## How to use this viewer:

    1. **Run data logging** during play:
       ```bash
       python k1_walk/k1_walk/scripts/skrl/play.py --task Isaac-K1-Rough-v0 --checkpoint path/to/checkpoint.pt --log_data
       ```

    2. **Launch this viewer**:
       ```bash
       streamlit run robot_data_viewer.py
       ```

    3. **Select data source** from the sidebar (Pickle or CSV)

    4. **Adjust parameters** and toggle visualizations as needed
    """)
