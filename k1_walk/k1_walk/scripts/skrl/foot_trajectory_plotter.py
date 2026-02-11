#!/usr/bin/env python3
"""
Streamlit UI for visualizing foot reference height trajectories.
Based on the foot_ref_height function implementation.

Run with: streamlit run foot_trajectory_plotter.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def compute_foot_trajectory(time, target_height, frequency, phase_offset=0.0):
    """
    Compute foot reference height trajectory.

    Args:
        time: Time array (seconds)
        target_height: Maximum height of foot lift (meters)
        frequency: Step frequency (Hz)
        phase_offset: Phase offset for the foot (0.0 for right, 0.5 for left)

    Returns:
        Desired foot height array
    """
    # Phase angle: 2π * (time / frequency + phase_offset)
    phase = 2.0 * np.pi * (time / frequency + phase_offset)

    # Desired height: target_height * clamp(sin(phase), min=0.0)
    # Only positive part of sine wave (foot is on ground when sin < 0)
    desired_height = target_height * np.maximum(np.sin(phase), 0.0)

    return desired_height


def main():
    st.title('Foot Reference Height Trajectory Plotter')
    st.markdown('Visualization of the `foot_ref_height` function from rewards.py')

    # Sidebar for parameters
    st.sidebar.header('Trajectory Parameters')

    target_height = st.sidebar.slider(
        'Target Height (m)',
        min_value=0.05,
        max_value=0.5,
        value=0.2,
        step=0.01,
        help='Maximum height the foot should reach during swing phase'
    )

    frequency = st.sidebar.slider(
        'Step Frequency (Hz)',
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help='Frequency of stepping (steps per second)'
    )

    step_dt = st.sidebar.number_input(
        'Step dt (s)',
        min_value=0.001,
        max_value=0.1,
        value=0.02,
        step=0.001,
        format='%.3f',
        help='Time step for simulation (decimation * physics dt)'
    )

    # Plot range settings
    st.sidebar.header('Plot Settings')

    duration = st.sidebar.slider(
        'Duration (s)',
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help='Total time to plot'
    )

    show_grid = st.sidebar.checkbox('Show Grid', value=True)
    show_phase_info = st.sidebar.checkbox('Show Phase Information', value=True)

    # Generate time array
    time = np.arange(0, duration, step_dt)

    # Compute trajectories
    right_foot_height = compute_foot_trajectory(time, target_height, frequency, phase_offset=0.0)
    left_foot_height = compute_foot_trajectory(time, target_height, frequency, phase_offset=0.5)

    # Create main plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time, right_foot_height, 'b-', linewidth=2, label='Right Foot')
    ax.plot(time, left_foot_height, 'r-', linewidth=2, label='Left Foot')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Foot Height (m)', fontsize=12)
    ax.set_title(f'Foot Reference Height Trajectories (Frequency: {frequency:.1f} Hz, Target Height: {target_height:.2f} m)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.02, target_height * 1.2)

    if show_grid:
        ax.grid(True, alpha=0.3)

    # Add horizontal line at ground level
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Ground Level')

    # Display plot
    st.pyplot(fig)

    # Show phase information
    if show_phase_info:
        st.subheader('Phase Information')

        col1, col2, col3 = st.columns(3)

        period = 1.0 / frequency
        col1.metric('Step Period', f'{period:.3f} s', help='Time for one complete step cycle')
        col2.metric('Phase Offset', 'π rad (0.5 cycle)', help='Left foot lags right foot by half cycle')
        col3.metric('Duty Cycle', '50%', help='Foot is in air for 50% of the cycle')

        st.markdown("""
        ### Trajectory Details

        **Right Foot Phase:** `phase_right = 2π × (time / frequency)`

        **Left Foot Phase:** `phase_left = 2π × (time / frequency + 0.5)`

        **Desired Height:** `desired_height = target_height × max(sin(phase), 0.0)`

        The phase offset of 0.5 (π radians) ensures that when the right foot is at maximum height,
        the left foot is on the ground, creating an alternating gait pattern.

        The `max(sin(phase), 0.0)` clamp ensures that:
        - When sin(phase) > 0: Foot is in swing phase (in the air)
        - When sin(phase) ≤ 0: Foot is in stance phase (on the ground, height = 0)
        """)

    # Show statistics
    st.subheader('Trajectory Statistics')

    col1, col2, col3, col4 = st.columns(4)

    steps_right = np.sum(np.diff(right_foot_height) > 0.01) / 2  # Count rising edges
    steps_left = np.sum(np.diff(left_foot_height) > 0.01) / 2

    col1.metric('Right Foot Steps', f'{int(steps_right)}')
    col2.metric('Left Foot Steps', f'{int(steps_left)}')
    col3.metric('Max Height Reached', f'{target_height:.3f} m')
    col4.metric('Total Samples', f'{len(time)}')

    # Optional: Show detailed phase plot
    if st.sidebar.checkbox('Show Phase Diagram', value=False):
        st.subheader('Phase Diagram')

        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Right foot phase
        phase_right = 2.0 * np.pi * (time / frequency)
        phase_right_wrapped = np.mod(phase_right, 2 * np.pi)

        ax1.plot(time, phase_right_wrapped / np.pi, 'b-', linewidth=2)
        ax1.set_ylabel('Phase (×π rad)', fontsize=11)
        ax1.set_title('Right Foot Phase', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, duration)
        ax1.set_ylim(0, 2)
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='π (peak height)')

        # Left foot phase
        phase_left = 2.0 * np.pi * (time / frequency + 0.5)
        phase_left_wrapped = np.mod(phase_left, 2 * np.pi)

        ax2.plot(time, phase_left_wrapped / np.pi, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Phase (×π rad)', fontsize=11)
        ax2.set_title('Left Foot Phase', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, duration)
        ax2.set_ylim(0, 2)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='π (peak height)')

        plt.tight_layout()
        st.pyplot(fig2)


if __name__ == '__main__':
    main()
