#!/usr/bin/env python3
"""
Web-based interactive plotter for exp(-x / sigma) using Streamlit.
Run with: streamlit run exponential_plotter_streamlit.py
Works in WSL without X server - opens in browser.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def exponential_function(x, sigma):
    """Calculate exp(-x / sigma)"""
    return np.exp(-x / sigma)


def main():
    st.title('Exponential Function Plotter: exp(-x / σ)')

    # Sidebar for parameters
    st.sidebar.header('Parameters')

    # Slider for sigma
    sigma = st.sidebar.slider(
        'σ (sigma)',
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.01
    )

    # Plot range settings
    st.sidebar.header('Plot Range')

    col1, col2 = st.sidebar.columns(2)
    with col1:
        x_min = st.number_input('x min', value=0.0, step=0.5)
        y_min = st.number_input('y min', value=0.0, step=0.1)
    with col2:
        x_max = st.number_input('x max', value=10.0, step=0.5)
        y_max = st.number_input('y max', value=1.1, step=0.1)

    # Validate ranges
    if x_min >= x_max:
        st.error('x min must be less than x max')
        return
    if y_min >= y_max:
        st.error('y min must be less than y max')
        return

    # Generate x values
    x = np.linspace(x_min, x_max, 500)
    y = exponential_function(x, sigma)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2, label=f'exp(-x / σ), σ={sigma:.2f}')

    # Add reference line at y=0.5 if it's within range
    if y_min <= 0.5 <= y_max:
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='y=0.5')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('exp(-x / σ)', fontsize=12)
    ax.set_title(f'Exponential Decay Function (σ = {sigma:.2f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Display plot
    st.pyplot(fig)

    # Show some statistics
    st.subheader('Function Properties')
    col1, col2, col3 = st.columns(3)
    col1.metric('σ value', f'{sigma:.2f}')
    col2.metric('Half-life (x at y=0.5)', f'{sigma * np.log(2):.2f}')
    col3.metric('Value at x=σ', f'{np.exp(-1):.3f}')


if __name__ == '__main__':
    main()
