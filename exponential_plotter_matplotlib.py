#!/usr/bin/env python3
"""
Interactive plotter for exp(-x / sigma) using matplotlib widgets.
Works without PyQt - uses matplotlib's built-in slider widget.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def exponential_function(x, sigma):
    """Calculate exp(-x / sigma)"""
    return np.exp(-x / sigma)


def main():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.2)

    # Initial sigma value
    sigma_init = 1.0

    # Generate x values
    x = np.linspace(0, 10, 500)
    y_init = exponential_function(x, sigma_init)

    # Initial plot
    line, = ax.plot(x, y_init, 'b-', linewidth=2, label=f'exp(-x / σ)')
    ref_line = ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='y=0.5')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('exp(-x / σ)', fontsize=12)
    ax.set_title(f'Exponential Decay Function (σ = {sigma_init:.2f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.1)

    # Create slider axis
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    sigma_slider = Slider(
        ax=ax_slider,
        label='σ (sigma)',
        valmin=0.1,
        valmax=5.0,
        valinit=sigma_init,
        valstep=0.01
    )

    # Update function for slider
    def update(val):
        sigma = sigma_slider.val
        y = exponential_function(x, sigma)
        line.set_ydata(y)
        ax.set_title(f'Exponential Decay Function (σ = {sigma:.2f})', fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()

    # Connect slider to update function
    sigma_slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()
