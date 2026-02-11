#!/usr/bin/env python3
"""
Interactive GUI for plotting exp(-x / sigma) with adjustable sigma parameter.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ExponentialPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Exponential Function Plotter: exp(-x / σ)')
        self.setGeometry(100, 100, 900, 700)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        main_layout.addWidget(self.canvas)

        # Slider layout
        slider_layout = QHBoxLayout()

        # Sigma label
        sigma_label = QLabel('σ (sigma):')
        slider_layout.addWidget(sigma_label)

        # Sigma slider (0.1 to 5.0)
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setMinimum(10)  # 0.1 * 100
        self.sigma_slider.setMaximum(500)  # 5.0 * 100
        self.sigma_slider.setValue(100)  # Default: 1.0 * 100
        self.sigma_slider.setTickPosition(QSlider.TicksBelow)
        self.sigma_slider.setTickInterval(50)
        self.sigma_slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(self.sigma_slider)

        # Sigma value label
        self.sigma_value_label = QLabel('1.00')
        self.sigma_value_label.setMinimumWidth(60)
        slider_layout.addWidget(self.sigma_value_label)

        main_layout.addLayout(slider_layout)

        # Initial plot
        self.update_plot()

    def update_plot(self):
        # Get sigma value from slider (divided by 100 for decimal precision)
        sigma = self.sigma_slider.value() / 100.0
        self.sigma_value_label.setText(f'{sigma:.2f}')

        # Generate x values
        x = np.linspace(0, 10, 500)

        # Calculate y = exp(-x / sigma)
        y = np.exp(-x / sigma)

        # Clear and plot
        self.ax.clear()
        self.ax.plot(x, y, 'b-', linewidth=2, label=f'exp(-x / σ), σ={sigma:.2f}')
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('exp(-x / σ)', fontsize=12)
        self.ax.set_title(f'Exponential Decay Function (σ = {sigma:.2f})', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=10)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 1.1)

        # Add horizontal line at y=0.5 for reference
        self.ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='y=0.5')

        # Refresh canvas
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    plotter = ExponentialPlotter()
    plotter.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
