"""
orientation_potential Reward Function Visualization GUI

Interactively plot quat_rotate_inverse output and orientation_potential values
for various roll and pitch angles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# Use a font that supports a wide range of characters
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


def euler_to_quat_wxyz(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    """Convert Euler angles (XYZ order) to quaternion (wxyz format)"""
    # scipy returns xyzw format, convert to wxyz
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    q_xyzw = r.as_quat()  # [x, y, z, w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # [w, x, y, z]


def quat_rotate_inverse_numpy(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Apply inverse quaternion rotation to a vector.
    Equivalent to isaaclab.utils.math.quat_rotate_inverse

    Args:
        q_wxyz: Quaternion in (w, x, y, z) format
        v: 3D vector

    Returns:
        Rotated vector
    """
    # Convert wxyz -> xyzw
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    r = Rotation.from_quat(q_xyzw)
    # Apply inverse rotation
    return r.inv().apply(v)


def compute_orientation_potential(roll: float, pitch: float, sigma: float = 0.5,
                                   enable_exp_func: bool = True) -> dict:
    """
    Compute orientation_potential for given roll and pitch angles.

    Returns:
        dict with keys: upright_vector, ux, uy, uz, err_value, potential
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    q_wxyz = euler_to_quat_wxyz(roll, pitch, 0.0)
    upright_vector = quat_rotate_inverse_numpy(q_wxyz, z_axis)

    ux, uy, uz = upright_vector
    err_value = ux**2 + uy**2

    if enable_exp_func:
        potential = np.exp(-err_value / sigma)
    else:
        potential = -err_value / sigma

    return {
        'upright_vector': upright_vector,
        'ux': ux,
        'uy': uy,
        'uz': uz,
        'err_value': err_value,
        'potential': potential
    }


def create_visualization():
    """Create interactive visualization GUI"""

    # Angle range (degrees)
    angle_range = np.linspace(-90, 90, 100)
    roll_deg, pitch_deg = np.meshgrid(angle_range, angle_range)

    # Convert to radians
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)

    # Initial parameters
    initial_sigma = 0.5
    initial_exp_func = True

    def compute_all_values(sigma, enable_exp_func):
        """Compute values at all grid points"""
        ux = np.zeros_like(roll_rad)
        uy = np.zeros_like(roll_rad)
        uz = np.zeros_like(roll_rad)
        err_value = np.zeros_like(roll_rad)
        potential = np.zeros_like(roll_rad)

        for i in range(roll_rad.shape[0]):
            for j in range(roll_rad.shape[1]):
                result = compute_orientation_potential(
                    roll_rad[i, j], pitch_rad[i, j],
                    sigma, enable_exp_func
                )
                ux[i, j] = result['ux']
                uy[i, j] = result['uy']
                uz[i, j] = result['uz']
                err_value[i, j] = result['err_value']
                potential[i, j] = result['potential']

        return ux, uy, uz, err_value, potential

    # Compute initial values
    ux, uy, uz, err_value, potential = compute_all_values(initial_sigma, initial_exp_func)

    # Figure creation
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('orientation_potential Reward Function Visualization', fontsize=14)

    # Subplot layout
    # Top row: 3D plot (potential), contour (potential)
    # Middle row: upright_vector components (ux, uy)
    # Bottom: sliders and controls

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    # 3D Surface Plot - Potential
    surf1 = ax1.plot_surface(roll_deg, pitch_deg, potential, cmap='viridis',
                              edgecolor='none', alpha=0.8)
    ax1.set_xlabel('Roll [deg]')
    ax1.set_ylabel('Pitch [deg]')
    ax1.set_zlabel('Potential')
    ax1.set_title('Potential (3D)')

    # Contour Plot - Potential
    cont2 = ax2.contourf(roll_deg, pitch_deg, potential, levels=50, cmap='viridis')
    cbar2 = fig.colorbar(cont2, ax=ax2)
    ax2.set_xlabel('Roll [deg]')
    ax2.set_ylabel('Pitch [deg]')
    ax2.set_title('Potential (Contour)')
    # Add contour lines
    ax2.contour(roll_deg, pitch_deg, potential, levels=10, colors='white', linewidths=0.5)

    # err_value (ux² + uy²)
    cont3 = ax3.contourf(roll_deg, pitch_deg, err_value, levels=50, cmap='hot')
    cbar3 = fig.colorbar(cont3, ax=ax3)
    ax3.set_xlabel('Roll [deg]')
    ax3.set_ylabel('Pitch [deg]')
    ax3.set_title('err_value (ux² + uy²)')
    ax3.contour(roll_deg, pitch_deg, err_value, levels=10, colors='white', linewidths=0.5)

    # ux component
    cont4 = ax4.contourf(roll_deg, pitch_deg, ux, levels=50, cmap='RdBu_r')
    cbar4 = fig.colorbar(cont4, ax=ax4)
    ax4.set_xlabel('Roll [deg]')
    ax4.set_ylabel('Pitch [deg]')
    ax4.set_title('upright_vector[x] (ux)')
    ax4.contour(roll_deg, pitch_deg, ux, levels=10, colors='black', linewidths=0.5)

    # uy component
    cont5 = ax5.contourf(roll_deg, pitch_deg, uy, levels=50, cmap='RdBu_r')
    cbar5 = fig.colorbar(cont5, ax=ax5)
    ax5.set_xlabel('Roll [deg]')
    ax5.set_ylabel('Pitch [deg]')
    ax5.set_title('upright_vector[y] (uy)')
    ax5.contour(roll_deg, pitch_deg, uy, levels=10, colors='black', linewidths=0.5)

    # uz component
    cont6 = ax6.contourf(roll_deg, pitch_deg, uz, levels=50, cmap='RdBu_r')
    cbar6 = fig.colorbar(cont6, ax=ax6)
    ax6.set_xlabel('Roll [deg]')
    ax6.set_ylabel('Pitch [deg]')
    ax6.set_title('upright_vector[z] (uz)')
    ax6.contour(roll_deg, pitch_deg, uz, levels=10, colors='black', linewidths=0.5)

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])

    # Axes for sliders
    ax_sigma = plt.axes([0.15, 0.08, 0.55, 0.03])
    ax_checkbox = plt.axes([0.75, 0.02, 0.2, 0.1])

    # Sigma slider
    sigma_slider = Slider(
        ax=ax_sigma,
        label='sigma',
        valmin=0.01,
        valmax=2.0,
        valinit=initial_sigma,
        valstep=0.01
    )

    # Enable exp function checkbox
    check = CheckButtons(ax_checkbox, ['enable_exp_func'], [initial_exp_func])

    # Update function
    def update(val=None):
        sigma = sigma_slider.val
        enable_exp_func = check.get_status()[0]

        # Recompute values
        ux_new, uy_new, uz_new, err_value_new, potential_new = compute_all_values(sigma, enable_exp_func)

        # Update plots
        ax1.clear()
        ax1.plot_surface(roll_deg, pitch_deg, potential_new, cmap='viridis',
                        edgecolor='none', alpha=0.8)
        ax1.set_xlabel('Roll [deg]')
        ax1.set_ylabel('Pitch [deg]')
        ax1.set_zlabel('Potential')
        ax1.set_title('Potential (3D)')

        ax2.clear()
        cont2_new = ax2.contourf(roll_deg, pitch_deg, potential_new, levels=50, cmap='viridis')
        ax2.contour(roll_deg, pitch_deg, potential_new, levels=10, colors='white', linewidths=0.5)
        ax2.set_xlabel('Roll [deg]')
        ax2.set_ylabel('Pitch [deg]')
        ax2.set_title('Potential (Contour)')

        ax3.clear()
        ax3.contourf(roll_deg, pitch_deg, err_value_new, levels=50, cmap='hot')
        ax3.contour(roll_deg, pitch_deg, err_value_new, levels=10, colors='white', linewidths=0.5)
        ax3.set_xlabel('Roll [deg]')
        ax3.set_ylabel('Pitch [deg]')
        ax3.set_title('err_value (ux² + uy²)')

        ax4.clear()
        ax4.contourf(roll_deg, pitch_deg, ux_new, levels=50, cmap='RdBu_r')
        ax4.contour(roll_deg, pitch_deg, ux_new, levels=10, colors='black', linewidths=0.5)
        ax4.set_xlabel('Roll [deg]')
        ax4.set_ylabel('Pitch [deg]')
        ax4.set_title('upright_vector[x] (ux)')

        ax5.clear()
        ax5.contourf(roll_deg, pitch_deg, uy_new, levels=50, cmap='RdBu_r')
        ax5.contour(roll_deg, pitch_deg, uy_new, levels=10, colors='black', linewidths=0.5)
        ax5.set_xlabel('Roll [deg]')
        ax5.set_ylabel('Pitch [deg]')
        ax5.set_title('upright_vector[y] (uy)')

        ax6.clear()
        ax6.contourf(roll_deg, pitch_deg, uz_new, levels=50, cmap='RdBu_r')
        ax6.contour(roll_deg, pitch_deg, uz_new, levels=10, colors='black', linewidths=0.5)
        ax6.set_xlabel('Roll [deg]')
        ax6.set_ylabel('Pitch [deg]')
        ax6.set_title('upright_vector[z] (uz)')

        fig.canvas.draw_idle()

    sigma_slider.on_changed(update)
    check.on_clicked(update)

    # Create another window showing values at specific angles
    def create_slice_plot():
        """Slice plots with fixed roll or pitch"""
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        fig2.suptitle('orientation_potential: Fixed Angle Slices', fontsize=14)

        angles = np.linspace(-90, 90, 200)
        angles_rad = np.deg2rad(angles)

        sigma = 0.5

        # Roll = 0, pitch varying
        potential_pitch = []
        ux_pitch = []
        uy_pitch = []
        for p in angles_rad:
            result = compute_orientation_potential(0, p, sigma, True)
            potential_pitch.append(result['potential'])
            ux_pitch.append(result['ux'])
            uy_pitch.append(result['uy'])

        axes2[0, 0].plot(angles, potential_pitch, 'b-', linewidth=2)
        axes2[0, 0].set_xlabel('Pitch [deg]')
        axes2[0, 0].set_ylabel('Potential')
        axes2[0, 0].set_title('Roll = 0, Pitch varying')
        axes2[0, 0].grid(True)
        axes2[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='potential=0.5')
        axes2[0, 0].legend()

        ax_twin = axes2[0, 0].twinx()
        ax_twin.plot(angles, ux_pitch, 'g--', linewidth=1, alpha=0.7, label='ux')
        ax_twin.set_ylabel('ux', color='g')

        # Pitch = 0, Roll varying
        potential_roll = []
        ux_roll = []
        uy_roll = []
        for r in angles_rad:
            result = compute_orientation_potential(r, 0, sigma, True)
            potential_roll.append(result['potential'])
            ux_roll.append(result['ux'])
            uy_roll.append(result['uy'])

        axes2[0, 1].plot(angles, potential_roll, 'b-', linewidth=2)
        axes2[0, 1].set_xlabel('Roll [deg]')
        axes2[0, 1].set_ylabel('Potential')
        axes2[0, 1].set_title('Pitch = 0, Roll varying')
        axes2[0, 1].grid(True)
        axes2[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='potential=0.5')
        axes2[0, 1].legend()

        ax_twin2 = axes2[0, 1].twinx()
        ax_twin2.plot(angles, uy_roll, 'g--', linewidth=1, alpha=0.7, label='uy')
        ax_twin2.set_ylabel('uy', color='g')

        # Compare different sigma values (Pitch varying)
        sigmas = [0.1, 0.25, 0.5, 1.0, 2.0]
        for s in sigmas:
            potential_s = []
            for p in angles_rad:
                result = compute_orientation_potential(0, p, s, True)
                potential_s.append(result['potential'])
            axes2[1, 0].plot(angles, potential_s, linewidth=2, label=f'sigma={s}')

        axes2[1, 0].set_xlabel('Pitch [deg]')
        axes2[1, 0].set_ylabel('Potential')
        axes2[1, 0].set_title('Potential vs sigma (Roll=0)')
        axes2[1, 0].grid(True)
        axes2[1, 0].legend()

        # Compare enable_exp_func True vs False
        potential_exp = []
        potential_no_exp = []
        for p in angles_rad:
            result_exp = compute_orientation_potential(0, p, sigma, True)
            result_no_exp = compute_orientation_potential(0, p, sigma, False)
            potential_exp.append(result_exp['potential'])
            potential_no_exp.append(result_no_exp['potential'])

        axes2[1, 1].plot(angles, potential_exp, 'b-', linewidth=2, label='enable_exp_func=True')
        axes2[1, 1].plot(angles, potential_no_exp, 'r-', linewidth=2, label='enable_exp_func=False')
        axes2[1, 1].set_xlabel('Pitch [deg]')
        axes2[1, 1].set_ylabel('Potential')
        axes2[1, 1].set_title(f'exp function comparison (sigma={sigma})')
        axes2[1, 1].grid(True)
        axes2[1, 1].legend()

        plt.tight_layout()

    def create_raw_upright_vector_plot():
        """Plot raw upright_vector components (ux, uy) in detail"""
        fig3, axes3 = plt.subplots(2, 3, figsize=(15, 9))
        fig3.suptitle('upright_vector Raw Values: quat_rotate_inverse(root_quat_w, [0,0,1])', fontsize=14)

        angles = np.linspace(-90, 90, 200)
        angles_rad = np.deg2rad(angles)

        # ===== Row 1: 1D slices =====

        # --- Panel [0,0]: ux vs Pitch (Roll=0) ---
        # When Roll=0, pitch rotation affects ux (forward/backward tilt)
        ux_pitch_r0 = []
        uy_pitch_r0 = []
        uz_pitch_r0 = []
        for p in angles_rad:
            result = compute_orientation_potential(0, p, 0.5, True)
            ux_pitch_r0.append(result['ux'])
            uy_pitch_r0.append(result['uy'])
            uz_pitch_r0.append(result['uz'])

        axes3[0, 0].plot(angles, ux_pitch_r0, 'r-', linewidth=2, label='ux (forward/back)')
        axes3[0, 0].plot(angles, uy_pitch_r0, 'g-', linewidth=2, label='uy (left/right)')
        axes3[0, 0].plot(angles, uz_pitch_r0, 'b-', linewidth=2, label='uz (vertical)')
        axes3[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes3[0, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes3[0, 0].set_xlabel('Pitch [deg]')
        axes3[0, 0].set_ylabel('upright_vector component')
        axes3[0, 0].set_title('Roll = 0: Pitch varying\n(ux = -sin(pitch))')
        axes3[0, 0].grid(True, alpha=0.3)
        axes3[0, 0].legend(loc='best')
        axes3[0, 0].set_ylim(-1.1, 1.1)

        # Add sin reference
        sin_pitch = -np.sin(angles_rad)
        axes3[0, 0].plot(angles, sin_pitch, 'r--', linewidth=1, alpha=0.5, label='-sin(pitch)')

        # --- Panel [0,1]: uy vs Roll (Pitch=0) ---
        # When Pitch=0, roll rotation affects uy (left/right tilt)
        ux_roll_p0 = []
        uy_roll_p0 = []
        uz_roll_p0 = []
        for r in angles_rad:
            result = compute_orientation_potential(r, 0, 0.5, True)
            ux_roll_p0.append(result['ux'])
            uy_roll_p0.append(result['uy'])
            uz_roll_p0.append(result['uz'])

        axes3[0, 1].plot(angles, ux_roll_p0, 'r-', linewidth=2, label='ux (forward/back)')
        axes3[0, 1].plot(angles, uy_roll_p0, 'g-', linewidth=2, label='uy (left/right)')
        axes3[0, 1].plot(angles, uz_roll_p0, 'b-', linewidth=2, label='uz (vertical)')
        axes3[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes3[0, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes3[0, 1].set_xlabel('Roll [deg]')
        axes3[0, 1].set_ylabel('upright_vector component')
        axes3[0, 1].set_title('Pitch = 0: Roll varying\n(uy = sin(roll))')
        axes3[0, 1].grid(True, alpha=0.3)
        axes3[0, 1].legend(loc='best')
        axes3[0, 1].set_ylim(-1.1, 1.1)

        # Add sin reference
        sin_roll = np.sin(angles_rad)
        axes3[0, 1].plot(angles, sin_roll, 'g--', linewidth=1, alpha=0.5, label='sin(roll)')

        # --- Panel [0,2]: Both ux and uy vs angle (combined view) ---
        axes3[0, 2].plot(angles, ux_pitch_r0, 'r-', linewidth=2, label='ux (Pitch, Roll=0)')
        axes3[0, 2].plot(angles, uy_roll_p0, 'g-', linewidth=2, label='uy (Roll, Pitch=0)')
        axes3[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes3[0, 2].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes3[0, 2].set_xlabel('Angle [deg]')
        axes3[0, 2].set_ylabel('upright_vector component')
        axes3[0, 2].set_title('ux vs Pitch and uy vs Roll\n(Both are sin functions)')
        axes3[0, 2].grid(True, alpha=0.3)
        axes3[0, 2].legend(loc='best')
        axes3[0, 2].set_ylim(-1.1, 1.1)

        # Highlight key angles
        for angle in [10, 20, 30, 45]:
            axes3[0, 2].axvline(x=angle, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            axes3[0, 2].axvline(x=-angle, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        # ===== Row 2: 2D heatmaps and combined effects =====

        # Create meshgrid for 2D plots
        angle_range_2d = np.linspace(-60, 60, 80)
        roll_2d, pitch_2d = np.meshgrid(angle_range_2d, angle_range_2d)
        roll_rad_2d = np.deg2rad(roll_2d)
        pitch_rad_2d = np.deg2rad(pitch_2d)

        ux_2d = np.zeros_like(roll_rad_2d)
        uy_2d = np.zeros_like(roll_rad_2d)
        err_2d = np.zeros_like(roll_rad_2d)

        for i in range(roll_rad_2d.shape[0]):
            for j in range(roll_rad_2d.shape[1]):
                result = compute_orientation_potential(roll_rad_2d[i, j], pitch_rad_2d[i, j], 0.5, True)
                ux_2d[i, j] = result['ux']
                uy_2d[i, j] = result['uy']
                err_2d[i, j] = result['err_value']

        # --- Panel [1,0]: ux heatmap ---
        im1 = axes3[1, 0].contourf(roll_2d, pitch_2d, ux_2d, levels=30, cmap='RdBu_r')
        axes3[1, 0].contour(roll_2d, pitch_2d, ux_2d, levels=[-0.5, -0.25, 0, 0.25, 0.5], colors='black', linewidths=1)
        fig3.colorbar(im1, ax=axes3[1, 0])
        axes3[1, 0].set_xlabel('Roll [deg]')
        axes3[1, 0].set_ylabel('Pitch [deg]')
        axes3[1, 0].set_title('ux = upright_vector[0]\n(mainly affected by Pitch)')
        axes3[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes3[1, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # --- Panel [1,1]: uy heatmap ---
        im2 = axes3[1, 1].contourf(roll_2d, pitch_2d, uy_2d, levels=30, cmap='RdBu_r')
        axes3[1, 1].contour(roll_2d, pitch_2d, uy_2d, levels=[-0.5, -0.25, 0, 0.25, 0.5], colors='black', linewidths=1)
        fig3.colorbar(im2, ax=axes3[1, 1])
        axes3[1, 1].set_xlabel('Roll [deg]')
        axes3[1, 1].set_ylabel('Pitch [deg]')
        axes3[1, 1].set_title('uy = upright_vector[1]\n(mainly affected by Roll)')
        axes3[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes3[1, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # --- Panel [1,2]: err_value = ux² + uy² heatmap ---
        im3 = axes3[1, 2].contourf(roll_2d, pitch_2d, err_2d, levels=30, cmap='hot')
        axes3[1, 2].contour(roll_2d, pitch_2d, err_2d, levels=[0.1, 0.25, 0.5, 0.75, 1.0], colors='white', linewidths=1)
        fig3.colorbar(im3, ax=axes3[1, 2])
        axes3[1, 2].set_xlabel('Roll [deg]')
        axes3[1, 2].set_ylabel('Pitch [deg]')
        axes3[1, 2].set_title('err_value = ux² + uy²\n(used in potential calculation)')
        axes3[1, 2].axhline(y=0, color='w', linestyle='-', linewidth=0.5)
        axes3[1, 2].axvline(x=0, color='w', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        # Print detailed explanation
        print("\n" + "="*80)
        print("upright_vector = quat_rotate_inverse(root_quat_w, [0, 0, 1])")
        print("="*80)
        print("\nPhysical interpretation:")
        print("- upright_vector represents the world Z-axis (up) in robot's local frame")
        print("- When robot is upright: upright_vector = [0, 0, 1]")
        print("- ux (upright_vector[0]): Forward/backward tilt component")
        print("  - Positive ux: Robot tilted backward")
        print("  - Negative ux: Robot tilted forward")
        print("  - ux ≈ -sin(pitch) when roll ≈ 0")
        print("- uy (upright_vector[1]): Left/right tilt component")
        print("  - Positive uy: Robot tilted to the right")
        print("  - Negative uy: Robot tilted to the left")
        print("  - uy ≈ sin(roll) when pitch ≈ 0")
        print("- uz (upright_vector[2]): Vertical component")
        print("  - uz = 1 when perfectly upright")
        print("  - uz decreases as robot tilts")
        print("\nerr_value = ux² + uy² represents total tilt error")
        print("potential = exp(-err_value / sigma) rewards being upright")

    # Print sample values to console
    def print_sample_values():
        """Print values at representative angles to console"""
        print("\n" + "="*80)
        print("orientation_potential Reward Values (sigma=0.5, enable_exp_func=True)")
        print("="*80)
        print(f"{'Roll [deg]':>12} {'Pitch [deg]':>12} {'ux':>10} {'uy':>10} {'uz':>10} {'err_value':>12} {'potential':>12}")
        print("-"*80)

        test_angles = [0, 5, 10, 15, 20, 30, 45, 60, 90]

        for roll in [0, 15, 30]:
            for pitch in test_angles:
                result = compute_orientation_potential(np.deg2rad(roll), np.deg2rad(pitch), 0.5, True)
                print(f"{roll:>12} {pitch:>12} {result['ux']:>10.4f} {result['uy']:>10.4f} {result['uz']:>10.4f} {result['err_value']:>12.4f} {result['potential']:>12.4f}")

        print("\n--- Important angle values ---")
        important_cases = [
            (0, 0, "Upright"),
            (10, 0, "Roll 10 deg"),
            (0, 10, "Pitch 10 deg"),
            (10, 10, "Roll/Pitch 10 deg"),
            (20, 0, "Roll 20 deg"),
            (0, 20, "Pitch 20 deg"),
            (30, 0, "Roll 30 deg"),
            (0, 30, "Pitch 30 deg"),
        ]

        print(f"\n{'State':>20} {'Roll':>8} {'Pitch':>8} {'potential':>12}")
        print("-"*50)
        for roll, pitch, desc in important_cases:
            result = compute_orientation_potential(np.deg2rad(roll), np.deg2rad(pitch), 0.5, True)
            print(f"{desc:>20} {roll:>8} deg {pitch:>8} deg {result['potential']:>12.4f}")

    # Print sample values
    print_sample_values()

    # Show slice plots
    create_slice_plot()

    # Show raw upright_vector values
    create_raw_upright_vector_plot()

    plt.show()


if __name__ == "__main__":
    create_visualization()
