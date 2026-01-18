import argparse
import math

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches

# --- Constants ---
SHAPE_TYPE_FLOOR = -1
SHAPE_TYPE_CIRCLE = 0
SHAPE_TYPE_POINT = 1
SHAPE_TYPE_RECTANGLE = 2

# Controls how quickly arrows fade (Number of frames)
FADE_FRAMES = 10

# Controls arrow length (LOWER value = LONGER arrows)
ARROW_SCALE = 2.5
ROTATION_SCALE = 1.0


def visualize_simulation(filepath: str, save_path: str = None):
    """
    Visualizes the 2D physics simulation from an HDF5 log file.
    """
    print(f"Loading simulation data from {filepath}...")

    with h5py.File(filepath, "r") as f:
        # --- Load Configuration ---
        if "init_config" not in f:
            raise ValueError("Invalid log file: 'init_config' group missing.")

        config = f["init_config"]
        num_shapes = config["num_shapes"][()]
        shape_types = config["shape_types"][:]

        # Load Radii
        if "radii" in config:
            radii = config["radii"][:]
        else:
            print("Warning: 'radii' not found in log. Using defaults.")
            radii = np.ones(num_shapes) * 0.5

        # Load Sides (for Rectangles)
        if "sides" in config:
            sides = config["sides"][:]
        else:
            sides = np.zeros((num_shapes, 2))

        # Load Floor config
        floor_active = config["floor"]["active"][()]
        floor_height = config["floor"]["height"][()] if floor_active else 0.0

        # --- Load Steps ---
        step_keys = sorted([k for k in f.keys() if k.startswith("step_")])
        num_steps = len(step_keys)

        times = []
        translations = []
        rotations = []

        contact_indices_log = []
        contact_Js_log = []

        for key in step_keys:
            step_grp = f[key]
            times.append(step_grp["time"][()])

            s_data = step_grp["shapes_data"]
            translations.append(s_data["translation"][:])
            rotations.append(s_data["rotation"][:])

            c_data = step_grp["contacts_data"]
            count = c_data["count"][()]

            if count > 0:
                contact_indices_log.append(c_data["indices"][:])
                contact_Js_log.append(c_data["Js"][:])
            else:
                contact_indices_log.append(np.empty((0, 2), dtype=int))
                contact_Js_log.append(np.empty((0, 2, 3)))

    times = np.array(times)
    translations = np.array(translations)
    rotations = np.array(rotations)
    print(f"Loaded {num_steps} steps.")

    # --- Pre-calculate Bounds ---
    all_x = []
    all_y = []

    for s_idx in range(num_shapes):
        xs = translations[:, s_idx, 0]
        ys = translations[:, s_idx, 1]

        if shape_types[s_idx] == SHAPE_TYPE_RECTANGLE:
            w, h = sides[s_idx]
            extent = math.sqrt(w**2 + h**2) / 2.0
        else:
            extent = radii[s_idx]

        all_x.extend([np.min(xs) - extent, np.max(xs) + extent])
        all_y.extend([np.min(ys) - extent, np.max(ys) + extent])

    if floor_active:
        all_y.extend([floor_height, floor_height - 1.0])

    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)

    pad_x = (max_x - min_x) * 0.1 + 0.5
    pad_y = (max_y - min_y) * 0.1 + 0.5

    # --- Setup Figure ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim(np.clip(min_x - pad_x, -100, 100), np.clip(max_x + pad_x, -100, 100))
    ax.set_ylim(np.clip(min_y - pad_y, -100, 100), np.clip(max_y + pad_y, -100, 100))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title("2D Physics Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # --- Initialize Graphics ---
    graphic_elements = []

    if floor_active:
        ax.axhline(y=floor_height, color="black", linewidth=2)
        rect = patches.Rectangle(
            (min_x - 100, floor_height - 100),
            (max_x - min_x) + 200,
            100,
            linewidth=0,
            facecolor="#e0e0e0",
            zorder=0,
        )
        ax.add_patch(rect)

    for i, s_type in enumerate(shape_types):
        pos = translations[0][i]

        if s_type == SHAPE_TYPE_POINT:
            r = 0.05
            patch = patches.Circle((pos[0], pos[1]), r, fc="red", ec="black", zorder=5)
        elif s_type == SHAPE_TYPE_RECTANGLE:
            w, h = sides[i]
            patch = patches.Rectangle(
                (0, 0), w, h, angle=0.0, fc="orange", ec="darkorange", alpha=0.9, zorder=5
            )
        else:
            r = radii[i]
            patch = patches.Circle(
                (pos[0], pos[1]), r, fc="cornflowerblue", ec="navy", alpha=0.9, zorder=5
            )

        ax.add_patch(patch)
        graphic_elements.append(patch)

    # --- Torque/Rotation Visualization Markers ---
    scat_ccw = ax.scatter(
        [], [], s=ROTATION_SCALE * 120, marker=r"$\curvearrowleft$", color="lime", zorder=11
    )
    scat_cw = ax.scatter(
        [], [], s=ROTATION_SCALE * 120, marker=r"$\curvearrowright$", color="lime", zorder=11
    )

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # --- State Management ---
    class SimState:
        def __init__(self):
            self.is_playing = False
            self.show_jacobians = True
            self.quiver = None
            self.persistent_contacts = {}

        def toggle_play(self, event=None):
            self.is_playing = not self.is_playing
            btn_play.label.set_text("Pause" if self.is_playing else "Play")

        def toggle_jacobians(self, event=None):
            self.show_jacobians = not self.show_jacobians
            btn_jacobian.label.set_text(f"Jacobians: {'ON' if self.show_jacobians else 'OFF'}")

    sim_state = SimState()

    # --- Update Function ---
    def update(frame):
        if save_path:
            idx = frame
        else:
            idx = int(slider.val)
            if sim_state.is_playing:
                idx = (idx + 1) % num_steps
                slider.set_val(idx)

        # 1. Update Shape Positions
        current_trans = translations[idx]
        current_rots = rotations[idx]

        for i, patch in enumerate(graphic_elements):
            if isinstance(patch, patches.Rectangle):
                cx, cy = current_trans[i]
                theta = current_rots[i]
                w, h = sides[i]
                dx, dy = -w / 2.0, -h / 2.0
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rot_x = dx * cos_t - dy * sin_t
                rot_y = dx * sin_t + dy * cos_t
                patch.set_xy((cx + rot_x, cy + rot_y))
                patch.angle = np.degrees(theta)
            else:
                patch.center = (current_trans[i][0], current_trans[i][1])

        # 2. Update Jacobians (Linear Quiver + Angular Scatter)
        if sim_state.quiver is not None:
            try:
                sim_state.quiver.remove()
            except ValueError:
                pass
            sim_state.quiver = None

        # Reset scatter plots
        scat_ccw.set_offsets(np.empty((0, 2)))
        scat_cw.set_offsets(np.empty((0, 2)))

        if sim_state.show_jacobians:
            curr_indices = contact_indices_log[idx]
            curr_Js = contact_Js_log[idx]

            active_keys = set()

            if len(curr_indices) > 0:
                for k, (idx_pair, j_pair) in enumerate(zip(curr_indices, curr_Js)):
                    key = tuple(idx_pair)
                    active_keys.add(key)
                    sim_state.persistent_contacts[key] = {
                        "life": FADE_FRAMES,
                        "J1": j_pair[0],
                        "J2": j_pair[1],
                    }

            arrow_x, arrow_y, arrow_u, arrow_v, arrow_colors = [], [], [], [], []
            ccw_points, ccw_colors = [], []
            cw_points, cw_colors = [], []

            keys_to_check = list(sim_state.persistent_contacts.keys())

            for key in keys_to_check:
                data = sim_state.persistent_contacts[key]

                if key not in active_keys:
                    data["life"] -= 1

                if data["life"] <= 0:
                    del sim_state.persistent_contacts[key]
                    continue

                alpha = data["life"] / FADE_FRAMES
                # Color for both arrows and markers
                base_color = (0.2, 1.0, 0.2, alpha)

                idx1, idx2 = key
                J1 = data["J1"]
                pos1 = current_trans[idx1]

                # Linear Components
                arrow_x.append(pos1[0])
                arrow_y.append(pos1[1])
                arrow_u.append(J1[0])
                arrow_v.append(J1[1])
                arrow_colors.append(base_color)

                # Angular Component (J1)
                # Threshold to avoid drawing noise
                if abs(J1[2]) > 0.05:
                    if J1[2] > 0:
                        ccw_points.append([pos1[0], pos1[1]])
                        ccw_colors.append(base_color)
                    else:
                        cw_points.append([pos1[0], pos1[1]])
                        cw_colors.append(base_color)

                if idx2 != -1:
                    J2 = data["J2"]
                    pos2 = current_trans[idx2]

                    # Linear
                    arrow_x.append(pos2[0])
                    arrow_y.append(pos2[1])
                    arrow_u.append(J2[0])
                    arrow_v.append(J2[1])
                    arrow_colors.append(base_color)

                    # Angular Component (J2)
                    if abs(J2[2]) > 0.05:
                        if J2[2] > 0:
                            ccw_points.append([pos2[0], pos2[1]])
                            ccw_colors.append(base_color)
                        else:
                            cw_points.append([pos2[0], pos2[1]])
                            cw_colors.append(base_color)

            # Update Linear Quiver
            if len(arrow_x) > 0:
                sim_state.quiver = ax.quiver(
                    arrow_x,
                    arrow_y,
                    arrow_u,
                    arrow_v,
                    color=arrow_colors,
                    scale=ARROW_SCALE,
                    scale_units="xy",
                    angles="xy",
                    width=0.002,
                    headwidth=4,
                    headlength=5,
                    zorder=10,
                )

            # Update Angular Scatters
            if ccw_points:
                scat_ccw.set_offsets(ccw_points)
                scat_ccw.set_color(ccw_colors)
            if cw_points:
                scat_cw.set_offsets(cw_points)
                scat_cw.set_color(cw_colors)

        time_text.set_text(f"Time: {times[idx]:.2f}s (Frame {idx})")

        ret = graphic_elements + [time_text, scat_ccw, scat_cw]
        if sim_state.quiver:
            ret.append(sim_state.quiver)
        return ret

    # --- Widgets ---
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(
        ax=ax_slider,
        label="Frame",
        valmin=0,
        valmax=num_steps - 1,
        valinit=0,
        valstep=1,
    )

    ax_play = plt.axes([0.05, 0.1, 0.1, 0.04])
    btn_play = Button(ax_play, "Play", hovercolor="0.975")

    ax_jac = plt.axes([0.05, 0.04, 0.2, 0.04])
    btn_jacobian = Button(ax_jac, "Jacobians: ON", hovercolor="0.975")

    # --- Callbacks ---
    slider.on_changed(lambda val: fig.canvas.draw_idle())
    btn_play.on_clicked(sim_state.toggle_play)
    btn_jacobian.on_clicked(sim_state.toggle_jacobians)

    anim = animation.FuncAnimation(
        fig, update, frames=num_steps, interval=20, blit=False, cache_frame_data=False
    )
    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".gif"):
            writer = "pillow"
        else:
            writer = "ffmpeg"
        ax_slider.set_visible(False)
        ax_play.set_visible(False)
        ax_jac.set_visible(False)
        anim.save(save_path, writer=writer, fps=30, dpi=300)
        print("Save complete.")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HDF5 Pass")
    parser.add_argument("--hdf_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    args = parser.parse_args()
    visualize_simulation(args.hdf_path, args.save_path)
