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

# Controls how quickly arrows fade (Number of frames)
FADE_FRAMES = 10

# Controls arrow length (LOWER value = LONGER arrows)
# 1.0 means a unit vector (length 1) is drawn as 1.0 unit long in the plot.
ARROW_SCALE = 2.5


def visualize_simulation(filepath: str):
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

        # Load Floor config
        floor_active = config["floor"]["active"][()]
        floor_height = config["floor"]["height"][()] if floor_active else 0.0

        # --- Load Steps ---
        step_keys = sorted([k for k in f.keys() if k.startswith("step_")])
        num_steps = len(step_keys)

        times = []
        translations = []

        contact_indices_log = []
        contact_Js_log = []

        for key in step_keys:
            step_grp = f[key]
            times.append(step_grp["time"][()])

            s_data = step_grp["shapes_data"]
            translations.append(s_data["translation"][:])

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
    print(f"Loaded {num_steps} steps.")

    # --- Pre-calculate Bounds ---
    all_x = []
    all_y = []

    for s_idx in range(num_shapes):
        r = radii[s_idx]
        xs = translations[:, s_idx, 0]
        ys = translations[:, s_idx, 1]
        all_x.extend([np.min(xs) - r, np.max(xs) + r])
        all_y.extend([np.min(ys) - r, np.max(ys) + r])

    if floor_active:
        all_y.extend([floor_height, floor_height - 1.0])

    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)

    pad_x = (max_x - min_x) * 0.1 + 0.5
    pad_y = (max_y - min_y) * 0.1 + 0.5

    # --- Setup Figure ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
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
        r = radii[i]
        if s_type == SHAPE_TYPE_POINT:
            r = 0.05
            patch = patches.Circle((pos[0], pos[1]), r, fc="red", ec="black", zorder=5)
        else:
            patch = patches.Circle(
                (pos[0], pos[1]), r, fc="cornflowerblue", ec="navy", alpha=0.9, zorder=5
            )

        ax.add_patch(patch)
        graphic_elements.append(patch)

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
        idx = int(slider.val)
        if sim_state.is_playing:
            idx = (idx + 1) % num_steps
            slider.set_val(idx)

        # 1. Update Shape Positions
        current_trans = translations[idx]
        for i, patch in enumerate(graphic_elements):
            patch.center = (current_trans[i][0], current_trans[i][1])

        # 2. Update Jacobians (Recreate Quiver with Fading)
        if sim_state.quiver is not None:
            try:
                sim_state.quiver.remove()
            except ValueError:
                pass
            sim_state.quiver = None

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

            keys_to_check = list(sim_state.persistent_contacts.keys())

            for key in keys_to_check:
                data = sim_state.persistent_contacts[key]

                if key not in active_keys:
                    data["life"] -= 1

                if data["life"] <= 0:
                    del sim_state.persistent_contacts[key]
                    continue

                alpha = data["life"] / FADE_FRAMES
                color = (0.2, 1.0, 0.2, alpha)  # Lime green

                idx1, idx2 = key
                J1 = data["J1"]
                pos1 = current_trans[idx1]

                arrow_x.append(pos1[0])
                arrow_y.append(pos1[1])
                arrow_u.append(J1[0])
                arrow_v.append(J1[1])
                arrow_colors.append(color)

                if idx2 != -1:
                    J2 = data["J2"]
                    pos2 = current_trans[idx2]
                    arrow_x.append(pos2[0])
                    arrow_y.append(pos2[1])
                    arrow_u.append(J2[0])
                    arrow_v.append(J2[1])
                    arrow_colors.append(color)

            if len(arrow_x) > 0:
                # scale=1.0 makes vectors length 1.0 in data units
                # width=0.003 ensures they stay thin and sharp
                sim_state.quiver = ax.quiver(
                    arrow_x,
                    arrow_y,
                    arrow_u,
                    arrow_v,
                    color=arrow_colors,
                    scale=ARROW_SCALE,
                    scale_units="xy",
                    angles="xy",
                    width=0.002,  # <-- Key change for thinness
                    headwidth=4,
                    headlength=5,  # Adjust head proportions
                    zorder=10,
                )

        time_text.set_text(f"Time: {times[idx]:.2f}s (Frame {idx})")

        ret = graphic_elements + [time_text]
        if sim_state.quiver:
            ret.append(sim_state.quiver)
        return ret

    # --- Widgets ---
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(
        ax=ax_slider, label="Frame", valmin=0, valmax=num_steps - 1, valinit=0, valstep=1
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
        fig, update, frames=None, interval=20, blit=False, cache_frame_data=False
    )
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        visualize_simulation(sys.argv[1])
    else:
        print("Usage: python visualization.py <path_to_log.h5>")
