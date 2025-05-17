import tkinter as tk
from tkinter import filedialog
import matplotlib

matplotlib.use("TkAgg")  # Essential for tkinter backend

import matplotlib.pyplot as plt  # Import pyplot after setting rcParams

plt.rcParams["interactive"] = (
    False  # Crucial: Disable Matplotlib's interactive mode behavior
)
plt.ioff()  # Ensure pyplot's global event loop integration is off

from matplotlib.widgets import RadioButtons, Slider, Button
import numpy as np
import os
from scipy.spatial.transform import Rotation


# --- Helper Transformation Functions ---
def translation_matrix(dx, dy, dz):
    return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])


def euler_to_rotation_matrix(roll_rad, pitch_rad, yaw_rad, sequence="zyx"):
    r = Rotation.from_euler(sequence, [yaw_rad, pitch_rad, roll_rad], degrees=False)
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    return mat


def apply_transform_to_points(points_3d, matrix_4x4):
    if points_3d.shape[0] == 0:
        return np.array([]).reshape(0, 3)
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    transformed_points_homogeneous = (matrix_4x4 @ points_homogeneous.T).T
    return transformed_points_homogeneous[:, :3]


# --- Trajectory Class ---
class Trajectory:
    _id_counter = 0

    def __init__(self, original_data, file_basename):
        self.id = Trajectory._id_counter
        Trajectory._id_counter += 1
        self.file_basename = file_basename

        self.original_full_data = np.array(original_data)
        self.original_timestamps = self.original_full_data[:, 0]
        self.original_positions = self.original_full_data[:, 1:4]

        if self.original_positions.shape[0] == 0:
            raise ValueError("Trajectory has no position data.")

        self.original_start_point = self.original_positions[0, :].copy()
        self.original_end_point = self.original_positions[-1, :].copy()

        self.base_transform = np.eye(4)
        self.local_translation_vector = np.array([0.0, 0.0, 0.0])
        self.local_rotation_angles_rad = np.array([0.0, 0.0, 0.0])

        self.current_positions = self.original_positions.copy()
        self.current_start_point = self.original_start_point.copy()
        self.current_end_point = self.original_end_point.copy()

        self.color = plt.cm.viridis(self.id / 10 if self.id < 10 else np.random.rand())

    def _compute_local_transform(self):
        T_to_origin = translation_matrix(
            -self.original_start_point[0],
            -self.original_start_point[1],
            -self.original_start_point[2],
        )
        R_local = euler_to_rotation_matrix(
            self.local_rotation_angles_rad[0],
            self.local_rotation_angles_rad[1],
            self.local_rotation_angles_rad[2],
        )
        T_from_origin = translation_matrix(
            self.original_start_point[0],
            self.original_start_point[1],
            self.original_start_point[2],
        )
        T_local_translate = translation_matrix(
            self.local_translation_vector[0],
            self.local_translation_vector[1],
            self.local_translation_vector[2],
        )
        local_transform_matrix = (
            T_local_translate @ T_from_origin @ R_local @ T_to_origin
        )
        return local_transform_matrix

    def update_current_positions_and_endpoints(self):
        local_T = self._compute_local_transform()
        final_T = self.base_transform @ local_T
        self.current_positions = apply_transform_to_points(
            self.original_positions, final_T
        )
        if self.current_positions.shape[0] > 0:
            self.current_start_point = self.current_positions[0, :]
            self.current_end_point = self.current_positions[-1, :]
        else:
            self.current_start_point = np.array([0, 0, 0])
            self.current_end_point = np.array([0, 0, 0])


# --- Global variables ---
loaded_trajectories = []
selected_trajectory_idx = -1
_is_updating_plot_flag = False

current_min_time_rel = 0.0
current_max_time_rel = 1.0
active_traj_min_time_abs = 0.0
active_traj_max_time_abs = 1.0

fig = None
ax_3d = None
radio_traj_selector = None
slider_dx, slider_dy, slider_dz = None, None, None
slider_roll, slider_pitch, slider_yaw = None, None, None
button_reset_transform = None
button_load = None
button_save_interval = None
slider_interval_start = None
slider_interval_end = None
text_interval_start = None
text_interval_end = None

x_lims_view, y_lims_view, z_lims_view = None, None, None
elev_view, azim_view = None, None


def load_new_trajectory():
    global loaded_trajectories, selected_trajectory_idx, radio_traj_selector
    global active_traj_min_time_abs, active_traj_max_time_abs
    global current_min_time_rel, current_max_time_rel
    global slider_interval_start, slider_interval_end
    global fig

    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select gt.txt trajectory file",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
    )
    root.destroy()

    if not filepath:
        if fig:
            fig.suptitle("File selection cancelled.", color="orange")
            if fig.canvas:
                fig.canvas.draw_idle()
        return

    file_basename = os.path.basename(filepath)
    if fig:
        fig.suptitle(f"Loading: {file_basename}...", color="blue")
        if fig.canvas:
            fig.canvas.draw_idle()

    try:
        data_list = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 8:
                    try:
                        data_list.append([float(p) for p in parts])
                    except ValueError:
                        print(f"Skipping malformed line: {line}")
                else:
                    print(f"Skipping line with incorrect number of columns: {line}")

        if not data_list:
            raise ValueError("No valid data points found.")
        new_traj = Trajectory(data_list, file_basename)

        if loaded_trajectories:
            prev_traj = loaded_trajectories[-1]
            translation_to_connect = (
                prev_traj.current_end_point - new_traj.original_start_point
            )
            new_traj.base_transform = translation_matrix(
                translation_to_connect[0],
                translation_to_connect[1],
                translation_to_connect[2],
            )
        else:
            new_traj.base_transform = np.eye(4)

        new_traj.update_current_positions_and_endpoints()
        loaded_trajectories.append(new_traj)
        selected_trajectory_idx = len(loaded_trajectories) - 1

        update_trajectory_selector_widget()
        select_trajectory(selected_trajectory_idx)

        if fig:
            fig.suptitle(
                f"Loaded & Selected: {new_traj.file_basename} ({len(new_traj.original_timestamps)} pts)",
                color="green",
            )
    except Exception as e:
        if fig:
            fig.suptitle(f"Error loading {file_basename}: {e}", color="red")
        print(f"Error loading file: {e}")
        if fig and fig.canvas:
            fig.canvas.draw_idle()

    update_plot(reset_view=(len(loaded_trajectories) == 1))


def update_trajectory_selector_widget():
    global radio_traj_selector
    if not radio_traj_selector:
        return
    ax_radio = radio_traj_selector.ax
    ax_radio.clear()
    if not loaded_trajectories:
        radio_traj_selector = RadioButtons(ax_radio, ("No trajectories loaded",))
        radio_traj_selector.set_active(0)
        if hasattr(radio_traj_selector, "circles"):
            for circle in radio_traj_selector.circles:
                circle.set_visible(False)
        return

    labels = [
        f"Traj {i}: {traj.file_basename[:20]}"
        for i, traj in enumerate(loaded_trajectories)
    ]
    radio_traj_selector = RadioButtons(ax_radio, labels)
    radio_traj_selector.on_clicked(
        lambda label: select_trajectory_by_label(label, labels)
    )
    if 0 <= selected_trajectory_idx < len(loaded_trajectories):
        radio_traj_selector.set_active(selected_trajectory_idx)
    for i, label_widget in enumerate(radio_traj_selector.labels):
        if i < len(loaded_trajectories):
            label_widget.set_color(loaded_trajectories[i].color)


def select_trajectory_by_label(label_str, all_labels):
    try:
        idx = all_labels.index(label_str)
        select_trajectory(idx)
    except ValueError:
        print(f"Error: Could not find trajectory for label '{label_str}'")


def select_trajectory(idx):
    global selected_trajectory_idx, active_traj_min_time_abs, active_traj_max_time_abs
    global current_min_time_rel, current_max_time_rel
    global slider_dx, slider_dy, slider_dz, slider_roll, slider_pitch, slider_yaw
    global slider_interval_start, slider_interval_end, text_interval_start, text_interval_end

    if not (0 <= idx < len(loaded_trajectories)):
        selected_trajectory_idx = -1
        if slider_dx:  # Check if UI elements are initialized
            for s_widget in [
                slider_dx,
                slider_dy,
                slider_dz,
                slider_roll,
                slider_pitch,
                slider_yaw,
                button_reset_transform,
                button_save_interval,
                slider_interval_start,
                slider_interval_end,
            ]:
                if s_widget and hasattr(
                    s_widget, "ax"
                ):  # Ensure widget and its axis exist
                    s_widget.ax.set_visible(False)
                    if hasattr(s_widget, "valtext"):  # For sliders, hide valtext too
                        s_widget.valtext.set_visible(False)
            if text_interval_start:
                text_interval_start.set_text("Start Time: N/A")
            if text_interval_end:
                text_interval_end.set_text("End Time: N/A")
        if fig:
            fig.suptitle("No trajectory selected.", color="black")
        update_plot()
        return

    selected_trajectory_idx = idx
    traj = loaded_trajectories[idx]
    if fig:
        fig.suptitle(f"Selected: Traj {idx} - {traj.file_basename}", color=traj.color)

    _programmatic_update = (
        True  # Flag to prevent slider callbacks from re-triggering logic
    )

    slider_dx.set_val(traj.local_translation_vector[0])
    slider_dy.set_val(traj.local_translation_vector[1])
    slider_dz.set_val(traj.local_translation_vector[2])
    slider_roll.set_val(np.degrees(traj.local_rotation_angles_rad[0]))
    slider_pitch.set_val(np.degrees(traj.local_rotation_angles_rad[1]))
    slider_yaw.set_val(np.degrees(traj.local_rotation_angles_rad[2]))

    for s_widget in [
        slider_dx,
        slider_dy,
        slider_dz,
        slider_roll,
        slider_pitch,
        slider_yaw,
        button_reset_transform,
        button_save_interval,
        slider_interval_start,
        slider_interval_end,
    ]:
        if s_widget and hasattr(s_widget, "ax"):
            s_widget.ax.set_visible(True)
            if hasattr(s_widget, "valtext"):
                s_widget.valtext.set_visible(True)

    active_traj_min_time_abs = np.min(traj.original_timestamps)
    active_traj_max_time_abs = np.max(traj.original_timestamps)
    if active_traj_max_time_abs <= active_traj_min_time_abs:
        active_traj_max_time_abs = active_traj_min_time_abs + 1.0

    current_min_time_rel = active_traj_min_time_abs
    current_max_time_rel = active_traj_max_time_abs

    slider_interval_start.ax.set_xlim(
        active_traj_min_time_abs, active_traj_max_time_abs
    )
    slider_interval_start.valmin = active_traj_min_time_abs
    slider_interval_start.valmax = active_traj_max_time_abs
    slider_interval_start.set_val(current_min_time_rel)

    slider_interval_end.ax.set_xlim(active_traj_min_time_abs, active_traj_max_time_abs)
    slider_interval_end.valmin = active_traj_min_time_abs
    slider_interval_end.valmax = active_traj_max_time_abs
    slider_interval_end.set_val(current_max_time_rel)

    _programmatic_update = False

    text_interval_start.set_text(f"Start: {current_min_time_rel:.3f}s")
    text_interval_end.set_text(f"End: {current_max_time_rel:.3f}s")

    if radio_traj_selector and radio_traj_selector.labels:
        radio_traj_selector.set_active(idx)
    update_plot()


_programmatic_update = False  # Global flag for slider updates


def on_transform_slider_change(val=None):
    global _programmatic_update
    if (
        _programmatic_update
        or selected_trajectory_idx == -1
        or not (0 <= selected_trajectory_idx < len(loaded_trajectories))
    ):
        return
    traj = loaded_trajectories[selected_trajectory_idx]
    traj.local_translation_vector = np.array(
        [slider_dx.val, slider_dy.val, slider_dz.val]
    )
    traj.local_rotation_angles_rad = np.radians(
        np.array([slider_roll.val, slider_pitch.val, slider_yaw.val])
    )
    traj.update_current_positions_and_endpoints()
    propagate_transforms(selected_trajectory_idx)
    update_plot()


def reset_selected_trajectory_transform(event):
    global _programmatic_update
    if selected_trajectory_idx == -1 or not (
        0 <= selected_trajectory_idx < len(loaded_trajectories)
    ):
        return
    traj = loaded_trajectories[selected_trajectory_idx]
    traj.local_translation_vector = np.array([0.0, 0.0, 0.0])
    traj.local_rotation_angles_rad = np.array([0.0, 0.0, 0.0])

    _programmatic_update = True
    slider_dx.set_val(0.0)
    slider_dy.set_val(0.0)
    slider_dz.set_val(0.0)
    slider_roll.set_val(0.0)
    slider_pitch.set_val(0.0)
    slider_yaw.set_val(0.0)
    _programmatic_update = False

    traj.update_current_positions_and_endpoints()
    propagate_transforms(selected_trajectory_idx)
    update_plot()


def propagate_transforms(changed_idx):
    for i in range(changed_idx + 1, len(loaded_trajectories)):
        prev_traj = loaded_trajectories[i - 1]
        curr_traj = loaded_trajectories[i]
        translation_to_reconnect = (
            prev_traj.current_end_point - curr_traj.original_start_point
        )
        curr_traj.base_transform = translation_matrix(
            translation_to_reconnect[0],
            translation_to_reconnect[1],
            translation_to_reconnect[2],
        )
        curr_traj.update_current_positions_and_endpoints()


def on_interval_slider_change(val=None):
    global current_min_time_rel, current_max_time_rel, _programmatic_update
    if (
        _programmatic_update
        or selected_trajectory_idx == -1
        or not slider_interval_start
    ):
        return

    new_start = slider_interval_start.val
    new_end = slider_interval_end.val
    time_range = active_traj_max_time_abs - active_traj_min_time_abs
    min_sep = 0.001 * time_range if time_range > 0 else 0.001

    _programmatic_update_temp = _programmatic_update  # Store current state
    _programmatic_update = True  # Prevent re-entrancy during set_val

    if new_start >= new_end:
        # Determine which slider was moved by checking 'val' against current slider values
        # This logic assumes 'val' is the new value of the slider that triggered the event
        if slider_interval_start.val == val:  # Start slider moved
            new_end = min(new_start + min_sep, active_traj_max_time_abs)
            slider_interval_end.set_val(new_end)
        elif slider_interval_end.val == val:  # End slider moved
            new_start = max(new_end - min_sep, active_traj_min_time_abs)
            slider_interval_start.set_val(new_start)
        # If val doesn't match either, it might be an initial call or complex state,
        # default to adjusting 'end' based on 'start'
        else:
            new_end = min(new_start + min_sep, active_traj_max_time_abs)
            slider_interval_end.set_val(new_end)

    _programmatic_update = _programmatic_update_temp  # Restore flag

    current_min_time_rel = new_start
    current_max_time_rel = new_end
    text_interval_start.set_text(f"Start: {current_min_time_rel:.3f}s")
    text_interval_end.set_text(f"End: {current_max_time_rel:.3f}s")

    if (
        not _programmatic_update
    ):  # Only update plot if not part of a programmatic cascade
        update_plot()


def update_plot(val=None, reset_view=False):
    global x_lims_view, y_lims_view, z_lims_view, elev_view, azim_view
    global fig, ax_3d, _is_updating_plot_flag

    if _is_updating_plot_flag:
        return
    _is_updating_plot_flag = True

    try:
        if not ax_3d or not fig or not fig.canvas:
            return

        current_x_lim, current_y_lim, current_z_lim = None, None, None
        current_elev, current_azim = None, None

        if not reset_view:
            try:  # Attempt to get current view settings
                if ax_3d.has_data():  # Check if axes has data/been drawn
                    current_x_lim = ax_3d.get_xlim()
                    current_y_lim = ax_3d.get_ylim()
                    current_z_lim = ax_3d.get_zlim()
                    current_elev = ax_3d.elev
                    current_azim = ax_3d.azim
            except Exception:
                pass  # Silently ignore if view cannot be retrieved

        ax_3d.clear()

        if not loaded_trajectories:
            ax_3d.set_title("Load a trajectory file")
        else:
            ax_3d.set_title("Multi-Trajectory Editor")

        all_points_for_lims = []
        for i, traj in enumerate(loaded_trajectories):
            positions = traj.current_positions
            if positions.shape[0] == 0:
                continue
            all_points_for_lims.append(positions)
            line_style, line_width, marker_size, alpha = ".-", 1.5, 3, 0.7
            label_prefix = f"Traj {i}"
            if i == selected_trajectory_idx:
                line_width, marker_size, alpha = 2.5, 6, 1.0
                epsilon = 1e-9
                interval_indices = np.where(
                    (traj.original_timestamps >= current_min_time_rel - epsilon)
                    & (traj.original_timestamps <= current_max_time_rel + epsilon)
                )[0]
                if len(interval_indices) > 0:
                    selected_interval_original_pos = traj.original_positions[
                        interval_indices, :
                    ]
                    local_T = traj._compute_local_transform()
                    final_T = traj.base_transform @ local_T
                    selected_interval_current_pos = apply_transform_to_points(
                        selected_interval_original_pos, final_T
                    )
                    if selected_interval_current_pos.shape[0] > 0:
                        ax_3d.plot(
                            selected_interval_current_pos[:, 0],
                            selected_interval_current_pos[:, 1],
                            selected_interval_current_pos[:, 2],
                            "-",
                            color="magenta",
                            linewidth=3,
                            alpha=0.8,
                            label=f"{label_prefix} Interval",
                            zorder=10,
                        )
                        ax_3d.plot(
                            [selected_interval_current_pos[0, 0]],
                            [selected_interval_current_pos[0, 1]],
                            [selected_interval_current_pos[0, 2]],
                            "o",
                            color="cyan",
                            markersize=8,
                            label="Interval Start",
                            zorder=11,
                        )
                        ax_3d.plot(
                            [selected_interval_current_pos[-1, 0]],
                            [selected_interval_current_pos[-1, 1]],
                            [selected_interval_current_pos[-1, 2]],
                            "o",
                            color="lime",
                            markersize=8,
                            label="Interval End",
                            zorder=11,
                        )
            ax_3d.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                line_style,
                color=traj.color,
                linewidth=line_width,
                markersize=marker_size,
                alpha=alpha,
                label=f"{label_prefix}: {traj.file_basename[:15]}",
            )
            if i == selected_trajectory_idx:
                ax_3d.plot(
                    [traj.current_start_point[0]],
                    [traj.current_start_point[1]],
                    [traj.current_start_point[2]],
                    "o",
                    color="gold",
                    markersize=10,
                    label="Sel. Traj Start",
                    markeredgecolor="black",
                    zorder=12,
                )
                ax_3d.plot(
                    [traj.current_end_point[0]],
                    [traj.current_end_point[1]],
                    [traj.current_end_point[2]],
                    "s",
                    color="orangered",
                    markersize=10,
                    label="Sel. Traj End",
                    markeredgecolor="black",
                    zorder=12,
                )
            else:
                ax_3d.plot(
                    [traj.current_start_point[0]],
                    [traj.current_start_point[1]],
                    [traj.current_start_point[2]],
                    "o",
                    color=traj.color,
                    markersize=4,
                    alpha=0.5,
                )

        ax_3d.set_xlabel("X Position")
        ax_3d.set_ylabel("Y Position")
        ax_3d.set_zlabel("Z Position")
        handles, labels = ax_3d.get_legend_handles_labels()
        if handles:
            ax_3d.legend(
                handles,
                labels,
                loc="upper left",
                fontsize="x-small",
                bbox_to_anchor=(1.05, 1),
            )
        ax_3d.grid(True)

        if reset_view or not all_points_for_lims:
            if all_points_for_lims:
                all_points_np = np.concatenate(all_points_for_lims, axis=0)
                if all_points_np.shape[0] > 0:
                    min_coords = np.min(all_points_np, axis=0)
                    max_coords = np.max(all_points_np, axis=0)
                    center = (min_coords + max_coords) / 2
                    ranges = np.abs(max_coords - min_coords)
                    ranges[ranges < 1e-6] = 1.0  # Avoid zero range
                    plot_radius = np.max(ranges) * 0.7
                    ax_3d.set_xlim(center[0] - plot_radius, center[0] + plot_radius)
                    ax_3d.set_ylim(center[1] - plot_radius, center[1] + plot_radius)
                    ax_3d.set_zlim(center[2] - plot_radius, center[2] + plot_radius)
            else:  # No data at all
                ax_3d.set_xlim(-1, 1)
                ax_3d.set_ylim(-1, 1)
                ax_3d.set_zlim(-1, 1)
            ax_3d.view_init(elev=20.0, azim=-60.0)  # Default view
            # Store this new default view
            x_lims_view, y_lims_view, z_lims_view = (
                ax_3d.get_xlim(),
                ax_3d.get_ylim(),
                ax_3d.get_zlim(),
            )
            elev_view, azim_view = ax_3d.elev, ax_3d.azim
        elif current_x_lim is not None:  # Restore previous view if available
            ax_3d.set_xlim(current_x_lim)
            ax_3d.set_ylim(current_y_lim)
            ax_3d.set_zlim(current_z_lim)
            if current_elev is not None and current_azim is not None:
                ax_3d.view_init(elev=current_elev, azim=current_azim)
            # And update global view storage
            x_lims_view, y_lims_view, z_lims_view = (
                current_x_lim,
                current_y_lim,
                current_z_lim,
            )
            elev_view, azim_view = current_elev, current_azim
        else:  # Fallback if no previous view and not resetting (should be rare)
            ax_3d.set_xlim(-1, 1)
            ax_3d.set_ylim(-1, 1)
            ax_3d.set_zlim(-1, 1)
            ax_3d.view_init(elev=20.0, azim=-60.0)
            x_lims_view, y_lims_view, z_lims_view = (
                ax_3d.get_xlim(),
                ax_3d.get_ylim(),
                ax_3d.get_zlim(),
            )
            elev_view, azim_view = ax_3d.elev, ax_3d.azim

        if fig and fig.canvas:
            fig.canvas.draw_idle()
    finally:
        _is_updating_plot_flag = False


def save_selected_trajectory_interval(event):
    global fig
    if selected_trajectory_idx == -1 or not (
        0 <= selected_trajectory_idx < len(loaded_trajectories)
    ):
        if fig:
            fig.suptitle("No trajectory selected to save.", color="red")
            fig.canvas.draw_idle()
        return
    traj = loaded_trajectories[selected_trajectory_idx]
    root_save = tk.Tk()
    root_save.withdraw()
    default_savename = f"{os.path.splitext(traj.file_basename)[0]}_interval_{current_min_time_rel:.2f}-{current_max_time_rel:.2f}_transformed.txt"
    save_filepath = filedialog.asksaveasfilename(
        title="Save selected trajectory interval (transformed)",
        initialfile=default_savename,
        defaultextension=".txt",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
    )
    root_save.destroy()
    if not save_filepath:
        if fig:
            fig.suptitle("Save cancelled.", color="orange")
            fig.canvas.draw_idle()
        return
    if fig:
        fig.suptitle(f"Saving to {os.path.basename(save_filepath)}...", color="blue")
        fig.canvas.draw_idle()
    try:
        epsilon = 1e-9
        indices_in_interval = np.where(
            (traj.original_timestamps >= current_min_time_rel - epsilon)
            & (traj.original_timestamps <= current_max_time_rel + epsilon)
        )[0]
        if len(indices_in_interval) == 0:
            if fig:
                fig.suptitle(
                    "No data points in selected interval to save.", color="orange"
                )
                fig.canvas.draw_idle()
            return
        interval_data_original = traj.original_full_data[indices_in_interval, :]
        interval_positions_original = interval_data_original[:, 1:4]
        interval_quaternions = interval_data_original[:, 4:8]
        interval_timestamps = interval_data_original[:, 0]
        local_T = traj._compute_local_transform()
        final_T = traj.base_transform @ local_T
        interval_positions_transformed = apply_transform_to_points(
            interval_positions_original, final_T
        )
        data_to_save = np.hstack(
            (
                interval_timestamps.reshape(-1, 1),
                interval_positions_transformed,
                interval_quaternions,
            )
        )
        with open(save_filepath, "w") as f:
            f.write(f"# Trajectory segment from original file: {traj.file_basename}\n")
            f.write(
                f"# Selected original time interval: {current_min_time_rel:.6f} to {current_max_time_rel:.6f}\n"
            )
            f.write(
                "# Applied transformations: Base Transform (connection), Local Transform (user edits)\n"
            )
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            np.savetxt(f, data_to_save, fmt="%.6f", delimiter=" ")
        if fig:
            fig.suptitle(
                f"Saved {len(data_to_save)} points to {os.path.basename(save_filepath)}",
                color="green",
            )
        print(f"Successfully saved {len(data_to_save)} points to {save_filepath}")
    except Exception as e:
        if fig:
            fig.suptitle(f"Error saving file: {e}", color="red")
        print(f"Error saving file: {e}")
    if fig and fig.canvas:
        fig.canvas.draw_idle()


def main_gui():
    global fig, ax_3d, radio_traj_selector
    global slider_dx, slider_dy, slider_dz, slider_roll, slider_pitch, slider_yaw
    global button_reset_transform, button_load, button_save_interval
    global slider_interval_start, slider_interval_end, text_interval_start, text_interval_end

    fig = plt.figure(figsize=(17, 10))  # Increased width slightly for legend
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], height_ratios=[0.1, 0.7, 0.2])
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_radio = fig.add_subplot(gs[0, 1])
    ax_radio.set_title("Trajectories", fontsize="medium")
    radio_traj_selector = RadioButtons(ax_radio, ("No trajectories loaded",))
    radio_traj_selector.set_active(0)
    if hasattr(radio_traj_selector, "circles"):
        for circle in radio_traj_selector.circles:
            circle.set_visible(False)
    ax_radio.axis("off")

    ax_transforms_panel = fig.add_subplot(gs[1, 1])
    ax_transforms_panel.set_title("Transform Selected Trajectory", fontsize="medium")
    ax_transforms_panel.axis("off")
    current_y_pos, slider_height, spacing = 0.95, 0.06, 0.01

    def make_slider(p_ax, y, lbl, vmin, vmax, vinit, vfmt="%.2f"):
        ax_s = p_ax.inset_axes([0.1, y - slider_height, 0.8, slider_height])
        return Slider(ax_s, lbl, vmin, vmax, valinit=vinit, valfmt=vfmt)

    slider_dx = make_slider(
        ax_transforms_panel, current_y_pos, "Translate X", -5.0, 5.0, 0.0
    )
    current_y_pos -= slider_height + spacing
    slider_dy = make_slider(
        ax_transforms_panel, current_y_pos, "Translate Y", -5.0, 5.0, 0.0
    )
    current_y_pos -= slider_height + spacing
    slider_dz = make_slider(
        ax_transforms_panel, current_y_pos, "Translate Z", -5.0, 5.0, 0.0
    )
    current_y_pos -= slider_height + spacing + 0.05
    slider_roll = make_slider(
        ax_transforms_panel, current_y_pos, "Roll (deg)", -180, 180, 0.0
    )
    current_y_pos -= slider_height + spacing
    slider_pitch = make_slider(
        ax_transforms_panel, current_y_pos, "Pitch (deg)", -90, 90, 0.0
    )
    current_y_pos -= slider_height + spacing
    slider_yaw = make_slider(
        ax_transforms_panel, current_y_pos, "Yaw (deg)", -180, 180, 0.0
    )
    current_y_pos -= slider_height + spacing + 0.05
    for s in [slider_dx, slider_dy, slider_dz, slider_roll, slider_pitch, slider_yaw]:
        s.on_changed(on_transform_slider_change)
        s.ax.set_visible(False)
        s.valtext.set_visible(False)

    ax_btn_reset = ax_transforms_panel.inset_axes(
        [0.3, current_y_pos - 0.08, 0.4, 0.08]
    )
    button_reset_transform = Button(
        ax_btn_reset, "Reset Transform", color="lightcoral", hovercolor="red"
    )
    button_reset_transform.on_clicked(reset_selected_trajectory_transform)
    button_reset_transform.ax.set_visible(False)

    ax_interval_file_panel = fig.add_subplot(gs[2, 1])
    ax_interval_file_panel.set_title("Interval & File Ops", fontsize="medium")
    ax_interval_file_panel.axis("off")
    current_y_pos_interval = 0.9
    ax_slider_start_int = ax_interval_file_panel.inset_axes(
        [0.1, current_y_pos_interval - slider_height, 0.8, slider_height]
    )
    slider_interval_start = Slider(
        ax_slider_start_int, "Interval Start", 0.0, 1.0, valinit=0.0, valfmt="%.3fs"
    )
    current_y_pos_interval -= slider_height + spacing
    ax_slider_end_int = ax_interval_file_panel.inset_axes(
        [0.1, current_y_pos_interval - slider_height, 0.8, slider_height]
    )
    slider_interval_end = Slider(
        ax_slider_end_int, "Interval End", 0.0, 1.0, valinit=1.0, valfmt="%.3fs"
    )
    current_y_pos_interval -= slider_height + spacing + 0.02
    text_interval_start = ax_interval_file_panel.text(
        0.5,
        current_y_pos_interval,
        "Start: N/A",
        ha="center",
        va="center",
        transform=ax_interval_file_panel.transAxes,
    )
    current_y_pos_interval -= 0.08
    text_interval_end = ax_interval_file_panel.text(
        0.5,
        current_y_pos_interval,
        "End: N/A",
        ha="center",
        va="center",
        transform=ax_interval_file_panel.transAxes,
    )
    current_y_pos_interval -= 0.12
    slider_interval_start.on_changed(on_interval_slider_change)
    slider_interval_start.ax.set_visible(False)
    slider_interval_start.valtext.set_visible(False)
    slider_interval_end.on_changed(on_interval_slider_change)
    slider_interval_end.ax.set_visible(False)
    slider_interval_end.valtext.set_visible(False)

    ax_btn_load = ax_interval_file_panel.inset_axes(
        [0.1, current_y_pos_interval - 0.1, 0.8, 0.1]
    )
    button_load = Button(
        ax_btn_load, "Load New Trajectory", color="lightcyan", hovercolor="cyan"
    )
    button_load.on_clicked(lambda event: load_new_trajectory())
    current_y_pos_interval -= 0.1 + 0.05
    ax_btn_save = ax_interval_file_panel.inset_axes(
        [0.1, current_y_pos_interval - 0.1, 0.8, 0.1]
    )
    button_save_interval = Button(
        ax_btn_save, "Save Selected Interval", color="lightgreen", hovercolor="green"
    )
    button_save_interval.on_clicked(save_selected_trajectory_interval)
    button_save_interval.ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Adjusted rect for legend space
    select_trajectory(-1)
    update_plot(reset_view=True)
    plt.show()


if __name__ == "__main__":
    main_gui()
