# Synchronized Video + Trajectory Visualization with Split functionality
# Shows video in upper panel, 3D trajectory in lower panel with current position highlighted

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import subprocess

# --- Configuration ---
DATA_FOLDER = "data"
SEQUENCE_NAME = "bbb2"  # Change this to visualize different sequences


def load_trajectory(tum_path):
    """Load TUM format trajectory file."""
    data = np.loadtxt(tum_path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]  # x, y, z
    quaternions = data[:, 4:8]  # qx, qy, qz, qw
    return timestamps, positions, quaternions, data


def save_trajectory(output_path, data):
    """Save trajectory in TUM format."""
    with open(output_path, 'w') as f:
        for row in data:
            f.write(f'{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} '
                    f'{row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f}\n')


def split_video(video_path, output_path, start_frame, end_frame, fps):
    """Split video using ffmpeg."""
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',
        output_path
    ]
    
    print(f"Splitting video: {start_frame} to {end_frame} -> {output_path}")
    subprocess.run(cmd, capture_output=True)


class TrajectoryVisualizer:
    def __init__(self, video_path, tum_path):
        self.video_path = video_path
        self.tum_path = tum_path
        self.is_playing = True
        self.current_frame = 0
        
        # Load trajectory
        self.traj_timestamps, self.traj_positions, self.traj_quaternions, self.traj_data = load_trajectory(tum_path)
        self.traj_timestamps_normalized = self.traj_timestamps - self.traj_timestamps[0]
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / self.fps
        
        print(f"Video: {self.total_frames} frames, {self.fps:.2f} fps, {self.video_duration:.2f}s")
        print(f"Trajectory: {len(self.traj_timestamps)} points, {self.traj_timestamps_normalized[-1]:.2f}s")
        
        self.setup_figure()
        
    def setup_figure(self):
        """Setup the matplotlib figure with buttons and slider."""
        self.fig = plt.figure(figsize=(10, 14))
        
        # Video subplot
        self.ax_video = self.fig.add_axes([0.05, 0.42, 0.9, 0.53])
        
        # Trajectory subplot
        self.ax_traj = self.fig.add_axes([0.05, 0.15, 0.9, 0.25], projection='3d')
        
        # Timeline slider
        self.ax_slider = self.fig.add_axes([0.15, 0.08, 0.7, 0.03])
        self.slider = Slider(self.ax_slider, 'Time', 0, self.video_duration, 
                             valinit=0, valstep=0.01)
        self.slider.on_changed(self.on_slider_change)
        self.slider_updating = False  # Flag to prevent feedback loop
        
        # Buttons
        self.ax_play = self.fig.add_axes([0.25, 0.02, 0.15, 0.04])
        self.ax_split = self.fig.add_axes([0.6, 0.02, 0.15, 0.04])
        
        self.btn_play = Button(self.ax_play, 'Pause')
        self.btn_split = Button(self.ax_split, 'Split Here')
        
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_split.on_clicked(self.split_at_current)
        
        # Initialize video frame
        ret, frame = self.cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.im = self.ax_video.imshow(frame_rgb)
        self.ax_video.axis('off')
        self.ax_video.set_title('Video', fontsize=12)
        
        # Plot full trajectory
        self.ax_traj.plot(self.traj_positions[:, 0], self.traj_positions[:, 1], self.traj_positions[:, 2], 
                          'b-', alpha=0.3, linewidth=1, label='Full trajectory')
        
        # Current position marker
        self.current_point, = self.ax_traj.plot([], [], [], 'ro', markersize=12, label='Current')
        self.trail_line, = self.ax_traj.plot([], [], [], 'g-', linewidth=2, alpha=0.8, label='Travelled')
        
        # Set trajectory axis labels
        self.ax_traj.set_xlabel('X (m)')
        self.ax_traj.set_ylabel('Y (m)')
        self.ax_traj.set_zlabel('Z (m)')
        self.ax_traj.set_title('Trajectory', fontsize=12)
        self.ax_traj.legend(loc='upper left', fontsize=8)
        
        # Set equal aspect ratio
        max_range = np.max([
            self.traj_positions[:, 0].max() - self.traj_positions[:, 0].min(),
            self.traj_positions[:, 1].max() - self.traj_positions[:, 1].min(),
            self.traj_positions[:, 2].max() - self.traj_positions[:, 2].min()
        ]) / 2.0
        mid = self.traj_positions.mean(axis=0)
        self.ax_traj.set_xlim(mid[0] - max_range, mid[0] + max_range)
        self.ax_traj.set_ylim(mid[1] - max_range, mid[1] + max_range)
        self.ax_traj.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
    def toggle_play(self, event):
        """Toggle play/pause state."""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Play' if not self.is_playing else 'Pause')
        self.fig.canvas.draw_idle()
    
    def on_slider_change(self, val):
        """Handle slider value change."""
        if self.slider_updating:
            return
        
        # Convert time to frame
        self.current_frame = int(val * self.fps)
        self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))
        
        # Update display
        self.update_display()
        
    def get_current_traj_idx(self):
        """Get trajectory index for current video frame."""
        current_time = self.current_frame / self.fps
        traj_idx = np.searchsorted(self.traj_timestamps_normalized, current_time)
        return min(traj_idx, len(self.traj_timestamps) - 1)
        
    def split_at_current(self, event):
        """Split video and trajectory at current position."""
        split_frame = self.current_frame
        split_traj_idx = self.get_current_traj_idx()
        
        if split_frame <= 0 or split_frame >= self.total_frames - 1:
            print("Cannot split at start or end of video")
            return
        if split_traj_idx <= 0 or split_traj_idx >= len(self.traj_timestamps) - 1:
            print("Cannot split at start or end of trajectory")
            return
        
        # Get base paths
        base_dir = os.path.dirname(self.video_path)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        traj_name = os.path.splitext(os.path.basename(self.tum_path))[0]
        video_ext = os.path.splitext(self.video_path)[1]
        
        # Output paths
        video_part1 = os.path.join(base_dir, f"{video_name}_part1{video_ext}")
        video_part2 = os.path.join(base_dir, f"{video_name}_part2{video_ext}")
        traj_part1 = os.path.join(base_dir, f"{traj_name}_part1.txt")
        traj_part2 = os.path.join(base_dir, f"{traj_name}_part2.txt")
        
        print(f"\n{'='*50}")
        print(f"Splitting at frame {split_frame}/{self.total_frames}, trajectory point {split_traj_idx}/{len(self.traj_timestamps)}")
        print(f"{'='*50}")
        
        # Split video using ffmpeg
        split_video(self.video_path, video_part1, 0, split_frame, self.fps)
        split_video(self.video_path, video_part2, split_frame, self.total_frames, self.fps)
        
        # Split trajectory
        traj_part1_data = self.traj_data[:split_traj_idx]
        traj_part2_data = self.traj_data[split_traj_idx:]
        
        save_trajectory(traj_part1, traj_part1_data)
        save_trajectory(traj_part2, traj_part2_data)
        
        print(f"\nCreated files:")
        print(f"  Video Part 1: {video_part1} ({split_frame} frames)")
        print(f"  Video Part 2: {video_part2} ({self.total_frames - split_frame} frames)")
        print(f"  Trajectory Part 1: {traj_part1} ({len(traj_part1_data)} points)")
        print(f"  Trajectory Part 2: {traj_part2} ({len(traj_part2_data)} points)")
        print(f"{'='*50}\n")
        
    def update_display(self):
        """Update video and trajectory display for current frame."""
        # Read video frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.im.set_array(frame_rgb)
        
        # Get trajectory index
        traj_idx = self.get_current_traj_idx()
        current_time = self.current_frame / self.fps
        
        # Update current position marker
        pos = self.traj_positions[traj_idx]
        self.current_point.set_data([pos[0]], [pos[1]])
        self.current_point.set_3d_properties([pos[2]])
        
        # Update trail
        self.trail_line.set_data(self.traj_positions[:traj_idx+1, 0], self.traj_positions[:traj_idx+1, 1])
        self.trail_line.set_3d_properties(self.traj_positions[:traj_idx+1, 2])
        
        # Update titles
        self.ax_video.set_title(f'Video - Frame {self.current_frame}/{self.total_frames} ({current_time:.2f}s)', fontsize=12)
        self.ax_traj.set_title(f'Trajectory - Point {traj_idx}/{len(self.traj_timestamps)}', fontsize=12)
        
        # Update slider without triggering callback
        self.slider_updating = True
        self.slider.set_val(current_time)
        self.slider_updating = False
        
        self.fig.canvas.draw_idle()
    
    def update(self, frame_number):
        """Update animation frame."""
        if not self.is_playing:
            return self.im, self.current_point, self.trail_line
        
        # Advance frame
        self.current_frame = (self.current_frame + max(1, int(self.fps / 15))) % self.total_frames
        
        self.update_display()
        
        return self.im, self.current_point, self.trail_line
    
    def run(self):
        """Start the visualization."""
        self.anim = FuncAnimation(self.fig, self.update, interval=1000/15, blit=False, cache_frame_data=False)
        plt.show()
        self.cap.release()


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, DATA_FOLDER)
    
    video_path = os.path.join(data_folder, f"{SEQUENCE_NAME}.mkv")
    tum_path = os.path.join(data_folder, f"{SEQUENCE_NAME}_gt.txt")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(tum_path):
        raise FileNotFoundError(f"Trajectory not found: {tum_path}")
    
    print(f"Visualizing sequence: {SEQUENCE_NAME}")
    print(f"Video: {video_path}")
    print(f"Trajectory: {tum_path}")
    print("-" * 50)
    
    viz = TrajectoryVisualizer(video_path, tum_path)
    viz.run()
