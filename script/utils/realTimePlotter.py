import socket
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import threading
from collections import deque

# Initialize plotting variables
fig, ax = plt.subplots()
dataDict = {}  # Each key will have its own deque
times = deque(maxlen=100)
lines = {}  # Keep track of line objects for each data series
colors = ['r', 'g', 'b', 'y']  # Predefined colors for the lines

# Global lock for thread safety
data_lock = threading.Lock()
start_time = None  # Initialize start time

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    return []

def adjust_y_axis():
    y_values = []
    for deque in dataDict.values():
        y_values.extend(list(deque))
    
    if not y_values:
        return

    ymin, ymax = min(y_values), max(y_values)
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    buffer = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - buffer, ymax + buffer)

def update(frame):
    global start_time
    with data_lock:
        if not times or start_time is None:
            return []

        # Convert the current time to seconds relative to the start time
        time_since_start = [t - start_time for t in times]
        current_time_relative = time_since_start[-1]
        
        ax.set_xlim(current_time_relative - 10, current_time_relative)
        adjust_y_axis()

        for key, color in zip(dataDict.keys(), colors):
            if key not in lines:
                lines[key], = ax.plot([], [], color=color, label=key)
            line = lines[key]
            # Update line data with time relative to start
            line.set_data(time_since_start, list(dataDict[key]))

        ax.legend()
        return list(lines.values())

def listenForData():
    global start_time
    host, port = '127.0.0.1', 65432
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f'Connected by {addr}')
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                received_data = json.loads(data.decode('utf-8'))

                current_time = time.time()
                with data_lock:
                    if start_time is None:
                        start_time = current_time  # Set the start time at the first data arrival
                    times.append(current_time)
                    for key, value in received_data.items():
                        if key not in dataDict:
                            dataDict[key] = deque(maxlen=100)
                        dataDict[key].append(value)

def animate():
    ani = FuncAnimation(fig, update, init_func=init, frames=None, blit=False, interval=1000, repeat=False)
    plt.show()

if __name__ == '__main__':
    threading.Thread(target=listenForData, daemon=True).start()
    animate()
