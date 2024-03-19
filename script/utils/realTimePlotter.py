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

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    # Initialize an empty list to store line objects for blitting
    line_objs = []
    for color, key in zip(colors, dataDict.keys()):
        # Ensure each data series has a corresponding line object
        if key not in lines:
            lines[key], = ax.plot([], [], color=color, label=key, animated=True)
        line_objs.append(lines[key])
    ax.legend()
    return line_objs  # Return the list of line objects

def update(frame):
    # Initialize an empty list to store updated line objects for blitting
    line_objs = []
    if times:
        ax.set_xlim(times[0], times[-1] + 1)
        all_values = [item for sublist in dataDict.values() for item in sublist]
        if all_values:
            global_min = min(all_values)
            global_max = max(all_values)
            buffer = (global_max - global_min) * 0.1 if all_values else 1
            ax.set_ylim(global_min - buffer, global_max + buffer)
        
        for key in dataDict:
            if key in lines:
                lines[key].set_data(list(times), list(dataDict[key]))
                line_objs.append(lines[key])

    return line_objs



def listenForData():
    host, port = '127.0.0.1', 65432
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()            
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                received_data = json.loads(data.decode('utf-8'))
                print('Received', received_data)
                
                current_time = time.time()
                times.append(current_time)
                for key, value in received_data.items():
                    if key not in dataDict:
                        dataDict[key] = deque(maxlen=100)
                    dataDict[key].append(value)

def animate():
    # Fix the warning by explicitly specifying frames
    ani = FuncAnimation(fig, update, init_func=init, frames=range(100), blit=True, interval=1000, repeat=False)
    plt.show()

if __name__ == '__main__':
    threading.Thread(target=listenForData, daemon=True).start()
    animate()
