# List of nodes
- camera
    - in: camera port 
    - out: camera frames
- orbslam (repo: ORB_SLAM3, env: orbslam, run: ./Examples/Monocular/mono_realtime Vocabulary/ORBvoc.txt ./em/calibration_Misumi_200p.yaml)
    - in: camera frames 
    - out: camera pose, other orbslam things
- navigation (repo: navigation , env: nav , run: python navigation.py -view tp -live on)
    - in: camera pose 
    - out: visualization in 3d cad (both fpv and tpv)
- daq (repo: controllers (sensors), run: read_sensors
    - in: daq port 
    - out: pressure readings
- keyboard controller (repo: controllers, env: orbslam, run: python main.py)
    - in: motors and arduino ports, pressure readings 
    - out: robot movements (arduino and motors commands)

# Build read_sensor
- cd sensors/cpp
- mkdir build
- cd build
- cmake ..
- make

This will create the executable `read_sensor` in the `sensors/daq/cpp/build` directory. This executable reads the sensor data from the daq system and sends it to on localhost:5000. The data are read by the main.py file below

Run the webapp to visualize the data:
- cd data_visualization
- npm install
- npm start

This will start the webapp on localhost:65432 which will visualize the data from the daq system (pressure sensors). The data are sent by the main.py file below, after it is read by the `read_sensor` executable and filtered.


# How to run
- turn on motors power source
- connect the first motor couple to usb port 1 and wait
- connect in order the second couple and arduino
- connect the daq system

Terminal 1
- cd ~/Desktop/github/controllers/sensors/data_visualization
- nvm use 20
- npm start
- open browser to http://localhost:65432 to visualize pressure sensors

Terminal 2
- cd ~/Desktop/github/navigation 
- conda activate nav
- python navigation.py -view tp -live on

Terminal 3
- cd ~/Desktop/github/controllers
- conda activate orbslam
- python main.py 

Terminal 4
- cd ~/Desktop/github/ORB_SLAM3
- conda activate orbslam
- ./build/broncho_rgbd config.ini

Note: when you stop you need to run and stop the main.py file again to restart the system

# Motors info
## Motor mapping
motor_1 = Couple12.M1 <br>
motor_2 = Couple12.M2 <br>
motor_3 = Couple34.M1 <br>
motor_4 = Couple34.M2

## Movement mapping
up = motor_1 <br>
down = motor_3 <br>
right = motor_2 <br>
left = motor_4

## Motor limits
M1 : [0, 3000000] <br>
M2 : [0, 4000000] <br>
M3 : [0, 3000000] <br>
M4 : [0, 4000000]

## Ranges
down-up range <br>
motor_3      -   motor_1 <br>
[-3000000,0) U [0,3000000]

left-right range <br>
motor_4      -   motor_2 <br>
[-4000000,0) U [0,4000000] <br>
4000000      - 0 - 4000000


