# controllers
Build read_sensor:
- cd sensors/cpp
- mkdir build
- cd build
- cmake ..
- make

This will create the executable `read_sensor` in the `sensors/cpp/build` directory. This executable reads the sensor data from the daq system and sends it to on localhost:5000. The data are read by the main.py file below

Run the webapp to visualize the data:
- cd data_visualization
- npm install
- npm start

This will start the webapp on localhost:65432 which will visualize the data from the daq system (pressure sensors). The data are sent by the main.py file below, after it is read by the `read_sensor` executable and filtered.

How to run the system:
- 
- turn on motors power source
- connect the first motor couple to usb port 1 and wait
- connect in order the second couple and arduino
- connect the daq system
- run the main.py file

Note: when you stop you need to run and stop the main.py file again to restart the system
