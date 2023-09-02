# test
0. create conda env with python 3.10 and activate before compilation
check whihc python version is better, it seems that v22.12 in general has less problems with 3.8 but apparently SofaGym works only with python 3.10 

Also remember to not build ROS ina conda environment
~~~
   conda create --name sofa python=3.10
~~~
follow [here](https://www.sofa-framework.org/community/doc/getting-started/build/linux/)
1. mkdir(s) as follows
~~~
sofa/
- build/
    - master/
    - v22.12/
- ext_plugin_repo
    - here all the external plugins
- git clone sofagym
-src
    -... git clone sofa here
~~~
2. go to sofa dir 
~~~
cd sofa
~~~
3. configure
~~~
cmake -S "./src/sofa/src" -B "./build/v22.12" -DSOFA_FETCH_SOFAPYTHON3=ON -DPLUGIN_SOFAPYTHON3=ON -DSOFA_EXTERNAL_DIRECTORIES=${HOME}/Desktop/github/test/sofa/ext_plugin_repo -DPLUGIN_STLIB=ON -DPLUGIN_BEAMADAPTER=ON -DPLUGIN_COSSERAT=ON -DPLUGIN_COLLISIONOBBCAPSULE=ON -DPLUGIN_MODELORDERREDUCTION=ON -DPLUGIN_SOFTROBOTS=ON 

for https://github.com/StanfordASL/soft-robot-control and without fetching SofaPython3
cmake -S "./src/sofa/src" -B "./build/v22.12" -DPLUGIN_SOFAPYTHON3=ON -DSOFA_EXTERNAL_DIRECTORIES=${HOME}/Desktop/github/test/sofa/ext_plugin_repo -DPLUGIN_SPLIB=ON -DPLUGIN_STLIB=ON -DPLUGIN_BEAMADAPTER=ON -DPLUGIN_COSSERAT=ON -DPLUGIN_COLLISIONOBBCAPSULE=ON -DPLUGIN_MODELORDERREDUCTION=ON -DPLUGIN_SOFTROBOTS=ON -DSOFA_BUILD_METIS=ON -DSOFTROBOTS_IGNORE_ERRORS=ON -DCMAKE_PREFIX_PATH=${HOME}/Desktop/github/test/sofa/src/sofa/src/applications/plugins/SofaPython3/Testing


if problem with SofaPython3testing add
-DCMAKE_PREFIX_PATH=/home/emanuele/Desktop/github/test/sofa/src/sofa/src/applications/plugins/SofaPython3/Testing

and be sure the folder exists

if it does not exists it happens after fetching SofaPython3, since apparently the last verision does not contain the folder Testing.
Add it manually either by searching on github SofaPython3TestingConfig.cmake or take it from sofa/utils/

~~~
4. build 
~~~
cd build/v22.12
cmake --build . --config Release -jX
use nproc to check the maximum number of processors you can use, if you use -j without number, cmake will use all the available processors, slowing down the system
~~~
5. test build
~~~
cd bin\Release
runSofa.exe
~~~

if you get:
runSofa: /home/emanuele/anaconda3/envs/sofa/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/emanuele/Desktop/github/test/sofa/build/v22.12/lib/libSofa.Simulation.Core.so.22.12.00)   

try:
conda install -c conda-forge gcc


# Sofa Gym
follow [here](https://github.com/SofaDefrost/SofaGym/tree/e5cc4048fd1fbd0b93fd6e98b3a3d4854d094cfd)


    


