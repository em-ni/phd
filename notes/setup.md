# Build SOFA
## Create env
Create conda env with python 3.10 and activate.</br>
Check which python version is better, it seems that v22.12 in general has less problems with 3.8 but apparently SofaGym works only with python 3.10.</br>
Currently using Python 3.10.11, both sofa script ([example](script/main)) and SofaGym examples ([example](sofa/SofaGym/test_env.py)) work.</br>
Also remember to not build ROS in conda environment
~~~
   conda create --name sim python=3.10
   conda activate sim
~~~

## Directories
mkdir(s) as follows (Instructions are taken from [here](https://www.sofa-framework.org/community/doc/getting-started/build/linux/))
~~~
sofa/
- build/
    - master/
    - v22.12/
- ext_plugin_repo
    - here all the external plugins
- ... git clone sofagym here
-src
    - ... git clone sofa here
~~~
## Configure and Build
~~~
cd sofa
cmake -S "./src/sofa/src" -B "./build/v22.12" -DSOFA_FETCH_SOFAPYTHON3=ON -DPLUGIN_SOFAPYTHON3=ON -DSOFA_EXTERNAL_DIRECTORIES=${HOME}/Desktop/github/sim/sofa/ext_plugin_repo -DPLUGIN_STLIB=ON -DPLUGIN_BEAMADAPTER=ON -DPLUGIN_COSSERAT=ON -DPLUGIN_COLLISIONOBBCAPSULE=ON -DPLUGIN_MODELORDERREDUCTION=ON -DPLUGIN_SOFTROBOTS=ON 
~~~
Be sure conda env is activated and build 
~~~
cmake --build ./build/v22.12 --config Release -j10
~~~
use nproc to check the maximum number of processors you can use, if you use -j without number, cmake will use all the available processors, slowing down the system. </br>
Test build
~~~
cd bin\Release
runSofa.exe
~~~
If it works, add to ~/.bashrc
~~~
export PATH="$HOME/Desktop/github/sim/sofa/build/v22.12/bin:$PATH"
~~~

### Alternatives and possible problems:</br>
From https://github.com/StanfordASL/soft-robot-control  (without fetching SofaPython3)
~~~
cmake -S "./src/sofa/src" -B "./build/v22.12" -DPLUGIN_SOFAPYTHON3=ON -DSOFA_EXTERNAL_DIRECTORIES=${HOME}/Desktop/github/sim/sofa/ext_plugin_repo -DPLUGIN_SPLIB=ON -DPLUGIN_STLIB=ON -DPLUGIN_BEAMADAPTER=ON -DPLUGIN_COSSERAT=ON -DPLUGIN_COLLISIONOBBCAPSULE=ON -DPLUGIN_MODELORDERREDUCTION=ON -DPLUGIN_SOFTROBOTS=ON -DSOFA_BUILD_METIS=ON -DSOFTROBOTS_IGNORE_ERRORS=ON -DCMAKE_PREFIX_PATH=${HOME}/Desktop/github/sim/sofa/src/sofa/src/applications/plugins/SofaPython3/Testing
~~~

If problem with SofaPython3testing add
~~~
-DCMAKE_PREFIX_PATH=/home/emanuele/Desktop/github/sim/sofa/src/sofa/src/applications/plugins/SofaPython3/Testing
~~~
and be sure the folder exists. </br>
If it does not exists it happens after fetching SofaPython3, since apparently the last verision does not contain the folder Testing.
Add it manually either by searching on github SofaPython3TestingConfig.cmake or take it from sofa/utils/

If you get something like </br>
/usr/bin/ld: cannot find /lib64/libpthread.so.0: No such file or directory
/usr/bin/ld: cannot find /usr/lib64/libpthread_nonshared.a: No such file or directory</br>
it can be solved with linker option for cmake. First locate the library (ex from terminal: locate libpthread.so) and get the directory, then rigenerate adding </br>
-DCMAKE_LIBRARY_PATH="/home/emanuele/anaconda3/envs/sofa/x86_64-conda-linux-gnu/sysroot/lib64;/home/emanuele/anaconda3/envs/sofa/x86_64-conda-linux-gnu/sysroot/usr/lib64" 
-DCMAKE_EXE_LINKER_FLAGS="-lpthread" </br>
(I think in general is -DCMAKE_EXE_LINKER_FLAGS="-lNAME_OF_THE_LIBRARY"), but I am not sure if it is the best solution, also it is probably related to the next big problem. </br>


If you get:</br>
runSofa: /home/emanuele/anaconda3/envs/sofa/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/emanuele/Desktop/github/sim/sofa/build/v22.12/lib/libSofa.Simulation.Core.so.22.12.00)  </br>
try:
~~~
conda install -c conda-forge gcc
~~~
!!! the last command causes a lot of problems if you try to build again, find a different solution !!!</br>
so if you want to rebuild after this command, use 
~~~
conda remove gcc
~~~
or conda uninstall gcc and the build again. Once built, you can install gcc again.

# Sofa Gym
follow [here](https://github.com/SofaDefrost/SofaGym/tree/e5cc4048fd1fbd0b93fd6e98b3a3d4854d094cfd)

Requirements for SofaGym:
~~~
conda remove gcc
conda install -c conda-forge gym==0.21
pip install psutil pygame glfw pyopengl imageio lxml chardet olefile pyparsing==2.1.0 stable-baselines3[extra] colorama
python setup.py bdist_wheel
pip install -v -e .
conda install -c conda-forge gcc
python test_env.py -e trunk-v0 -ep 100 -s 100
conda install -c conda-forge gcc
~~~

# Setup environment
1. If first time copy ./sofa.sh in ${HOME}/bin 
2. source sofa.sh


# Script
Requirements:
~~~
pip install gmsh trimesh vtk
~~~
