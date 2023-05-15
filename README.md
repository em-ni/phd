# test
REMEMBER TO ACTIVATE THE ENVIRONMENT TO SET THE CORRECT PYTHON
0. mkdir(s) as follows
~~~
sofa/
- build/
    - master/
    - v22.12/
-src
    -... git clone sofa here
~~~

1. install boost, TinyXML and Eigen3 (:x64-windows) with vcpkg  

2. cd sofa

3. cmake -G "Visual Studio 17 2022" -S ".\src" -B ".\build\v22.12" -DTinyXML_DIR=C:\dev\vcpkg\installed\x64-windows\include -DBOOST_ROOT=C:\dev\vcpkg\installed\x64-windows\include -DEIGEN3_INCLUDE_DIR=C:\dev\vcpkg\installed\x64-windows\include -Dpybind11_DIR=C:\dev\vcpkg\installed\x64-windows\share\pybind11 -DSOFA_FETCH_SOFAPYTHON3=ON -DPLUGIN_SOFAPYTHON3=ON -DSOFA_EXTERNAL_DIRECTORIES=C:\Users\manun\Desktop\github\test\sofa\ext_plugin_repo -DPLUGIN_SPLIB=ON -DPLUGIN_STLIB=ON -DPLUGIN_BEAMADAPTER=ON -DPLUGIN_COSSERAT=ON -DPLUGIN_COLLISIONOBBCAPSULE=ON -DPLUGIN_MODELORDERREDUCTION=ON -DPLUGIN_SOFTROBOTS=ON

4. cd build\v22.12

5. cmake --build . --config Release  

6. cd bin\Release

7. runSofa.exe


-DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake

# Sofa Gym
follow [here](https://github.com/SofaDefrost/SofaGym/tree/e5cc4048fd1fbd0b93fd6e98b3a3d4854d094cfd)

1. create conda env with python 3.10
~~~
   conda create --name sofa python=3.10
~~~
2. tell env's python where to find sofa packages:  
~~~
   echo C:\path\to\sofa\build\v22.12\lib\python3\site-packages > path\to\anaconda3\envs\sofa\Lib\site-packages\sofa.pth
~~~
    


