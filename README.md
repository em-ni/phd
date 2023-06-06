# test
0. create conda env with python 3.10 and activate before compilation
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
cmake -S "./src/sofa/src" -B "./build/v22.12" -DSOFA_FETCH_SOFAPYTHON3=ON -DPLUGIN_SOFAPYTHON3=ON -DSOFA_EXTERNAL_DIRECTORIES=${HOME}/Desktop/github/test/sofa/ext_plugin_repo -DPLUGIN_SPLIB=ON -DPLUGIN_STLIB=ON -DPLUGIN_BEAMADAPTER=ON -DPLUGIN_COSSERAT=ON -DPLUGIN_COLLISIONOBBCAPSULE=ON -DPLUGIN_MODELORDERREDUCTION=ON -DPLUGIN_SOFTROBOTS=ON
~~~
4. build
~~~
cd build/v22.12
cmake --build . --config Release  
~~~
5. test build
~~~
cd bin\Release
runSofa.exe
~~~

# Sofa Gym
follow [here](https://github.com/SofaDefrost/SofaGym/tree/e5cc4048fd1fbd0b93fd6e98b3a3d4854d094cfd)


    


