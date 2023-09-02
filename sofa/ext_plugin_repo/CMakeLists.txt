cmake_minimum_required(VERSION 2.8.12)

find_package(Sofa.Framework)

#sofa_add_subdirectory(plugin ./SofaMAMMOBOT  SofaMAMMOBOT)
#sofa_add_subdirectory(plugin ./SPLIB  SPLIB)
sofa_add_subdirectory(plugin ./STLIB  STLIB)
sofa_add_subdirectory(plugin ./SoftRobots  SoftRobots)
sofa_add_subdirectory(plugin ./ModelOrderReduction  ModelOrderReduction)
sofa_add_subdirectory(plugin ./Cosserat  Cosserat)
#sofa_add_subdirectory(plugin ./SofaPython3  SofaPython3)

sofa_add_subdirectory(plugin ./MyPlugin MyPlugin)
