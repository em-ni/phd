
cmake_minimum_required(VERSION 3.22)

# Top-level CMake wrapper for SOFA external plugins repository
# See: https://www.sofa-framework.org/docs/user/plugin/build_plugin/

project(sofa_external_plugins NONE)

# Find the Sofa framework (must be available when configuring SOFA in-tree)
find_package(SofaFramework REQUIRED)

# Register each plugin directory with SOFA using the provided helper macro.
# The first argument to `sofa_add_subdirectory` is the kind (plugin), the
# second is the relative path to the plugin directory and the third is the
# plugin project name (as defined in the plugin's CMakeLists.txt).

sofa_add_subdirectory(plugin Cosserat Cosserat)
sofa_add_subdirectory(plugin ModelOrderReduction ModelOrderReduction)
sofa_add_subdirectory(plugin STLIB STLIB)
# Do NOT add SofaPython3 here if you enable SOFA_FETCH_SOFAPYTHON3/PLUGIN_SOFAPYTHON3
# in the main SOFA build; it would create duplicate targets. Rely on SOFA's
# fetched/in-tree SofaPython3 instead.
# sofa_add_subdirectory(plugin SofaPython3 SofaPython3)
sofa_add_subdirectory(plugin SoftRobots SoftRobots)

message(STATUS "Configured external plugin repository: ${CMAKE_CURRENT_SOURCE_DIR}")
