cmake_minimum_required(VERSION 3.10)  # Set the minimum version of CMake required

project(read_sensors VERSION 1.0 LANGUAGES C CXX)  # Name of your project

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Source files
set(SRC_EXPR
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c*
)

# Header files
set(HEADER_EXPR
    ${CMAKE_CURRENT_LIST_DIR}/include/*.c*
)

file(GLOB HEADERS ${HEADER_EXPR})
file(GLOB SRCS ${SRC_EXPR})

# Include directories
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
    /usr/local/include/libusb-1.0
)  

# Build
add_executable(${PROJECT_NAME} ${HEADERS} ${SRCS})

# Link libraries
link_directories(/usr/local/lib)  
target_link_libraries(${PROJECT_NAME} usb-1.0 hidapi-libusb)
