cmake_minimum_required(VERSION 3.12)
project(MyPlugin VERSION 1.0)

#look for all packages on which your project depends
find_package(SofaFramework REQUIRED)

set(HEADER_FILES
    MyHeaderFile1.h
    MyHeaderFile2.h
    MyHeaderFile3.h
)

set(SOURCE_FILES
    MySourceFile1.cpp
    MySourceFile2.cpp
    MySourceFile3.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MYPLUGIN")