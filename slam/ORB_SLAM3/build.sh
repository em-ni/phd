#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [release|debug|ros]"
    exit 1
fi

BUILD_OPTION=$1

if [ "$BUILD_OPTION" = "release" ]; then
    CMAKE_BUILD_TYPE="Release"
    ROS_BUILD=0
elif [ "$BUILD_OPTION" = "debug" ]; then
    CMAKE_BUILD_TYPE="Debug"
    ROS_BUILD=0
elif [ "$BUILD_OPTION" = "ros" ]; then
    CMAKE_BUILD_TYPE="Release"
    ROS_BUILD=1
else
    echo "Invalid argument: $BUILD_OPTION"
    echo "Usage: $0 [release|debug|ros]"
    exit 1
fi

echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="Release"
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="Release"
make -j

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="Release"
make -j

cd ../../../

# Check if ORBvoc.txt exists before uncompressing
if [ ! -f "Vocabulary/ORBvoc.txt" ]; then
    echo "Uncompressing vocabulary ..."
    cd Vocabulary
    tar -xf ORBvoc.txt.tar.gz
    cd ..
else
    echo "Vocabulary is already uncompressed."
fi

echo "Configuring and building ORB_SLAM3 ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
make -j6

if [ "$ROS_BUILD" -eq 1 ]; then
    echo "Building ROS nodes"

    cd ../../Examples/ROS/ORB_SLAM3
    mkdir -p build
    cd build
    cmake .. -DROS_BUILD_TYPE=Release
    make -j
fi
