cmake_minimum_required(VERSION 3.9)

project( stereo_estimation )

###############
# packages #
###############

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)

###############
# executables #
###############

add_executable( OpenCV_naive_stereo src/main.cpp )
target_link_libraries( OpenCV_naive_stereo ${OpenCV_LIBS} )

add_executable( pcjbf_experiment src/pcjbf_cli.cpp src/source_jbu.cpp src/pcjbf.cpp )
target_compile_features(pcjbf_experiment PRIVATE cxx_std_17)
target_link_libraries( pcjbf_experiment ${OpenCV_LIBS} )

if(OpenMP_CXX_FOUND)
    target_link_libraries(OpenCV_naive_stereo OpenMP::OpenMP_CXX)
endif()
