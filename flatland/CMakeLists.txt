cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
cmake_policy(SET CMP0015 NEW)
project(LIGHT_LOCALIZATION)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)
set(OPENEXR_LIBRARIES IlmImf Half)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
if(APPLE)
    set(GLLIBS ${OPENGL_LIBRARIES})
else()
    find_package(GLEW REQUIRED)
    include_directories(${GLEW_INCLUDE_DIRS})
    set(GLLIBS ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})
endif()

CUDA_ADD_LIBRARY(cudaconvolution
    convolutionSeparable.cu convolutionSeparable.h)
CUDA_ADD_LIBRARY(cudamap
    cudamap.cu cudamap.h)

set(CMAKE_CXX_FLAGS "-Wno-deprecated -std=c++0x")
add_executable(visual
    visual.cpp
    options.h
    loadshader.cpp loadshader.h
    opengl_compat.cpp opengl_compat.h
    fileio.cpp fileio.h
    trackball.cpp trackball.h
    solveCeres.cpp solveCeres.h)
target_link_libraries(visual cudamap cudaconvolution ${CUDA_LIBRARIES} ${OPENEXR_LIBRARIES} ${PNG_LIBRARIES}
    ${GLLIBS} ${CERES_LIBRARIES})
add_custom_command(TARGET visual POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/*.glsl
    $<TARGET_FILE_DIR:visual>)
