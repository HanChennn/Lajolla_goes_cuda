cmake_minimum_required(VERSION 3.10)
project(lajolla_gpu)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE} "--std=c++17" "--expt-relaxed-constexpr" "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
########################################

include_directories(${CMAKE_SOURCE_DIR}/src)

set(SRC
    src/3rdparty/miniz.h
    src/3rdparty/miniz.c
    src/3rdparty/pugiconfig.hpp
    src/3rdparty/pugixml.hpp
    src/3rdparty/pugixml.cpp
    src/3rdparty/stb_image.h
    src/3rdparty/tinyexr.h
    src/3rdparty/tinyply.h

    src/filters/box.inl
    src/filters/gaussian.inl
    src/filters/tent.inl

    src/lights/diffuse_area_light.inl
    src/lights/envmap.inl

    src/materials/lambertian.inl
    src/materials/roughdielectric.inl
    src/materials/roughplastic.inl
    src/materials/disney_diffuse.inl
    src/materials/disney_metal.inl
    src/materials/disney_glass.inl
    src/materials/disney_clearcoat.inl
    src/materials/disney_sheen.inl
    src/materials/disney_bsdf.inl

    src/parsers/load_serialized.h
    src/parsers/parse_obj.h
    src/parsers/parse_ply.h
    src/parsers/parse_scene.h
    src/parsers/parsed_light.h
    src/parsers/parsed_shape.h
    src/parsers/parsed_table_dist.h
    src/parsers/parsed_texture.h
    src/parsers/load_serialized.cpp
    src/parsers/parse_obj.cpp
    src/parsers/parse_ply.cpp
    src/parsers/parsed_light.cpp
    src/parsers/parse_scene.cpp


    src/shapes/sphere.inl
    src/shapes/triangle_mesh.inl

    src/camera.h
    src/cuda_utils.h
    src/cuda_log.h
    src/filter.h
    src/flexception.h
    src/frame.h
    src/image.h
    src/intersection.h
    src/lajolla.h
    src/light.h
    src/material.h
    src/matrix.h
    src/microfacet.h
    src/mipmap.h
    src/path_tracing.h
    src/point_and_normal.h
    src/ray.h
    src/render.h
    src/scene.h
    src/shape.h
    src/spectrum.h
    src/table_dist.h
    src/texture.h
    src/transform.h
    src/vector.h

    src/camera.cpp
    src/image.cpp
    src/scene.cpp
    src/main.cpp

    src/render.cu
)

list(SORT SRC)
source_group(Sources FILES ${SRC})
cuda_add_executable(${CMAKE_PROJECT_NAME} ${SRC})

target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_CUDART_LIBRARY} ${CUDA_LIBRARIES})