cmake_minimum_required(VERSION 3.27)
project(natten_mps_ext LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python + nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Find PyTorch
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# ---- Compile .metal -> .metallib ----
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/natten_mps/_core/shaders")
set(METAL_SOURCE "${SHADER_DIR}/natten_nb_1d.metal")
set(AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/natten_nb_1d.air")
set(METALLIB_FILE "${CMAKE_CURRENT_BINARY_DIR}/natten_nb_1d.metallib")

add_custom_command(
    OUTPUT ${AIR_FILE}
    COMMAND xcrun -sdk macosx metal -c ${METAL_SOURCE} -o ${AIR_FILE}
    DEPENDS ${METAL_SOURCE}
    COMMENT "Compiling Metal shader to AIR"
)

add_custom_command(
    OUTPUT ${METALLIB_FILE}
    COMMAND xcrun -sdk macosx metallib ${AIR_FILE} -o ${METALLIB_FILE}
    DEPENDS ${AIR_FILE}
    COMMENT "Linking AIR to metallib"
)

add_custom_target(metal_shaders DEPENDS ${METALLIB_FILE})

# ---- Nanobind extension module ----
nanobind_add_module(_nanobind_ext
    src/natten_mps/_core/_nanobind_ext.cpp
)

# Force Objective-C++ compilation for Metal API usage
set_source_files_properties(
    src/natten_mps/_core/_nanobind_ext.cpp
    PROPERTIES LANGUAGE OBJCXX
)

target_link_libraries(_nanobind_ext PRIVATE
    ${TORCH_LIBRARIES}
    "-framework Metal"
    "-framework Foundation"
)

target_include_directories(_nanobind_ext PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

add_dependencies(_nanobind_ext metal_shaders)

# ---- Install ----
install(TARGETS _nanobind_ext LIBRARY DESTINATION natten_mps/_core)
install(FILES ${METALLIB_FILE} DESTINATION natten_mps/_core/shaders)
