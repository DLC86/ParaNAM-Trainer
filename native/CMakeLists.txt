cmake_minimum_required(VERSION 3.22)
project(ParaNAMPlayback LANGUAGES CXX)
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUNTIME "${CMAKE_CURRENT_LIST_DIR}/../external/playback-runtime")
set(CORE "${RUNTIME}/external/nam_core_parametric")
set(AUDIO "${RUNTIME}/external/AudioDSPTools")
# Mirror the current plugin's final single-network convolution interpolation.
# The core is pinned in external/playback-runtime and never modified by users
# at runtime. Apply idempotent patches only while configuring the native build.
set(PNAM_BASE_PATCH "${RUNTIME}/patches/core-hyperwavenet-single-weight-smoothing.patch")
if(NOT EXISTS "${CORE}/NAM/kernel_weight_smoothing.h")
  execute_process(
    COMMAND git -C "${CORE}" apply --reverse --check "${PNAM_BASE_PATCH}"
    RESULT_VARIABLE BASE_ALREADY OUTPUT_QUIET ERROR_QUIET)
  if(NOT BASE_ALREADY EQUAL 0)
    execute_process(
      COMMAND git -C "${CORE}" apply --check "${PNAM_BASE_PATCH}"
      RESULT_VARIABLE BASE_CAN_APPLY ERROR_VARIABLE BASE_ERROR)
    if(NOT BASE_CAN_APPLY EQUAL 0)
      message(FATAL_ERROR "Pinned playback NAM smoothing patch: ${BASE_ERROR}")
    endif()
    execute_process(
      COMMAND git -C "${CORE}" apply "${PNAM_BASE_PATCH}"
      RESULT_VARIABLE BASE_RESULT)
    if(NOT BASE_RESULT EQUAL 0)
      message(FATAL_ERROR "Playback NAM smoothing baseline patch failed")
    endif()
  endif()
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${RUNTIME}/patches/apply-direct-kernel-smoothing.py" "${CORE}"
  RESULT_VARIABLE DIRECT_RESULT OUTPUT_VARIABLE DIRECT_OUTPUT ERROR_VARIABLE DIRECT_ERROR)
if(NOT DIRECT_RESULT EQUAL 0)
  message(FATAL_ERROR "Playback direct-kernel weight smoothing patch: ${DIRECT_ERROR}")
endif()
message(STATUS "${DIRECT_OUTPUT}")
# Preserve the 100/100 ms single-network smoothing; accelerate only the
# conditioned delta convolution using the 0.5.2 plugin's preallocated GEMM.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/apply-gemm-delta-kernel.py" "${CORE}"
  RESULT_VARIABLE GEMM_RESULT OUTPUT_VARIABLE GEMM_OUTPUT ERROR_VARIABLE GEMM_ERROR)
if(NOT GEMM_RESULT EQUAL 0)
  message(FATAL_ERROR "Playback GEMM delta-kernel smoothing patch: ${GEMM_ERROR}")
endif()
message(STATUS "${GEMM_OUTPUT}")
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS "${CORE}/NAM/*.cpp" "${CORE}/NAM/*/*.cpp")
# Keep the plugin DSP unchanged except for Windows Unicode path interpretation.
file(READ "${RUNTIME}/Source/DSP/NamModelEngine.cpp" ENGINE_SOURCE)
string(REPLACE "std::filesystem::path(modelPath)" "std::filesystem::u8path(modelPath)" ENGINE_SOURCE "${ENGINE_SOURCE}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/NamModelEngine.cpp" CONTENT "${ENGINE_SOURCE}" @ONLY)
add_library(paranam_playback SHARED playback.cpp "${CMAKE_CURRENT_BINARY_DIR}/NamModelEngine.cpp" ${CORE_SOURCES})
target_include_directories(paranam_playback PRIVATE "${RUNTIME}/Source")
target_include_directories(paranam_playback SYSTEM PRIVATE "${CORE}" "${CORE}/NAM"
  "${CORE}/Dependencies/eigen" "${CORE}/Dependencies/nlohmann" "${AUDIO}")
target_compile_definitions(paranam_playback PRIVATE EIGEN_MPL2_ONLY=1 EIGEN_NO_DEBUG=1
  NAM_SAMPLE_FLOAT=1 NAM_ENABLE_A2_FAST=1 NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
  target_compile_options(paranam_playback PRIVATE /O2 /W0 /utf-8)
else()
  target_compile_options(paranam_playback PRIVATE -O3 -w)
endif()
install(TARGETS paranam_playback RUNTIME DESTINATION . LIBRARY DESTINATION .)
