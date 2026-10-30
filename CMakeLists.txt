cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(gputt LANGUAGES CXX VERSION 0.1)

include(GNUInstallDirs)

set(_gputt_backend "${GPUTT_BACKEND}")
if(NOT GPUTT_BACKEND)
  if(CMAKE_CUDA_ARCHITECTURES AND CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR "Both GPU architecture families are set; select GPUTT_BACKEND=CUDA or HIP")
  elseif(CMAKE_HIP_ARCHITECTURES)
    set(_gputt_backend HIP)
  elseif(CMAKE_CUDA_ARCHITECTURES)
    set(_gputt_backend CUDA)
  endif()
endif()
set(GPUTT_BACKEND "${_gputt_backend}" CACHE STRING "GPU backend: CUDA or HIP" FORCE)
set_property(CACHE GPUTT_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT GPUTT_BACKEND MATCHES "^(CUDA|HIP)$")
  message(FATAL_ERROR "Set GPUTT_BACKEND=CUDA or HIP and the corresponding CMAKE_<BACKEND>_ARCHITECTURES")
endif()
if(NOT CMAKE_${GPUTT_BACKEND}_ARCHITECTURES)
  message(FATAL_ERROR "Please define CMAKE_${GPUTT_BACKEND}_ARCHITECTURES explicitly")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(GPUTT_BACKEND STREQUAL "HIP")
  if(CMAKE_HIP_PLATFORM AND NOT CMAKE_HIP_PLATFORM STREQUAL "amd")
    message(FATAL_ERROR "Use GPUTT_BACKEND=CUDA for NVIDIA GPUs")
  endif()
  set(CMAKE_HIP_PLATFORM amd)
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)
  set(_gputt_runtime hip::host)
else()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(_gputt_runtime CUDA::cudart)
endif()

if(UNIX)
  if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
  else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
  endif()
endif()

option(GPUTT_BUILD_PYTHON "Build Python bindings" ${PROJECT_IS_TOP_LEVEL})
option(GPUTT_BUILD_TESTS "Build correctness tests" ${PROJECT_IS_TOP_LEVEL})
option(GPUTT_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(GPUTT_BUILD_EXAMPLES "Build example executable" ${PROJECT_IS_TOP_LEVEL})
option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)

file(GLOB gputt_SOURCE_FILES "src/*.cu")
set_source_files_properties(${gputt_SOURCE_FILES} PROPERTIES LANGUAGE ${GPUTT_BACKEND})
add_library(gputt STATIC ${gputt_SOURCE_FILES})
add_library(gputt::gputt ALIAS gputt)
target_compile_features(gputt PUBLIC cxx_std_17)
set_property(TARGET gputt PROPERTY POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(gputt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/gputt>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/gputt>)
target_link_libraries(gputt PUBLIC ${_gputt_runtime})

if(ENABLE_NVTOOLS)
  if(NOT GPUTT_BACKEND STREQUAL "CUDA" OR NOT TARGET CUDA::nvToolsExt)
    message(FATAL_ERROR "ENABLE_NVTOOLS requires a CUDA toolkit providing nvToolsExt (CUDA 12 or earlier)")
  endif()
  target_compile_definitions(gputt PRIVATE ENABLE_NVTOOLS)
  target_link_libraries(gputt PRIVATE CUDA::nvToolsExt)
endif()

foreach(_kind IN ITEMS test bench example)
  if((_kind STREQUAL "test" AND GPUTT_BUILD_TESTS) OR
     (_kind STREQUAL "bench" AND GPUTT_BUILD_BENCHMARKS) OR
     (_kind STREQUAL "example" AND GPUTT_BUILD_EXAMPLES))
    set(_source "${CMAKE_CURRENT_SOURCE_DIR}/src/${_kind}/${_kind}.cu")
    set_source_files_properties(${_source} PROPERTIES LANGUAGE ${GPUTT_BACKEND})
    add_executable(gputt_${_kind} ${_source})
    target_link_libraries(gputt_${_kind} PRIVATE gputt::gputt)
  endif()
endforeach()
if(GPUTT_BUILD_TESTS)
  enable_testing()
  add_test(NAME gputt_test COMMAND gputt_test)
endif()

if(GPUTT_BUILD_PYTHON)
  if("${Python3_VERSION}" STREQUAL "")
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  else()
    find_package(Python3 ${Python3_VERSION} EXACT REQUIRED COMPONENTS Interpreter Development)
  endif()
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
  set(PYBIND11_PYTHON_VERSION ${Python3_VERSION})
  add_subdirectory(ThirdParty/pybind11)
  set(_python_sources src/python/gputt.cu src/python/gputt_module.cu)
  set_source_files_properties(${_python_sources} PROPERTIES LANGUAGE ${GPUTT_BACKEND})
  pybind11_add_module(gputt_python ${_python_sources})
  target_compile_definitions(gputt_python PRIVATE gputt_EXPORTS)
  target_link_libraries(gputt_python PRIVATE gputt::gputt)
  set_target_properties(gputt_python PROPERTIES LIBRARY_OUTPUT_NAME gputt)
  install(TARGETS gputt_python LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(TARGETS gputt EXPORT gputt-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/gputt/gputt.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gputt)
install(EXPORT gputt-targets NAMESPACE gputt:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gputt)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/gputtConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gputt")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gputtConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/gputtConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/gputtConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gputt)
