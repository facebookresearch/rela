CMAKE_MINIMUM_REQUIRED(VERSION 3.15 FATAL_ERROR)
project(atari)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

# add dep
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../rela rela)

# python lib
pybind11_add_module(atari ${CMAKE_CURRENT_SOURCE_DIR}/pybind.cc)

target_include_directories(atari PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(
  atari PUBLIC SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/Arcade-Learning-Environment/src
)

target_link_directories(
  atari PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Arcade-Learning-Environment
)
target_link_libraries(atari PUBLIC ale)
target_link_libraries(atari PUBLIC _rela)
