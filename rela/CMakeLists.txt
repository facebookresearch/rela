cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")


# get and append paths for finding dep
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})


# find packages & third_party
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)
find_package(Torch REQUIRED)


# message(${CMAKE_CURRENT_SOURCE_DIR}/../)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11 third_party/pybind11
)


# lib for other c++ programs
add_library(_rela
  types.cc
)
target_include_directories(_rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(_rela PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_rela PUBLIC ${TORCH_LIBRARIES})


# python lib
pybind11_add_module(rela pybind.cc)
target_link_libraries(rela PUBLIC _rela)
# target_include_directories(rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# # # GRPC and Protocol Buffers libraries location
# # list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX}/lib)
# # # message("${CMAKE_PREFIX_PATH}")

# # Cmake find modules
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/rpc/cmake)
# message(${CMAKE_CURRENT_SOURCE_DIR})

# find_package(Protobuf 3.8 REQUIRED)
# find_package(GRPC REQUIRED)
# find_package(OpenSSL REQUIRED)



# # # rpc
# # execute_process(COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/rpc/proto_gen.sh)
# # add_library(_rpc
# #   rpc/proto_src/env.pb.cc
# #   rpc/proto_src/env.grpc.pb.cc)
# # target_link_libraries(_rpc PUBLIC
# #   _rela
# #   gRPC::grpc++_reflection
# #   protobuf::libprotobuf
# #   OpenSSL::SSL
# # )


# # # tests
# # add_executable(test_data_channel src_cpp/test/test_data_channel.cc)
# # target_link_libraries(test_data_channel _rela)

# # add_executable(test_multistep_transition src_cpp/test/test_multistep_transition.cc)
# # target_link_libraries(test_multistep_transition _rela)

# # add_executable(test_sum_tree src_cpp/test/test_sum_tree.cc)
# # target_link_libraries(test_sum_tree _rela)

# # add_executable(test_context src_cpp/test/test_context.cc)
# # target_link_libraries(test_context _rela)


# # # build rpc test
# # add_executable(test_rpc_env rpc/test_rpc_env.cc)
# # target_include_directories(test_rpc_env PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# # target_link_libraries(test_rpc_env _rpc)


# # pybind lib
# pybind11_add_module(rela src_cpp/pybind.cc)
# # target_link_libraries(rela PUBLIC _rela _rpc)
# target_link_libraries(rela PUBLIC _rela)
# target_include_directories(rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# # target_include_directories(rela PUBLIC ${TORCH_INCLUDE_DIRS})
# # target_include_directories(rela PUBLIC ${PYTHON_INCLUDE_DIRS})
# # target_link_libraries(rela PUBLIC ${TORCH_LIBRARIES})


# # disable grpc warnings
# set_source_files_properties(
#   ${CMAKE_CURRENT_SOURCE_DIR}/rpc/proto_src/env.pb.cc
#   ${CMAKE_CURRENT_SOURCE_DIR}/rpc/proto_src/env.grpc.pb.cc
#   ${CMAKE_CURRENT_SOURCE_DIR}/rpc/test_rpc_env.cc
#   ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp/pybind.cc
#   PROPERTIES
#   COMPILE_FLAGS -w
# )
