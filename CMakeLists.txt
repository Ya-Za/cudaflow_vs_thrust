cmake_minimum_required(VERSION 3.0.0)
project(cudaflow_vs_thrust VERSION 0.1.0 LANGUAGES CXX CUDA)

add_executable(cudaflow_vs_thrust main.cu)
target_include_directories(cudaflow_vs_thrust PRIVATE "/home/yasin/codes/lib/taskflow")
target_link_libraries(cudaflow_vs_thrust PRIVATE pthread)

target_compile_features(cudaflow_vs_thrust PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(cudaflow_vs_thrust PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wfatal-errors>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-Wall,-Wextra,-Wfatal-errors>
)
