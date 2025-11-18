cmake_minimum_required(VERSION 3.10)
project(randomized_algo_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(random_qsort src/random_qsort.cpp)

# On some systems we need -static-libstdc++? Keep minimal and portable.
