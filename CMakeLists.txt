cmake_minimum_required(VERSION 3.10)

project(CUDA-by-Example LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(./common)

add_subdirectory(chapter10)