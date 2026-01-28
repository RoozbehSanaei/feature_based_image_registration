cmake_minimum_required(VERSION 3.16)
project(feature_registration_no_cv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(feature_registration_no_cv feature_registration_no_cv.cpp)
