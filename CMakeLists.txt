cmake_minimum_required(VERSION 3.15)
project(cplus_leetcode)

set(CMAKE_CXX_STANDARD 11)

add_executable(cplus_leetcode main.cpp container/emplace_back_check.cpp container/emplace_back_check.h container/rvalue_check.cpp container/rvalue_check.h container/vector_vector_emplace_check.cpp container/vector_vector_emplace_check.h container/random_engine.h)