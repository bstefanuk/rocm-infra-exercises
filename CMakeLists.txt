cmake_minimum_required(VERSION 3.25)
project(simple-co-load LANGUAGES CXX HIP)

find_package(hip REQUIRED)

add_executable(simple-co-load
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(simple-co-load
    PRIVATE
        hip::host
        hip::device
)

