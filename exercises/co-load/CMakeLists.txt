cmake_minimum_required(VERSION 3.25)
project(simple-co-load LANGUAGES CXX HIP)

find_package(hip REQUIRED)

add_executable(client-lib-bundle
    ${CMAKE_CURRENT_SOURCE_DIR}/client_lib_bundle.cpp
)

target_link_libraries(client-lib-bundle
    PRIVATE
        hip::host
        hip::device
)

add_library(device-lib)

target_sources(device-lib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib.cpp
)

target_link_libraries(device-lib
    PRIVATE
        hip::device
)


add_executable(client)

target_sources(client
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/client.cpp
)

target_link_libraries(client
    PRIVATE
        hip::host
)