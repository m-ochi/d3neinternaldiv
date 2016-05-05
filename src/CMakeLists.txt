cmake_minimum_required(VERSION 2.8)
project(D3NE_dev)

find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -lpthread -Ofast -ffast-math -pthread -ffast-math -Wunused-result -Wint-to-pointer-cast -g -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG  "-Wall -O0 -g") 

set(SOURCE_FILES
    pteself/src/cmdline/cmdline.h
    pteself/src/aliasmethod/aliastable.cpp
    pteself/src/aliasmethod/aliastable.h
    train.cpp
    pteself/src/pte.cpp
    pteself/src/pte.h
    pteself/src/constantsettings.h
    pteself/src/constantsettings.cpp
    pteself/src/network.cpp
    pteself/src/network.h
    pteself/src/vertex.cpp
    pteself/src/vertex.h
    pteself/src/edge.cpp
    pteself/src/edge.h
    pteself/src/embeddedvector.cpp
    pteself/src/embeddedvector.h
    pteself/src/samplers.cpp
    pteself/src/samplers.h
    pteself/src/networkall.cpp
    pteself/src/networkall.h
    pteself/src/sigmoid.cpp
    pteself/src/sigmoid.h 
    d3ne.cpp
    d3ne.h)

add_executable(train_d3ne ${SOURCE_FILES})
target_link_libraries( train_d3ne ${CMAKE_THREAD_LIBS_INIT} )

