cmake_minimum_required(VERSION 3.0)

project(test1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "E:/OpenCV/opencv_build/install/x64/mingw/lib")
find_package(OpenCV REQUIRED)

add_executable(AHE main.cpp)
target_link_libraries(AHE PRIVATE ${OpenCV_LIBS})

install(TARGETS AHE
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

