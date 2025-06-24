cmake_minimum_required(VERSION 3.10)
project(proxy_server)

# libtorch 경로 지정
set(CMAKE_PREFIX_PATH "/libtorch")

# C++17 필수
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 소스 추가
add_executable(proxy proxy.cpp)

# libtorch 찾기
find_package(Torch REQUIRED)

# 링크
target_link_libraries(proxy "${TORCH_LIBRARIES}")
set_property(TARGET proxy PROPERTY CXX_STANDARD 17)
