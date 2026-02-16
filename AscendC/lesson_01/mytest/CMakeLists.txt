# AscendC HelloWorld
# 编译前：source /usr/local/Ascend/cann/set_env.sh
# 并设置：export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"
# 或直接执行：bash run.sh
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(lesson_01_helloworld LANGUAGES ASC CXX)

# 核函数在 hello_world.asc，主程序在 main.cpp
add_executable(demo hello_world.asc main.cpp)

# 主机端/设备侧编译需要 C++ 标准库头文件；main.cpp 需要 CANN 的 acl 头文件
target_include_directories(demo PRIVATE
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/include
    /usr/include/c++/11
    /usr/include/x86_64-linux-gnu/c++/11
)
# NPU 架构：dav-2201(Atlas A2/A3)、dav-m200-vec(310P 矢量核)、dav-c220-vec 等，按实际设备修改
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# 链接：CANN 运行时库 + libstdc++
target_link_directories(demo PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    /usr/lib/gcc/x86_64-linux-gnu/11
)
target_link_libraries(demo PRIVATE
    ascendc_runtime ascendcl runtime register error_manager profapi
    ge_common_base ascendalog mmpa dl ascend_dump c_sec
    stdc++
)
