/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * 说明：acl/acl.h 来自 CANN 安装目录，不在样例仓库中。
 * 路径示例：/usr/local/Ascend/cann-8.5.0/x86_64-linux/include/acl/acl.h
 * 编译时 CMake 会通过 -I 传入该路径；若 IDE 报红，可配置 .vscode/c_cpp_properties.json 的 includePath。
 */
#include "acl/acl.h"                                        // 包含ACL库头文件，用于使用ACL框架的API
extern void hello_world_do(uint32_t coreDim, void *stream); // 声明外部函数，该函数在其他文件中定义

int32_t main(int argc, char const *argv[])
{
    aclInit(nullptr);             // 初始化ACL库，nullptr表示使用默认配置
    int32_t deviceId = 0;         // 指定使用的设备ID，昇腾设备通常从0开始编号
    aclrtSetDevice(deviceId);     // 绑定当前进程到指定设备（deviceId=0）
    aclrtStream stream = nullptr; // 定义流对象，用于管理任务执行顺序
    aclrtCreateStream(&stream);   // 创建流，后续任务通过此流提交执行

    // 8 indicates the kernel function will be executed on eight cores.
    constexpr uint32_t blockDim = 8;  // 定义核函数执行的核心数为8
    hello_world_do(blockDim, stream); // 调用核函数，传入核心数和流对象
    aclrtSynchronizeStream(stream);   // 同步流，等待流中所有任务执行完成

    aclrtDestroyStream(stream); // 销毁流，释放流资源
    aclrtResetDevice(deviceId); // 重置设备，释放设备上的资源
    aclFinalize();              // 反初始化ACL库，释放ACL相关资源
    return 0;                   // 程序正常退出
}