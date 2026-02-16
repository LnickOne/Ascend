#!/bin/bash
# AscendC HelloWorld 编译与运行
# 需已安装 CANN toolkit 并先 source 环境：source /usr/local/Ascend/cann/set_env.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 编译时 bisheng 需要找到 C++ 标准库头文件（cstdint、vector 等）
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"

echo "== 加载 CANN 环境 =="
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [ -f "$HOME/Ascend/cann/set_env.sh" ]; then
    source "$HOME/Ascend/cann/set_env.sh"
else
    echo "错误: 未找到 set_env.sh，请先安装 CANN 并设置环境"
    exit 1
fi

echo "== 编译（核函数 hello_world.asc + 主程序 main.cpp） =="
rm -rf build && mkdir -p build && cd build
cmake .. && make -j

echo ""
echo "== 运行（需有 NPU 设备，无 NPU 时会在 kernel launch 报错） =="
./demo
