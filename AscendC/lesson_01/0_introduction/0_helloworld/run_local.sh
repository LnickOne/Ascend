#!/bin/bash
# 在 lesson_01/0_introduction/0_helloworld 下执行：bash run_local.sh
# 说明：若项目路径含中文，CANN 生成的 CMake 会报错，故在 /tmp 下构建后运行。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/tmp/helloworld_$(whoami)_$$"
SOC_VERSION="${1:-Ascend310P3}"

# 加载 CANN 环境（优先 cann，其次 ascend-toolkit）
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
    CANN_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}"
elif [ -n "$ASCEND_HOME_PATH" ] && [ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]; then
    source "$ASCEND_HOME_PATH/bin/setenv.bash"
    CANN_PATH="$ASCEND_HOME_PATH"
else
    echo "[ERROR] 未找到 CANN 环境，请先安装并 source set_env.sh"
    exit 1
fi

export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"

echo "[INFO] CANN_PATH=$CANN_PATH  SOC_VERSION=$SOC_VERSION"
echo "[INFO] 构建目录（避免中文路径）: $BUILD_DIR"

cp -r "$SCRIPT_DIR" "$BUILD_DIR"
cd "$BUILD_DIR"
rm -rf build out
mkdir -p build
cmake -S . -B build -DSOC_VERSION="$SOC_VERSION" -DASCEND_CANN_PACKAGE_PATH="$CANN_PATH"
cmake --build build -j
cmake --install build 2>/dev/null || true

echo "[INFO] 运行 main（无 NPU 时 kernel 会报错，属正常）"
"$BUILD_DIR/build/main" 2>&1 || true
echo "[INFO] 构建产物在: $BUILD_DIR/build/main"
