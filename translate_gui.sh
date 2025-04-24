#!/bin/bash
# Markdown/MDX文件翻译工具GUI启动脚本

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 找不到Python3。请确保Python3已安装。"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(dirname "$0")"

echo "正在启动Markdown/MDX文件翻译工具GUI界面..."

# 执行GUI启动脚本
python3 "$SCRIPT_DIR/translate_gui.py"

if [ $? -ne 0 ]; then
    echo "启动GUI界面时出现错误。"
    echo "请确认已安装必要的依赖: pip install tkinterdnd2"
    read -p "按任意键继续..."
    exit 1
fi

exit 0 