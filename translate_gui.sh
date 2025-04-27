#!/bin/bash
# Markdown/MDX文件翻译工具GUI启动脚本 (PySide6版本)

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 找不到Python3。请确保Python3已安装。"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

echo "正在启动Markdown/MDX文件翻译工具GUI界面(PySide6版本)..."

# 设置必要的环境变量
export TK_SILENCE_DEPRECATION=1
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
# export MOCK_TRANSLATION=true
export QT_MAC_WANTS_LAYER=1 # 修复macOS上的某些Qt渲染问题
# 优先使用当前目录的llm_api.py，然后才是tools目录的
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH:$SCRIPT_DIR/tools" 

echo "当前工作目录: $(pwd)"
echo "Python路径: $PYTHONPATH"
echo "检查llm_api.py文件: "
if [ -f "llm_api.py" ]; then
    echo "✓ 当前目录存在llm_api.py"
else
    echo "✗ 当前目录不存在llm_api.py"
    if [ -f "tools/llm_api.py" ]; then
        echo "✓ tools目录存在llm_api.py"
    else
        echo "✗ tools目录不存在llm_api.py"
        echo "错误: 无法找到llm_api.py文件，翻译功能可能无法正常工作。"
    fi
fi

# 检查并激活虚拟环境
if [ -d ".venv" ]; then
    echo "发现虚拟环境，正在激活..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "警告: 找不到虚拟环境激活脚本，可能导致依赖问题"
    fi
else
    echo "警告: 未找到虚拟环境，将使用系统Python"
fi

# 检查并安装必要的依赖
echo "检查必要的依赖..."
# 先检查一下依赖是否已安装
MISSING_DEPS=0
python3 -c "import requests" 2>/dev/null || MISSING_DEPS=1
python3 -c "import google.generativeai" 2>/dev/null || MISSING_DEPS=1
python3 -c "import anthropic" 2>/dev/null || MISSING_DEPS=1
python3 -c "import openai" 2>/dev/null || MISSING_DEPS=1

if [ $MISSING_DEPS -eq 1 ]; then
    echo "检测到缺少依赖，正在尝试安装..."
    python3 -m pip install -r requirements.txt
else
    echo "所有依赖已安装，跳过安装步骤"
fi

# 执行GUI启动脚本
echo "启动翻译GUI界面..."
# 明确指定Python导入路径，确保能找到当前目录的llm_api.py
PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" python3 "$SCRIPT_DIR/translate_gui.py"

if [ $? -ne 0 ]; then
    echo "======================="
    echo "启动GUI界面时出现错误。"
    echo "======================="
    echo "错误详情:"
    # 尝试执行导入测试，首先测试当前目录下的llm_api
    python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); import llm_api; print('✓ 可以成功导入当前目录的llm_api模块')" || 
    python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/tools'); from tools import llm_api; print('✓ 可以成功导入tools目录的llm_api模块')" || 
    echo "✗ 无法导入任何位置的llm_api模块"
    
    # 检查依赖
    echo "\n依赖检查:"
    python3 -c "import requests; print('✓ requests依赖已安装')" || echo "✗ 缺少requests依赖"
    python3 -c "import google.generativeai; print('✓ google.generativeai依赖已安装')" || echo "✗ 缺少google.generativeai依赖"
    python3 -c "import anthropic; print('✓ anthropic依赖已安装')" || echo "✗ 缺少anthropic依赖"
    python3 -c "import openai; print('✓ openai依赖已安装')" || echo "✗ 缺少openai依赖"
    
    echo "\n解决方法:"
    echo "1. 确保llm_api.py文件存在于当前目录或tools目录"
    echo "2. 确认已安装必要的依赖: pip install requests google-generativeai anthropic openai"
    echo "3. 如果使用虚拟环境，请确认已激活正确的环境"
    
    read -p "按任意键继续..."
    exit 1
fi

exit 0
