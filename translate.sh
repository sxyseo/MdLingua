#!/bin/bash
# Markdown/MDX文件翻译工具启动脚本
# 用法: 
#   translate.sh 源文件夹 目标文件夹 [LLM提供商] [模型名称] [批处理大小] [延迟秒数] [源语言] [目标语言]
#   translate.sh --file 源文件 目标文件夹 [LLM提供商] [模型名称] [延迟秒数] [源语言] [目标语言]

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 找不到Python3。请确保Python3已安装。"
    exit 1
fi

# 显示使用帮助
show_usage() {
    echo "用法: "
    echo "  translate.sh 源文件夹 目标文件夹 [LLM提供商] [模型名称] [批处理大小] [延迟秒数] [源语言] [目标语言]"
    echo "  translate.sh --file 源文件 目标文件夹 [LLM提供商] [模型名称] [延迟秒数] [源语言] [目标语言]"
    echo ""
    echo "参数:"
    echo "  源文件夹/源文件 - 包含需要翻译的Markdown/MDX文件的文件夹或单个Markdown/MDX文件"
    echo "  目标文件夹      - 保存翻译后的Markdown/MDX文件的文件夹"
    echo "  LLM提供商       - 可选参数, 默认为anthropic, 可选值: openai, anthropic, azure, deepseek, gemini, local, siliconflow"
    echo "  模型名称        - 可选参数, 如果未指定则使用提供商的默认模型"
    echo "  批处理大小      - 可选参数, 同时处理的文件数, 默认为5 (仅文件夹模式有效)"
    echo "  延迟秒数        - 可选参数, 每个文件处理后的延迟时间(秒), 默认为3秒"
    echo "  源语言          - 可选参数, 源文件语言, 默认为auto(自动检测), 可选如zh(中文),en(英文),ja(日语)等"
    echo "  目标语言        - 可选参数, 目标文件语言, 默认为en(英文), 可选如zh(中文),fr(法语),de(德语)等"
    echo ""
    echo "兼容旧版本:"
    echo "  也支持使用--direction参数指定翻译方向(zh2en或en2zh)"
    exit 1
}

# 参数检查
if [ $# -lt 1 ]; then
    show_usage
fi

# 判断是否是文件模式
if [ "$1" = "--file" ]; then
    if [ $# -lt 3 ]; then
        show_usage
    fi
    
    # 设置单文件模式参数
    SOURCE_FILE="$2"
    TARGET="$3"
    PROVIDER="anthropic"
    MODEL=""
    DELAY=3
    SOURCE_LANG="auto"
    TARGET_LANG="en"
    
    # 根据提供的参数覆盖默认值
    if [ $# -ge 4 ]; then PROVIDER="$4"; fi
    if [ $# -ge 5 ]; then MODEL="$5"; fi
    if [ $# -ge 6 ]; then DELAY="$6"; fi
    if [ $# -ge 7 ]; then SOURCE_LANG="$7"; fi
    if [ $# -ge 8 ]; then TARGET_LANG="$8"; fi
    
    echo "开始翻译单个文件..."
    echo "源文件: $SOURCE_FILE"
    echo "目标文件夹: $TARGET"
    echo "LLM提供商: $PROVIDER"
    if [ -n "$MODEL" ]; then
        echo "模型: $MODEL"
    fi
    echo "延迟时间: $DELAY 秒"
    echo "源语言: $SOURCE_LANG"
    echo "目标语言: $TARGET_LANG"
    echo ""
    
    # 获取脚本所在目录
    SCRIPT_DIR="$(dirname "$0")"
    
    # 构建命令行
    CMD="python3 $SCRIPT_DIR/translate_md.py --file \"$SOURCE_FILE\" --target \"$TARGET\" --provider \"$PROVIDER\" --delay $DELAY --source-lang $SOURCE_LANG --target-lang $TARGET_LANG"
    
    # 如果指定了模型，添加模型参数
    if [ -n "$MODEL" ]; then
        CMD="$CMD --model \"$MODEL\""
    fi
    
else
    # 目录模式
    if [ $# -lt 2 ]; then
        show_usage
    fi
    
    # 设置目录模式参数
    SOURCE="$1"
    TARGET="$2"
    PROVIDER="anthropic"
    MODEL=""
    BATCH=5
    DELAY=3
    SOURCE_LANG="auto"
    TARGET_LANG="en"
    
    # 根据提供的参数覆盖默认值
    if [ $# -ge 3 ]; then PROVIDER="$3"; fi
    if [ $# -ge 4 ]; then MODEL="$4"; fi
    if [ $# -ge 5 ]; then BATCH="$5"; fi
    if [ $# -ge 6 ]; then DELAY="$6"; fi
    if [ $# -ge 7 ]; then SOURCE_LANG="$7"; fi
    if [ $# -ge 8 ]; then TARGET_LANG="$8"; fi
    
    echo "开始翻译..."
    echo "源文件夹: $SOURCE"
    echo "目标文件夹: $TARGET"
    echo "LLM提供商: $PROVIDER"
    if [ -n "$MODEL" ]; then
        echo "模型: $MODEL"
    fi
    echo "批处理大小: $BATCH"
    echo "延迟时间: $DELAY 秒"
    echo "源语言: $SOURCE_LANG"
    echo "目标语言: $TARGET_LANG"
    echo ""
    
    # 获取脚本所在目录
    SCRIPT_DIR="$(dirname "$0")"
    
    # 构建命令行
    CMD="python3 $SCRIPT_DIR/translate_md.py --source \"$SOURCE\" --target \"$TARGET\" --provider \"$PROVIDER\" --batch $BATCH --delay $DELAY --source-lang $SOURCE_LANG --target-lang $TARGET_LANG"
    
    # 如果指定了模型，添加模型参数
    if [ -n "$MODEL" ]; then
        CMD="$CMD --model \"$MODEL\""
    fi
fi

# 执行翻译脚本
echo "执行命令: $CMD"
eval $CMD

if [ $? -ne 0 ]; then
    echo "翻译过程中出现错误。"
    exit 1
else
    echo "翻译完成！结果保存在 $TARGET"
fi

exit 0 