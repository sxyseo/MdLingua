@echo off
setlocal enabledelayedexpansion

REM Markdown/MDX文件翻译工具启动脚本
REM 用法: 
REM   translate.bat 源文件夹 目标文件夹 [LLM提供商] [模型名称] [批处理大小] [延迟秒数] [源语言] [目标语言]
REM   translate.bat --file 源文件 目标文件夹 [LLM提供商] [模型名称] [延迟秒数] [源语言] [目标语言]

REM 检查参数
if "%~1"=="" (
    goto :usage
)

REM 判断是否是文件模式
if "%~1"=="--file" (
    if "%~2"=="" (
        goto :usage
    )
    if "%~3"=="" (
        goto :usage
    )
    goto :file_mode
)

REM 文件夹模式
if "%~2"=="" (
    goto :usage
)
goto :dir_mode

:usage
echo 用法: 
echo   translate.bat 源文件夹 目标文件夹 [LLM提供商] [模型名称] [批处理大小] [延迟秒数] [源语言] [目标语言]
echo   translate.bat --file 源文件 目标文件夹 [LLM提供商] [模型名称] [延迟秒数] [源语言] [目标语言]
echo.
echo 参数:
echo   源文件夹/源文件 - 包含需要翻译的Markdown/MDX文件的文件夹或单个Markdown/MDX文件
echo   目标文件夹      - 保存翻译后的Markdown/MDX文件的文件夹
echo   LLM提供商       - 可选参数, 默认为anthropic, 可选值: openai, anthropic, azure, deepseek, gemini, local, siliconflow
echo   模型名称        - 可选参数, 如果未指定则使用提供商的默认模型
echo   批处理大小      - 可选参数, 同时处理的文件数, 默认为5 (仅文件夹模式有效)
echo   延迟秒数        - 可选参数, 每个文件处理后的延迟时间(秒), 默认为3秒
echo   源语言          - 可选参数, 源文件语言, 默认为auto(自动检测), 可选如zh(中文),en(英文),ja(日语)等
echo   目标语言        - 可选参数, 目标文件语言, 默认为en(英文), 可选如zh(中文),fr(法语),de(德语)等
echo.
echo 兼容旧版本:
echo   也支持使用--direction参数指定翻译方向(zh2en或en2zh)
exit /b 1

:file_mode
REM 设置默认值和命令行参数
set "SOURCE_FILE=%~2"
set "TARGET=%~3"
set "PROVIDER=anthropic"
set "MODEL="
set "DELAY=3"
set "SOURCE_LANG=auto"
set "TARGET_LANG=en"

if not "%~4"=="" (
    set "PROVIDER=%~4"
)
if not "%~5"=="" (
    set "MODEL=%~5"
)
if not "%~6"=="" (
    set "DELAY=%~6"
)
if not "%~7"=="" (
    set "SOURCE_LANG=%~7"
)
if not "%~8"=="" (
    set "TARGET_LANG=%~8"
)

echo 开始翻译单个文件...
echo 源文件: %SOURCE_FILE%
echo 目标文件夹: %TARGET%
echo LLM提供商: %PROVIDER%
if not "%MODEL%"=="" (
    echo 模型: %MODEL%
)
echo 延迟时间: %DELAY% 秒
echo 源语言: %SOURCE_LANG%
echo 目标语言: %TARGET_LANG%
echo.

REM 获取脚本所在目录
set "SCRIPT_DIR=%~dp0"

REM 构建命令行
set "CMD=python %SCRIPT_DIR%translate_md.py --file "%SOURCE_FILE%" --target "%TARGET%" --provider "%PROVIDER%" --delay %DELAY% --source-lang %SOURCE_LANG% --target-lang %TARGET_LANG%"

REM 如果指定了模型，添加模型参数
if not "%MODEL%"=="" (
    set "CMD=!CMD! --model "%MODEL%""
)

REM 执行翻译脚本
echo 执行命令: %CMD%
%CMD%

if %ERRORLEVEL% neq 0 (
    echo 翻译过程中出现错误。
    exit /b 1
) else (
    echo 翻译完成！结果保存在 %TARGET%
)

exit /b 0

:dir_mode
REM 设置默认值和命令行参数
set "SOURCE=%~1"
set "TARGET=%~2"
set "PROVIDER=anthropic"
set "MODEL="
set "BATCH=5"
set "DELAY=3"
set "SOURCE_LANG=auto"
set "TARGET_LANG=en"

if not "%~3"=="" (
    set "PROVIDER=%~3"
)
if not "%~4"=="" (
    set "MODEL=%~4"
)
if not "%~5"=="" (
    set "BATCH=%~5"
)
if not "%~6"=="" (
    set "DELAY=%~6"
)
if not "%~7"=="" (
    set "SOURCE_LANG=%~7"
)
if not "%~8"=="" (
    set "TARGET_LANG=%~8"
)

echo 开始翻译...
echo 源文件夹: %SOURCE%
echo 目标文件夹: %TARGET%
echo LLM提供商: %PROVIDER%
if not "%MODEL%"=="" (
    echo 模型: %MODEL%
)
echo 批处理大小: %BATCH%
echo 延迟时间: %DELAY% 秒
echo 源语言: %SOURCE_LANG%
echo 目标语言: %TARGET_LANG%
echo.

REM 获取脚本所在目录
set "SCRIPT_DIR=%~dp0"

REM 构建命令行
set "CMD=python %SCRIPT_DIR%translate_md.py --source "%SOURCE%" --target "%TARGET%" --provider "%PROVIDER%" --batch %BATCH% --delay %DELAY% --source-lang %SOURCE_LANG% --target-lang %TARGET_LANG%"

REM 如果指定了模型，添加模型参数
if not "%MODEL%"=="" (
    set "CMD=!CMD! --model "%MODEL%""
)

REM 执行翻译脚本
echo 执行命令: %CMD%
%CMD%

if %ERRORLEVEL% neq 0 (
    echo 翻译过程中出现错误。
    exit /b 1
) else (
    echo 翻译完成！结果保存在 %TARGET%
)

exit /b 0 