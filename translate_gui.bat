@echo off
REM Markdown/MDX文件翻译工具GUI启动脚本

REM 获取脚本所在目录
set "SCRIPT_DIR=%~dp0"

echo 正在启动Markdown/MDX文件翻译工具GUI界面...

REM 执行GUI启动脚本
python "%SCRIPT_DIR%translate_gui.py"

if %ERRORLEVEL% neq 0 (
    echo 启动GUI界面时出现错误。
    echo 请确认已安装必要的依赖: pip install tkinterdnd2
    pause
    exit /b 1
)

exit /b 0 