#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM API工具 - 提供对各种LLM服务的访问

这个模块提供了对各种LLM服务的访问，包括OpenAI、Anthropic、Azure OpenAI、Google Gemini等。
主要用于翻译工具和其他需要LLM服务的工具。
"""

import os
import sys
import json
import argparse
import requests
import logging
import mimetypes
import base64
import time
import httpx
from typing import Optional, Dict, Any, List, Union, NamedTuple
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("llm_api")

class TokenUsage(NamedTuple):
    """跟踪令牌使用情况的数据类"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None

class APIResponse(NamedTuple):
    """API响应的数据类"""
    content: str
    token_usage: TokenUsage
    cost: float
    thinking_time: float
    provider: str
    model: str

class TokenTracker:
    """跟踪API调用和令牌使用情况"""
    
    def __init__(self):
        self.requests = []
        self.total_cost = 0.0
        self.total_tokens = 0
        
    def track_request(self, response: APIResponse):
        """记录API请求"""
        self.requests.append(response)
        self.total_cost += response.cost
        self.total_tokens += response.token_usage.total_tokens
        
        # 记录详细信息
        logger.info(f"LLM请求: {response.provider}/{response.model}")
        logger.info(f"令牌使用: 提示={response.token_usage.prompt_tokens}, "
                    f"完成={response.token_usage.completion_tokens}, "
                    f"总计={response.token_usage.total_tokens}")
        if response.token_usage.reasoning_tokens:
            logger.info(f"推理令牌: {response.token_usage.reasoning_tokens}")
        logger.info(f"成本: ${response.cost:.6f}, 思考时间: {response.thinking_time:.2f}秒")
        
    def calculate_openai_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """计算OpenAI API调用的成本"""
        # 2024年5月的价格
        if model == "gpt-4o":
            return (prompt_tokens * 5 / 1000000) + (completion_tokens * 15 / 1000000)
        elif model == "gpt-4o-mini":
            return (prompt_tokens * 0.5 / 1000000) + (completion_tokens * 1.5 / 1000000)
        elif model == "o1":
            return (prompt_tokens * 15 / 1000000) + (completion_tokens * 75 / 1000000)
        elif model == "gpt-4-turbo":
            return (prompt_tokens * 10 / 1000000) + (completion_tokens * 30 / 1000000)
        elif model == "gpt-3.5-turbo":
            return (prompt_tokens * 0.5 / 1000000) + (completion_tokens * 1.5 / 1000000)
        # 默认价格（如果模型未知）
        return (prompt_tokens * 5 / 1000000) + (completion_tokens * 15 / 1000000)
    
    def calculate_claude_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """计算Claude API调用的成本"""
        # 2024年5月的价格
        if "claude-3-5-sonnet" in model:
            return (prompt_tokens * 3 / 1000000) + (completion_tokens * 15 / 1000000)
        elif "claude-3-opus" in model:
            return (prompt_tokens * 15 / 1000000) + (completion_tokens * 75 / 1000000)
        elif "claude-3-sonnet" in model:
            return (prompt_tokens * 3 / 1000000) + (completion_tokens * 15 / 1000000)
        elif "claude-3-haiku" in model:
            return (prompt_tokens * 0.25 / 1000000) + (completion_tokens * 1.25 / 1000000)
        elif "claude-2" in model:
            return (prompt_tokens * 8 / 1000000) + (completion_tokens * 24 / 1000000)
        # 默认价格（如果模型未知）
        return (prompt_tokens * 3 / 1000000) + (completion_tokens * 15 / 1000000)

# 创建一个全局令牌跟踪器实例
_token_tracker = TokenTracker()

def get_token_tracker() -> TokenTracker:
    """获取全局令牌跟踪器实例"""
    return _token_tracker

# 加载环境变量（如果有.env文件）
def load_environment():
    """按优先顺序从.env文件加载环境变量"""
    # 优先顺序:
    # 1. 系统环境变量（已加载）
    # 2. .env.local（用户特定的覆盖）
    # 3. .env（项目默认值）
    # 4. .env.example（示例配置）
    
    try:
        from dotenv import load_dotenv
        
        env_files = ['.env.local', '.env', '.env.example']
        env_loaded = False
        
        print("当前工作目录:", Path('.').absolute(), file=sys.stderr)
        print("查找环境文件:", env_files, file=sys.stderr)
        
        for env_file in env_files:
            env_path = Path('.') / env_file
            logger.debug(f"检查 {env_path.absolute()}")
            if env_path.exists():
                logger.info(f"找到 {env_file}，正在加载变量...")
                load_dotenv(dotenv_path=env_path)
                env_loaded = True
                logger.info(f"从 {env_file} 加载了环境变量")
                # 打印加载的键（出于安全考虑不打印值）
                with open(env_path) as f:
                    keys = [line.split('=')[0].strip() for line in f if '=' in line and not line.startswith('#')]
                    logger.debug(f"从 {env_file} 加载的键: {keys}")
        
        if not env_loaded:
            logger.warning("警告: 未找到.env文件。仅使用系统环境变量。")
            logger.debug(f"可用的系统环境变量: {list(os.environ.keys())}")
            
    except ImportError:
        logger.warning("警告: 未安装python-dotenv。仅使用系统环境变量。")

# 在模块导入时加载环境变量
load_environment()

# API密钥（从环境变量获取）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://msopenai.openai.azure.com")
AZURE_OPENAI_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT", "gpt-4o-ms")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1")

# 默认模型
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "azure": AZURE_OPENAI_MODEL_DEPLOYMENT,
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
    "local": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "siliconflow": "mixtral-8x7b"
}

# 创建一个httpx客户端，用于处理代理
http_client = httpx.Client(
    timeout=60.0,
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
)

def encode_image_file(image_path: str) -> tuple[str, str]:
    """
    将图像文件编码为base64并确定其MIME类型
    
    Args:
        image_path (str): 图像文件的路径
        
    Returns:
        tuple: (base64编码字符串, MIME类型)
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = 'image/png'  # 如果无法确定类型，默认为PNG
        
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    return encoded_string, mime_type

def create_llm_client(provider="openai"):
    """
    创建LLM客户端
    
    Args:
        provider (str): LLM提供商名称
        
    Returns:
        客户端对象
    """
    if provider == "openai":
        try:
            from openai import OpenAI
            
            api_key = OPENAI_API_KEY
            if not api_key:
                raise ValueError("环境变量中未找到OPENAI_API_KEY")
            return OpenAI(api_key=api_key, http_client=http_client)
        except ImportError:
            logger.error("请安装OpenAI包: pip install openai")
            return None
            
    elif provider == "azure":
        try:
            from openai import AzureOpenAI
            
            api_key = AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("环境变量中未找到AZURE_OPENAI_API_KEY")
            return AzureOpenAI(
                api_key=api_key,
                api_version="2024-08-01-preview",
                azure_endpoint=AZURE_ENDPOINT,
                http_client=http_client
            )
        except ImportError:
            logger.error("请安装OpenAI包: pip install openai")
            return None
            
    elif provider == "deepseek":
        try:
            from openai import OpenAI
            
            api_key = DEEPSEEK_API_KEY
            if not api_key:
                raise ValueError("环境变量中未找到DEEPSEEK_API_KEY")
            return OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                http_client=http_client
            )
        except ImportError:
            logger.error("请安装OpenAI包: pip install openai")
            return None
            
    elif provider == "siliconflow":
        try:
            from openai import OpenAI
            
            api_key = SILICONFLOW_API_KEY
            if not api_key:
                raise ValueError("环境变量中未找到SILICONFLOW_API_KEY")
            return OpenAI(
                api_key=api_key,
                base_url=SILICONFLOW_API_URL,
                http_client=http_client
            )
        except ImportError:
            logger.error("请安装OpenAI包: pip install openai")
            return None
            
    elif provider == "anthropic":
        try:
            from anthropic import Anthropic
            
            api_key = ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError("环境变量中未找到ANTHROPIC_API_KEY")
            return Anthropic(api_key=api_key)
        except ImportError:
            logger.error("请安装Anthropic包: pip install anthropic")
            return None
            
    elif provider == "gemini":
        try:
            import google.generativeai as genai
            
            api_key = GOOGLE_API_KEY
            if not api_key:
                raise ValueError("环境变量中未找到GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            return genai
        except ImportError:
            logger.error("请安装Google GenerativeAI包: pip install google-generativeai")
            return None
            
    elif provider == "local":
        try:
            from openai import OpenAI
            
            return OpenAI(
                base_url=LOCAL_LLM_URL,
                api_key="not-needed",
                http_client=http_client
            )
        except ImportError:
            logger.error("请安装OpenAI包: pip install openai")
            return None
            
    else:
        logger.error(f"不支持的提供商: {provider}")
        return None

def simulate_translation(prompt: str) -> str:
    """
    模拟翻译功能，用于测试或演示
    
    Args:
        prompt: 翻译提示词
        
    Returns:
        模拟的翻译结果
    """
    logger.info("正在使用模拟翻译功能...")
    
    # 将提示词分割成行
    lines = prompt.strip().split("\n")
    
    # 找到原始内容的起始和结束
    start_index = -1
    end_index = -1
    
    # 识别翻译方向
    source_lang = "中文"
    target_lang = "英文"
    for i, line in enumerate(lines):
        if "翻译专家" in line:
            if "中文到英文" in line:
                source_lang = "中文"
                target_lang = "英文"
            elif "英文到中文" in line:
                source_lang = "英文"
                target_lang = "中文"
        
        if line.strip().endswith("文档：") or "文档:" in line:
            start_index = i + 1
        elif "翻译：" in line:
            end_index = i
            break
    
    # 如果找不到明确的分界线，尝试从中间分割
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        middle = len(lines) // 2
        start_index = middle // 2
        end_index = middle + middle // 2
    
    # 提取原始内容
    original_content = "\n".join(lines[start_index:end_index]).strip()
    
    # 根据翻译方向执行模拟翻译
    if source_lang == "中文" and target_lang == "英文":
        return simulate_zh_to_en(original_content)
    elif source_lang == "英文" and target_lang == "中文":
        return simulate_en_to_zh(original_content)
    else:
        # 默认行为，添加标记
        return f"[模拟翻译] {original_content}"

def simulate_zh_to_en(content: str) -> str:
    """模拟中文到英文的翻译"""
    # 字典映射常见中文词汇/短语到英文
    zh_to_en_dict = {
        "标题": "Title",
        "简介": "Introduction",
        "目录": "Table of Contents",
        "章节": "Chapter",
        "部分": "Section",
        "小节": "Subsection",
        "段落": "Paragraph",
        "示例": "Example",
        "注意": "Note",
        "警告": "Warning",
        "提示": "Tip",
        "总结": "Summary",
        "结论": "Conclusion",
        "参考文献": "References",
        "附录": "Appendix",
        "图表": "Figure",
        "表格": "Table",
        "代码": "Code",
        "安装": "Installation",
        "配置": "Configuration",
        "使用方法": "Usage",
        "快速开始": "Quick Start",
        "高级用法": "Advanced Usage",
        "入门指南": "Getting Started",
        "教程": "Tutorial",
        "开发者指南": "Developer Guide",
        "常见问题": "FAQ",
        "疑难解答": "Troubleshooting",
        "如何使用": "How to Use",
        "文件结构": "File Structure",
        "项目结构": "Project Structure",
        "主要功能": "Key Features",
        "什么是": "What is",
        "为什么": "Why",
        "如何": "How to",
        "什么时候": "When to",
        "在哪里": "Where to",
        "谁": "Who",
        "详细说明": "Details",
        "更多信息": "More Information",
        "相关链接": "Related Links",
        "依赖": "Dependencies",
        "需求": "Requirements",
        "支持": "Support",
        "贡献": "Contributing",
        "许可证": "License",
        "版权": "Copyright",
        "作者": "Author",
        "联系方式": "Contact",
        "感谢": "Acknowledgments",
        "翻译": "Translation",
        "工具": "Tool",
        "应用": "Application",
        "框架": "Framework",
        "库": "Library",
        "插件": "Plugin",
        "扩展": "Extension",
        "服务": "Service",
        "命令行": "Command Line",
        "界面": "Interface",
        "源代码": "Source Code",
        "构建": "Build",
        "部署": "Deployment",
        "测试": "Test",
        "调试": "Debug",
        "执行": "Execute",
        "运行": "Run",
        "更新": "Update",
        "升级": "Upgrade",
        "下载": "Download",
        "上传": "Upload",
        "导入": "Import",
        "导出": "Export",
        "备份": "Backup",
        "恢复": "Restore",
        "开始": "Start",
        "停止": "Stop",
        "暂停": "Pause",
        "继续": "Continue",
        "重启": "Restart",
        "输入": "Input",
        "输出": "Output",
        "参数": "Parameter",
        "选项": "Option",
        "默认值": "Default Value",
        "可选": "Optional",
        "必须": "Required",
        "环境变量": "Environment Variable",
        "配置文件": "Configuration File",
        "日志": "Log",
        "错误": "Error",
        "警告": "Warning",
        "信息": "Information",
        "调试信息": "Debug Information",
        "异常": "Exception",
        "问题": "Issue",
        "解决方案": "Solution",
        "功能": "Feature",
        "版本": "Version",
        "发布": "Release",
        "更新日志": "Changelog",
        "路线图": "Roadmap",
        "实现": "Implementation",
        "设计": "Design",
        "架构": "Architecture",
        "模式": "Pattern",
        "接口": "Interface",
        "类": "Class",
        "方法": "Method",
        "函数": "Function",
        "变量": "Variable",
        "常量": "Constant",
        "属性": "Property",
        "对象": "Object",
        "实例": "Instance",
        "中文": "Chinese",
        "英文": "English",
        "翻译": "Translation",
        # 增加常见短语和句型
        "这是": "This is",
        "这个": "This",
        "一个": "a",
        "在这里": "Here",
        "可以": "can",
        "我们": "We",
        "你可以": "You can",
        "请": "Please",
        "使用": "use",
        "查看": "view",
        "编辑": "edit",
        "创建": "create",
        "删除": "delete",
        "添加": "add",
        "修改": "modify",
        "文件": "file",
        "文档": "document",
        "页面": "page",
        "网站": "website",
        "程序": "program",
        "软件": "software",
        "系统": "system",
        "语言": "language",
        "学习": "learn",
        "开发": "develop",
        "设置": "settings",
        "选择": "select",
        "点击": "click",
        "按钮": "button",
        "链接": "link",
        "菜单": "menu",
        "选项": "option",
        "首页": "home page",
        "关于": "about",
        "联系": "contact",
        "帮助": "help",
        "说明": "instructions",
        "指南": "guide",
        "示例": "example",
        "示范": "demonstration",
        "演示": "presentation",
        "项目": "project",
        "完成": "complete",
        "结束": "end",
        "开始": "start",
        "继续": "continue",
        "返回": "return",
        "提交": "submit",
        "保存": "save",
        "加载": "load",
        "导出": "export",
        "导入": "import",
        "上传": "upload",
        "下载": "download",
        "登录": "login",
        "注册": "register",
        "用户": "user",
        "密码": "password",
        "账户": "account",
        "管理": "manage",
        "控制": "control",
        "设定": "set",
        "自定义": "customize",
        "预设": "preset",
        "默认": "default",
        "显示": "display",
        "隐藏": "hide",
        "打开": "open",
        "关闭": "close",
        "启用": "enable",
        "禁用": "disable",
        "支持": "support",
    }
    
    # 常用句式替换模式
    sentence_patterns = [
        (r'这是一个(.*?)的(.*?)', r'This is a \2 that \1'),
        (r'这是(.*?)的(.*?)', r'This is the \2 of \1'),
        (r'(.*?)是一个(.*?)', r'\1 is a \2'),
        (r'(.*?)可以(.*?)', r'\1 can \2'),
        (r'如何(.*?)\?', r'How to \1?'),
        (r'为什么(.*?)\?', r'Why \1?'),
        (r'什么是(.*?)\?', r'What is \1?'),
        (r'(.*?)是什么\?', r'What is \1?'),
        (r'(.*?)在哪里\?', r'Where is \1?'),
        (r'如果你想(.*?)，你可以(.*?)', r'If you want to \1, you can \2'),
        (r'请确保(.*?)', r'Please make sure to \1'),
        (r'你需要(.*?)', r'You need to \1'),
        (r'我们建议(.*?)', r'We recommend \1'),
        (r'首先，(.*?)，然后(.*?)', r'First, \1, then \2'),
    ]
    
    # 分割内容为行处理
    lines = content.split('\n')
    translated_lines = []
    
    for line in lines:
        # 检测特殊格式行（代码块、标题等）
        if line.strip().startswith("```") or line.strip() == "":
            # 保持代码块标记和空行不变
            translated_lines.append(line)
            continue
            
        if line.strip().startswith("#"):
            # 处理标题行
            title_parts = line.split(' ', 1)
            if len(title_parts) > 1:
                heading = title_parts[0]
                text = title_parts[1]
                # 翻译标题文本
                for zh, en in zh_to_en_dict.items():
                    text = text.replace(zh, en)
                translated_lines.append(f"{heading} {text}")
            else:
                translated_lines.append(line)
            continue
        
        # 普通文本行的处理
        translated_line = line
        
        # 1. 先应用短语替换
        for zh, en in zh_to_en_dict.items():
            translated_line = translated_line.replace(zh, en)
        
        # 2. 如果没有发生任何变化，尝试应用句式模式
        if translated_line == line and line.strip():
            for pattern, replacement in sentence_patterns:
                import re
                if re.search(pattern, translated_line):
                    translated_line = re.sub(pattern, replacement, translated_line)
                    break
        
        # 3. 如果还是没有变化，添加英文标记
        if translated_line == line and line.strip():
            # 模拟完全翻译
            words = line.split()
            if len(words) > 0:
                # 添加英文风格的随机翻译
                translated_line = f"This is the English translation of the Chinese text: '{line}'"
            
        translated_lines.append(translated_line)
    
    # 模拟API延迟
    time.sleep(0.5)
    
    return "\n".join(translated_lines)

def simulate_en_to_zh(content: str) -> str:
    """模拟英文到中文的翻译"""
    # 与上面的字典相反
    en_to_zh_dict = {}
    zh_to_en_dict = {
        "标题": "Title",
        "简介": "Introduction",
        "目录": "Table of Contents",
        # ... 其他词汇映射
    }
    
    # 创建反向字典
    for zh, en in zh_to_en_dict.items():
        en_to_zh_dict[en] = zh
    
    # 分割内容为行处理
    lines = content.split('\n')
    translated_lines = []
    
    for line in lines:
        # 检测特殊格式行（代码块、标题等）
        if line.strip().startswith("```") or line.strip() == "":
            # 保持代码块标记和空行不变
            translated_lines.append(line)
            continue
            
        if line.strip().startswith("#"):
            # 处理标题行
            title_parts = line.split(' ', 1)
            if len(title_parts) > 1:
                heading = title_parts[0]
                text = title_parts[1]
                # 翻译标题文本
                for en, zh in en_to_zh_dict.items():
                    text = text.replace(en, zh)
                translated_lines.append(f"{heading} {text}")
            else:
                translated_lines.append(line)
            continue
                
        # 普通文本行
        translated_line = line
        for en, zh in en_to_zh_dict.items():
            translated_line = translated_line.replace(en, zh)
        
        # 如果没有任何词被翻译，添加一个中文前缀
        if translated_line == line and line.strip():
            translated_line = f"[中文] {line}"
            
        translated_lines.append(translated_line)
    
    # 模拟API延迟
    time.sleep(0.5)
    
    return "\n".join(translated_lines)

def query_llm(prompt: str, provider: str = "anthropic", model: Optional[str] = None, image_path: Optional[str] = None) -> str:
    """
    向指定的LLM提供商发送查询
    
    Args:
        prompt: 提示词
        provider: LLM提供商 (openai, anthropic, azure, deepseek, gemini, local, siliconflow)
        model: 要使用的模型（如果为None，将使用提供商的默认模型）
        image_path: 可选的图像文件路径（用于支持图像的模型）
        
    Returns:
        模型响应
    """
    # 检查是否开启了模拟翻译模式
    if os.getenv("MOCK_TRANSLATION", "").lower() == "true":
        logger.info(f"模拟翻译模式已启用，不调用实际API")
        return simulate_translation(prompt)
    
    # 其余代码保持不变
    start_time = time.time()
    
    # 如果未指定模型，使用默认模型
    if model is None:
        model = DEFAULT_MODELS.get(provider, "")
    
    try:
        # 针对不同提供商的API调用
        if provider == "openai":
            response = query_openai(prompt, model, image_path)
        elif provider == "anthropic":
            response = query_anthropic(prompt, model, image_path)
        elif provider == "azure":
            response = query_azure_openai(prompt, model, image_path)
        elif provider == "deepseek":
            response = query_deepseek(prompt, model)
        elif provider == "gemini":
            response = query_gemini(prompt, model, image_path)
        elif provider == "local":
            response = query_local_llm(prompt, model, image_path)
        elif provider == "siliconflow":
            response = query_siliconflow(prompt, model)
        else:
            raise ValueError(f"不支持的提供商: {provider}")
            
        thinking_time = time.time() - start_time
        logger.debug(f"LLM思考时间: {thinking_time:.2f}秒")
        
        return response.content
    except Exception as e:
        logger.error(f"调用{provider} API失败: {str(e)}")
        raise

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试LLM API")
    
    parser.add_argument("--prompt", required=True, help="发送给LLM的提示词")
    parser.add_argument("--provider", default="anthropic", 
                        choices=["openai", "anthropic", "azure", "deepseek", "gemini", "local", "siliconflow"], 
                        help="LLM提供商")
    parser.add_argument("--model", help="要使用的模型（如果未指定，则使用提供商的默认模型）")
    parser.add_argument("--image", help="可选的图片路径")
    
    return parser.parse_args()

def main():
    """主函数，用于命令行测试"""
    args = parse_args()
    
    # 设置环境变量启用模拟翻译（测试用）
    # os.environ["MOCK_TRANSLATION"] = "true"
    
    # 调用LLM
    response = query_llm(args.prompt, provider=args.provider, model=args.model, image_path=args.image)
    
    # 输出响应
    print(response)
    
    # 输出令牌使用统计
    token_tracker = get_token_tracker()
    print(f"\n令牌使用统计:", file=sys.stderr)
    print(f"总令牌: {token_tracker.total_tokens}", file=sys.stderr)
    print(f"估计成本: ${token_tracker.total_cost:.6f}", file=sys.stderr)
    print(f"请求数: {len(token_tracker.requests)}", file=sys.stderr)

def query_siliconflow(prompt: str, model: str) -> APIResponse:
    """
    调用SiliconFlow API
    
    Args:
        prompt: 提示词
        model: 模型名称
        
    Returns:
        API响应
    """
    logger.info(f"调用SiliconFlow API (模型: {model})...")
    
    try:
        # 直接创建客户端，避免使用create_llm_client
        from openai import OpenAI
        import httpx
        
        api_key = SILICONFLOW_API_KEY
        if not api_key:
            raise ValueError("环境变量中未找到SILICONFLOW_API_KEY")
        
        # 使用httpx客户端，明确控制代理参数
        http_client = httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # 明确只传递OpenAI客户端支持的参数
        client = OpenAI(
            api_key=api_key,
            base_url=SILICONFLOW_API_URL,
            http_client=http_client
        )
        
        start_time = time.time()
        
        # 创建聊天完成请求
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        
        # 提取内容和令牌使用情况
        content = response.choices[0].message.content
        token_usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        # 估算成本（使用OpenAI的成本模型作为近似值）
        cost = _token_tracker.calculate_openai_cost(
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
            "gpt-3.5-turbo"  # 估算成本时使用的参考模型
        )
        
        thinking_time = time.time() - start_time
        
        # 创建并返回响应
        api_response = APIResponse(
            content=content,
            token_usage=token_usage,
            cost=cost,
            thinking_time=thinking_time,
            provider="siliconflow",
            model=model
        )
        
        # 记录令牌使用情况
        _token_tracker.track_request(api_response)
        
        return api_response
        
    except Exception as e:
        logger.error(f"SiliconFlow API调用失败: {str(e)}")
        raise

def query_openai(prompt: str, model: str, image_path: Optional[str] = None) -> APIResponse:
    """
    调用OpenAI API
    
    Args:
        prompt: 提示词
        model: 模型名称
        image_path: 可选的图像文件路径
        
    Returns:
        API响应
    """
    logger.info(f"调用OpenAI API (模型: {model})...")
    
    try:
        # 使用create_llm_client创建客户端
        client = create_llm_client("openai")
        
        if not client:
            raise ValueError("无法创建OpenAI客户端，请检查API密钥")
        
        start_time = time.time()
        
        messages = []
        
        # 添加系统消息
        messages.append({"role": "system", "content": "你是一个有用的助手。"})
        
        # 处理图像
        if image_path and (model == "gpt-4o" or model == "gpt-4-vision-preview"):
            logger.info(f"处理图像: {image_path}")
            encoded_image, mime_type = encode_image_file(image_path)
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}"
                    }
                }
            ]
            messages.append({"role": "user", "content": content})
        else:
            # 没有图像的普通请求
            messages.append({"role": "user", "content": prompt})
        
        # 创建聊天完成请求
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        
        # 提取内容和令牌使用情况
        content = response.choices[0].message.content
        token_usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        # 估算成本
        cost = _token_tracker.calculate_openai_cost(
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
            model
        )
        
        thinking_time = time.time() - start_time
        
        # 创建并返回响应
        api_response = APIResponse(
            content=content,
            token_usage=token_usage,
            cost=cost,
            thinking_time=thinking_time,
            provider="openai",
            model=model
        )
        
        # 记录令牌使用情况
        _token_tracker.track_request(api_response)
        
        return api_response
        
    except Exception as e:
        logger.error(f"OpenAI API调用失败: {str(e)}")
        raise

def query_anthropic(prompt: str, model: str, image_path: Optional[str] = None) -> APIResponse:
    """
    调用Anthropic API
    
    Args:
        prompt: 提示词
        model: 模型名称
        image_path: 可选的图像文件路径
        
    Returns:
        API响应
    """
    logger.info(f"调用Anthropic API (模型: {model})...")
    client = create_llm_client("anthropic")
    
    if not client:
        raise ValueError("无法创建Anthropic客户端，请检查API密钥")
    
    start_time = time.time()
    
    try:
        # 创建消息
        if image_path and "claude-3" in model:
            logger.info(f"处理图像: {image_path}")
            encoded_image, mime_type = encode_image_file(image_path)
            
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.3,
                system="你是一个有用的助手。",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded_image}}
                        ]
                    }
                ]
            )
        else:
            # 没有图像的普通请求
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.3,
                system="你是一个有用的助手。",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        
        # 提取内容和令牌使用情况
        content = message.content[0].text
        token_usage = TokenUsage(
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            total_tokens=message.usage.input_tokens + message.usage.output_tokens
        )
        
        # 估算成本
        cost = _token_tracker.calculate_claude_cost(
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
            model
        )
        
        thinking_time = time.time() - start_time
        
        # 创建并返回响应
        api_response = APIResponse(
            content=content,
            token_usage=token_usage,
            cost=cost,
            thinking_time=thinking_time,
            provider="anthropic",
            model=model
        )
        
        # 记录令牌使用情况
        _token_tracker.track_request(api_response)
        
        return api_response
        
    except Exception as e:
        logger.error(f"Anthropic API调用失败: {str(e)}")
        raise

def query_azure_openai(prompt: str, model: str, image_path: Optional[str] = None) -> APIResponse:
    """
    调用Azure OpenAI API
    
    Args:
        prompt: 提示词
        model: 模型名称
        image_path: 可选的图像文件路径
        
    Returns:
        API响应
    """
    logger.info(f"调用Azure OpenAI API (模型部署: {model})...")
    
    try:
        # 直接创建客户端，避免使用create_llm_client
        from openai import AzureOpenAI
        
        api_key = AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("环境变量中未找到AZURE_OPENAI_API_KEY")
            
        # 明确只传递必要的参数
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=AZURE_ENDPOINT
        )
        
        start_time = time.time()
        
        messages = []
        
        # 添加系统消息
        messages.append({"role": "system", "content": "你是一个有用的助手。"})
        
        # 处理图像
        if image_path and ("gpt-4" in model or "vision" in model):
            logger.info(f"处理图像: {image_path}")
            encoded_image, mime_type = encode_image_file(image_path)
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}"
                    }
                }
            ]
            messages.append({"role": "user", "content": content})
        else:
            # 没有图像的普通请求
            messages.append({"role": "user", "content": prompt})
        
        # 创建聊天完成请求
        response = client.chat.completions.create(
            model=model,  # 在Azure中，这是你的部署名称
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        
        # 提取内容和令牌使用情况
        content = response.choices[0].message.content
        token_usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        # 估算成本（假设与OpenAI相同）
        cost = _token_tracker.calculate_openai_cost(
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
            "gpt-4o"  # 使用gpt-4o的价格作为估算
        )
        
        thinking_time = time.time() - start_time
        
        # 创建并返回响应
        api_response = APIResponse(
            content=content,
            token_usage=token_usage,
            cost=cost,
            thinking_time=thinking_time,
            provider="azure",
            model=model
        )
        
        # 记录令牌使用情况
        _token_tracker.track_request(api_response)
        
        return api_response
        
    except Exception as e:
        logger.error(f"Azure OpenAI API调用失败: {str(e)}")
        raise

def query_deepseek(prompt: str, model: str) -> APIResponse:
    """
    调用DeepSeek API
    
    Args:
        prompt: 提示词
        model: 模型名称
        
    Returns:
        API响应
    """
    logger.info(f"调用DeepSeek API (模型: {model})...")
    
    try:
        # 直接创建客户端，避免使用create_llm_client
        from openai import OpenAI
        import httpx
        
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("环境变量中未找到DEEPSEEK_API_KEY")
            
        # 使用httpx客户端，明确控制代理参数
        http_client = httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # 明确只传递必要的参数
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            http_client=http_client
        )
        
        start_time = time.time()
        
        # 创建聊天完成请求
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        
        # 提取内容和令牌使用情况
        content = response.choices[0].message.content
        token_usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        # 估算成本（使用OpenAI的成本模型作为近似值）
        cost = _token_tracker.calculate_openai_cost(
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
            "gpt-3.5-turbo"  # 估算成本时使用的参考模型
        )
        
        thinking_time = time.time() - start_time
        
        # 创建并返回响应
        api_response = APIResponse(
            content=content,
            token_usage=token_usage,
            cost=cost,
            thinking_time=thinking_time,
            provider="deepseek",
            model=model
        )
        
        # 记录令牌使用情况
        _token_tracker.track_request(api_response)
        
        return api_response
        
    except Exception as e:
        logger.error(f"DeepSeek API调用失败: {str(e)}")
        raise

def query_gemini(prompt: str, model: str, image_path: Optional[str] = None) -> APIResponse:
    """
    调用Google Gemini API
    
    Args:
        prompt: 提示词
        model: 模型名称
        image_path: 可选的图像文件路径
        
    Returns:
        API响应
    """
    logger.info(f"调用Google Gemini API (模型: {model})...")
    genai = create_llm_client("gemini")
    
    if not genai:
        raise ValueError("无法创建Google Gemini客户端，请检查API密钥")
    
    start_time = time.time()
    
    try:
        # 设置模型
        model_obj = genai.GenerativeModel(model)
        
        # 处理请求
        if image_path and "vision" in model:
            logger.info(f"处理图像: {image_path}")
            image = genai.upload_image(image_path)
            response = model_obj.generate_content([prompt, image])
        else:
            response = model_obj.generate_content(prompt)
        
        # 提取内容
        content = response.text
        
        # 由于Gemini API不直接提供令牌数，我们需要估计
        # 简单估算：每个单词约1.3个令牌
        prompt_words = len(prompt.split())
        response_words = len(content.split())
        prompt_tokens = int(prompt_words * 1.3)
        completion_tokens = int(response_words * 1.3)
        
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        # 估算成本（使用OpenAI的简化成本模型）
        cost = 0.00002 * (prompt_tokens + completion_tokens)
        
        thinking_time = time.time() - start_time
        
        # 创建并返回响应
        api_response = APIResponse(
            content=content,
            token_usage=token_usage,
            cost=cost,
            thinking_time=thinking_time,
            provider="gemini",
            model=model
        )
        
        # 记录令牌使用情况
        _token_tracker.track_request(api_response)
        
        return api_response
        
    except Exception as e:
        logger.error(f"Google Gemini API调用失败: {str(e)}")
        raise

def query_local_llm(prompt: str, model: str, image_path: Optional[str] = None) -> APIResponse:
    """
    调用本地LLM
    
    Args:
        prompt: 提示词
        model: 模型名称
        image_path: 可选的图像文件路径（本地LLM可能不支持）
        
    Returns:
        API响应
    """
    logger.info(f"调用本地LLM (模型: {model})...")
    
    try:
        # 使用create_llm_client创建客户端
        client = create_llm_client("local")
        
        if not client:
            raise ValueError("无法创建本地LLM客户端")
        
        start_time = time.time()
        
        # 创建聊天完成请求
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        
        # 提取内容
        content = response.choices[0].message.content
        
        # 为本地LLM创建模拟令牌使用情况
        prompt_tokens = len(prompt.split()) * 2  # 粗略估计
        completion_tokens = len(content.split()) * 2  # 粗略估计
        
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        # 本地LLM没有成本
        cost = 0.0
        
        thinking_time = time.time() - start_time
        
        # 创建并返回响应
        api_response = APIResponse(
            content=content,
            token_usage=token_usage,
            cost=cost,
            thinking_time=thinking_time,
            provider="local",
            model=model
        )
        
        # 记录令牌使用情况
        _token_tracker.track_request(api_response)
        
        return api_response
        
    except Exception as e:
        logger.error(f"本地LLM调用失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 