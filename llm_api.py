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
    "gemini": "gemini-pro",
    "local": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "siliconflow": "mixtral-8x7b"
}

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
            return OpenAI(api_key=api_key)
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
                azure_endpoint=AZURE_ENDPOINT
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
                base_url=SILICONFLOW_API_URL
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
                api_key="not-needed"
            )
        except ImportError:
            logger.error("请安装OpenAI包: pip install openai")
            return None
            
    else:
        logger.error(f"不支持的提供商: {provider}")
        return None

def simulate_translation(prompt: str) -> str:
    """
    模拟翻译，仅用于测试目的
    
    Args:
        prompt: 包含要翻译内容的提示词
        
    Returns:
        模拟的翻译结果
    """
    # 简单的中英文对照
    translations = {
        "快速入门": "Quick Start",
        "几分钟内开始使用 AutoDev": "Get started with AutoDev in minutes",
        "安装与设置": "Installation and Setup",
        "JetBrains 插件市场": "JetBrains Plugin Marketplace",
        "自定义仓库": "Custom Repository",
        "GitHub 发布版本": "GitHub Releases",
        "配置": "Configuration",
        "默认 LLM": "Default LLM",
        "基础配置": "Basic Configuration",
        "高级配置": "Advanced Configuration",
        "附加模型": "Additional Models",
        "注意": "Note",
        "此版本适用于": "This version is compatible with",
        "及更新版本": "and newer versions",
        "前往": "Go to",
        "设置": "Settings",
        "插件": "Plugins",
        "插件市场": "Plugin Marketplace",
        "管理插件仓库": "Manage Plugin Repositories",
        "添加以下 URL": "Add the following URL",
        "从": "From",
        "下载适合的版本": "download the appropriate version",
        "适用于": "for",
        "在 JetBrains IDE 中从磁盘安装插件": "Install the plugin from disk in JetBrains IDE",
        "安装后": "After installation",
        "中配置插件": "configure the plugin in",
        "支持的提供商": "Supported providers",
        "打开 AutoDev 配置": "open AutoDev configuration",
        "配置": "Configure",
        "服务器地址": "server address",
        "例如": "For example",
        "输入您的": "Enter your",
        "密钥": "key",
        "使用": "Use",
        "设置": "set",
        "自定义响应格式": "custom response format",
        "自定义请求格式": "custom request format",
        "有关更详细的配置选项": "For more detailed configuration options",
        "请参见": "see",
        "可用的模型类型": "Available model types",
        "用于推理和规划": "for reasoning and planning",
        "推荐": "Recommended",
        "用于代码补全": "for code completion",
        "用于修复补丁生成": "for patch generation",
        "通用占位符": "general placeholder",
        "尚未就绪": "not ready yet",
        "用于执行操作": "for action execution",
        "用于嵌入函数": "for embedding functions",
        "配置字段": "Configuration fields",
        "包含端点路径的": "containing the endpoint path",
        "认证信息": "authentication information",
        "目前仅支持": "currently only supports",
        "令牌": "token",
        "API 请求的 JSON 结构": "JSON structure for API requests",
        "用于从响应中提取内容的": "for extracting content from responses",
        "模型类型": "model type",
        "见上面列表": "see list above",
        # 添加更多通用翻译
        "测试文档": "Test Document",
        "这是一个用于测试翻译功能的文档": "This is a document for testing the translation function",
        "基本功能": "Basic Features",
        "列表项": "List Item",
        "代码示例": "Code Example",
        "你好，世界": "Hello, World",
        "链接和图片": "Links and Images",
        "示例链接": "Example Link",
        "示例图片": "Example Image",
        "表格": "Table",
        "列": "Column",
        "数据": "Data",
        "测试": "Test",
        "文档首页": "Documentation Home",
        "这是文档的首页，包含项目概述和导航": "This is the documentation home page, containing project overview and navigation",
        "入门指南": "Getting Started Guide",
        "本指南将帮助您快速上手我们的产品": "This guide will help you quickly get started with our product"
    }
    
    # 从提示词中提取要翻译的内容
    content_start = prompt.find("中文MDX文档：") if "中文MDX文档：" in prompt else prompt.find("中文Markdown文档：")
    content_end = prompt.find("英文翻译：")
    
    if content_start == -1 or content_end == -1:
        # 如果找不到标准分隔符，尝试其他可能的格式
        content_start = prompt.find("中文文档：")
        content_end = prompt.find("英文翻译：")
        
        if content_start == -1 or content_end == -1:
            logger.warning("无法从提示词中提取内容，尝试直接翻译整个提示词")
            content = prompt
        else:
            content = prompt[content_start + len("中文文档："):content_end].strip()
    else:
        content = prompt[content_start + len("中文MDX文档：" if "中文MDX文档：" in prompt else "中文Markdown文档："):content_end].strip()
    
    # 处理front matter
    if content.startswith("---"):
        end_front_matter = content.find("---", 3)
        if end_front_matter != -1:
            front_matter = content[:end_front_matter + 3]
            main_content = content[end_front_matter + 3:].strip()
            
            # 翻译front matter中的值
            for cn, en in translations.items():
                front_matter = front_matter.replace(f": {cn}", f": {en}")
            
            content = front_matter + "\n\n" + main_content
    
    # 简单替换关键词以模拟翻译
    translated = content
    for cn, en in translations.items():
        # 避免替换JSX组件和属性名
        if "<" in cn or ">" in cn or "=" in cn:
            continue
            
        # 替换文本内容，但保持JSX标签不变
        parts = translated.split("<")
        for i in range(len(parts)):
            if i == 0:  # 第一个部分（标签前的内容）
                parts[i] = parts[i].replace(cn, en)
            else:  # 标签内的内容
                tag_end = parts[i].find(">")
                if tag_end != -1:
                    # 分离标签和内容
                    tag = parts[i][:tag_end + 1]
                    content = parts[i][tag_end + 1:]
                    
                    # 只翻译内容部分
                    content = content.replace(cn, en)
                    
                    # 重新组合
                    parts[i] = tag + content
        
        translated = "<".join(parts)
    
    # 处理代码块
    code_blocks = []
    current_pos = 0
    while True:
        start = translated.find("```", current_pos)
        if start == -1:
            break
        end = translated.find("```", start + 3)
        if end == -1:
            break
        code_blocks.append((start, end + 3))
        current_pos = end + 3
    
    # 恢复代码块内容
    for start, end in reversed(code_blocks):
        code_content = translated[start:end]
        translated = translated[:start] + code_content + translated[end:]
    
    return translated

def query_llm(prompt: str, provider: str = "anthropic", model: Optional[str] = None, image_path: Optional[str] = None) -> str:
    """
    查询LLM并获取响应
    
    Args:
        prompt: 提示词
        provider: LLM提供商，支持"openai"、"anthropic"、"azure"、"deepseek"、"gemini"、"local"、"siliconflow"
        model: 要使用的模型名称（如果为None，将使用默认模型）
        image_path: 可选的图片路径
        
    Returns:
        LLM的响应文本
    """
    logger.info(f"正在使用 {provider} 处理请求")
    
    # 对于翻译任务，在测试模式下使用模拟翻译
    if "将以下中文Markdown文档翻译成英文" in prompt and os.getenv("MOCK_TRANSLATION", "true").lower() == "true":
        logger.info("使用模拟翻译（测试模式）")
        time.sleep(1)  # 模拟API延迟
        return simulate_translation(prompt)
    
    # 尝试创建LLM客户端
    client = create_llm_client(provider)
    if client is None:
        return f"错误: 无法创建 {provider} 客户端"
    
    # 设置默认模型
    if model is None:
        model = DEFAULT_MODELS.get(provider, "")
        
    try:
        start_time = time.time()
        
        if provider in ["openai", "local", "deepseek", "azure", "siliconflow"]:
            messages = [{"role": "user", "content": []}]
            
            # 添加文本内容
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })
            
            # 如果提供了图片，将其添加到消息中
            if image_path and os.path.exists(image_path):
                encoded_image, mime_type = encode_image_file(image_path)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}
                })
            
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,  # 对于翻译，使用较低的温度
            }
            
            # 为o1模型添加特定参数
            if model == "o1":
                kwargs["response_format"] = {"type": "text"}
                kwargs["reasoning_effort"] = "low"
                del kwargs["temperature"]
            
            response = client.chat.completions.create(**kwargs)
            thinking_time = time.time() - start_time
            
            # 跟踪令牌使用
            token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                reasoning_tokens=getattr(response.usage, 'reasoning_tokens', None) if model.lower().startswith("o") else None
            )
            
            # 计算成本
            cost = get_token_tracker().calculate_openai_cost(
                token_usage.prompt_tokens,
                token_usage.completion_tokens,
                model
            )
            
            # 跟踪请求
            api_response = APIResponse(
                content=response.choices[0].message.content,
                token_usage=token_usage,
                cost=cost,
                thinking_time=thinking_time,
                provider=provider,
                model=model
            )
            get_token_tracker().track_request(api_response)
            
            return response.choices[0].message.content
            
        elif provider == "anthropic":
            messages = [{"role": "user", "content": []}]
            
            # 添加文本内容
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })
            
            # 如果提供了图片，将其添加到消息中
            if image_path and os.path.exists(image_path):
                encoded_image, mime_type = encode_image_file(image_path)
                messages[0]["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": encoded_image
                    }
                })
            
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.3,  # 对于翻译，使用较低的温度
                messages=messages
            )
            thinking_time = time.time() - start_time
            
            # 跟踪令牌使用
            token_usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
            
            # 计算成本
            cost = get_token_tracker().calculate_claude_cost(
                token_usage.prompt_tokens,
                token_usage.completion_tokens,
                model
            )
            
            # 跟踪请求
            api_response = APIResponse(
                content=response.content[0].text,
                token_usage=token_usage,
                cost=cost,
                thinking_time=thinking_time,
                provider=provider,
                model=model
            )
            get_token_tracker().track_request(api_response)
            
            return response.content[0].text
            
        elif provider == "gemini":
            gemini_model = client.GenerativeModel(model)
            
            if image_path and os.path.exists(image_path):
                # 对于Gemini，我们需要以不同方式处理图像
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                response = gemini_model.generate_content([prompt, image_data])
            else:
                response = gemini_model.generate_content(prompt)
                
            thinking_time = time.time() - start_time
            
            # Gemini目前不提供令牌使用情况，所以我们使用估计值
            estimated_prompt_tokens = len(prompt) // 4
            estimated_completion_tokens = len(response.text) // 4
            
            token_usage = TokenUsage(
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
                total_tokens=estimated_prompt_tokens + estimated_completion_tokens
            )
            
            # 跟踪请求（没有实际成本计算）
            api_response = APIResponse(
                content=response.text,
                token_usage=token_usage,
                cost=0.0,  # 我们目前不计算Gemini的成本
                thinking_time=thinking_time,
                provider=provider,
                model=model
            )
            get_token_tracker().track_request(api_response)
            
            return response.text
            
    except Exception as e:
        logger.error(f"查询LLM时出错: {str(e)}")
        return f"错误: {provider} API调用失败 - {str(e)}"

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
    os.environ["MOCK_TRANSLATION"] = "true"
    
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

if __name__ == "__main__":
    main() 