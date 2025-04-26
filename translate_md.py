#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Markdown/MDX文件翻译工具 - 支持多语言翻译

用法:
    python3 translate_md.py --source <源文件夹> --target <目标文件夹> [--provider <LLM提供商>] [--model <模型名称>] 
                           [--source-lang <源语言>] [--target-lang <目标语言>]

选项:
    --source       源文件夹路径，包含需要翻译的Markdown/MDX文件
    --file         单个源文件路径，指定要翻译的Markdown/MDX文件
    --target       目标文件夹路径，用于保存翻译后的Markdown/MDX文件
    --provider     LLM提供商，默认为"anthropic"，可选值: "openai", "anthropic", "azure", "deepseek", "gemini", "local", "siliconflow"
    --model        要使用的模型名称（如果未指定，则使用提供商的默认模型）
    --source-lang  源语言，默认为"auto"(自动检测)，可指定特定语言如"zh"(中文),"en"(英文),"ja"(日语)等
    --target-lang  目标语言，默认为"en"(英文)，可指定任何支持的语言如"zh"(中文),"fr"(法语),"es"(西班牙语)等
    --direction    兼容旧版本的翻译方向，默认为""(使用source-lang和target-lang)，可选值: "zh2en", "en2zh"
    --batch        批处理大小，同时处理的文件数，默认为5
    --delay        每个文件处理后的延迟时间(秒)，避免API速率限制，默认为3秒
    --help         显示帮助信息

示例:
    python3 translate_md.py --source ./docs --target ./en/docs --provider anthropic --source-lang zh --target-lang en
    python3 translate_md.py --file ./blog/post.md --target ./fr/blog --provider openai --model gpt-4o --source-lang en --target-lang fr
    python3 translate_md.py --source ./docs --target ./de/docs --provider siliconflow --model mixtral-8x7b --source-lang zh --target-lang de
    python3 translate_md.py --source ./ja/docs --target ./ko/docs --provider anthropic --source-lang ja --target-lang ko
"""

import os
import sys
import argparse
import time
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import re
from typing import Dict, Optional, List, Tuple

try:
    # 尝试导入LLM API工具（从当前目录）
    from llm_api import query_llm, get_token_tracker
except ImportError:
    try:
        # 尝试从tools目录导入（兼容旧版本）
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tools.llm_api import query_llm, get_token_tracker
    except ImportError:
        # 如果导入失败，我们将定义一个简单的函数来模拟LLM API调用
        def query_llm(prompt, provider="anthropic", model=None, image_path=None):
            print(f"[DEBUG] 发送到 {provider} 的提示词: {prompt[:100]}...")
            return f"Error: LLM API 工具未找到，无法翻译。请确保 llm_api.py 存在于当前目录或 tools/llm_api.py 存在。"
        
        def get_token_tracker():
            return None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("translate_md.log")
    ]
)
logger = logging.getLogger("translate_md")

# 支持的语言代码和名称映射
LANGUAGES = {
    "auto": "自动检测",
    "zh": "中文",
    "en": "英文",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "it": "意大利语",
    "ru": "俄语",
    "pt": "葡萄牙语",
    "nl": "荷兰语",
    "ar": "阿拉伯语",
    "hi": "印地语",
    "vi": "越南语",
    "th": "泰语",
    "id": "印尼语",
    "tr": "土耳其语",
    "pl": "波兰语",
    "sv": "瑞典语",
    "he": "希伯来语",
    "el": "希腊语",
    "da": "丹麦语",
    "fi": "芬兰语",
    "cs": "捷克语",
    "hu": "匈牙利语",
    "uk": "乌克兰语",
    "no": "挪威语",
    "ro": "罗马尼亚语"
}

# 语言检测正则表达式
LANG_PATTERN = {
    "zh": re.compile(r'[\u4e00-\u9fff]'),                    # 中文
    "ja": re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),       # 日语平假名和片假名
    "ko": re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF]'),       # 韩语
    "ru": re.compile(r'[\u0400-\u04FF]'),                    # 西里尔字母(俄语)
    "ar": re.compile(r'[\u0600-\u06FF]'),                    # 阿拉伯语
    "hi": re.compile(r'[\u0900-\u097F]'),                    # 印地语
    "th": re.compile(r'[\u0E00-\u0E7F]'),                    # 泰语
    "he": re.compile(r'[\u0590-\u05FF]'),                    # 希伯来语
    "el": re.compile(r'[\u0370-\u03FF]'),                    # 希腊语
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Markdown/MDX文件翻译工具 - 支持多语言翻译")
    
    file_source_group = parser.add_mutually_exclusive_group(required=True)
    file_source_group.add_argument("--source", help="源文件夹路径，包含需要翻译的Markdown/MDX文件")
    file_source_group.add_argument("--file", help="单个源文件路径，指定要翻译的Markdown/MDX文件")
    
    parser.add_argument("--target", required=True, help="目标文件夹路径，用于保存翻译后的Markdown/MDX文件")
    parser.add_argument("--provider", default="anthropic", 
                       choices=["openai", "anthropic", "azure", "deepseek", "gemini", "local", "siliconflow"], 
                       help="LLM提供商，默认为anthropic")
    parser.add_argument("--model", help="要使用的模型（如果未指定，则使用提供商的默认模型）")
    parser.add_argument("--source-lang", default="auto", help="源语言，默认为auto(自动检测)，可指定特定语言如zh(中文),en(英文),ja(日语)等")
    parser.add_argument("--target-lang", default="en", help="目标语言，默认为en(英文)，可指定任何支持的语言")
    parser.add_argument("--direction", default="", choices=["", "zh2en", "en2zh"], 
                       help="兼容旧版本的翻译方向，默认为空(使用source-lang和target-lang)，可选zh2en或en2zh")
    parser.add_argument("--batch", type=int, default=5, help="批处理大小，同时处理的文件数，默认为5")
    parser.add_argument("--delay", type=int, default=3, help="每个文件处理后的延迟时间(秒)，默认为3秒")
    
    args = parser.parse_args()
    
    # 兼容旧版本的direction参数
    if args.direction:
        if args.direction == "zh2en":
            args.source_lang = "zh"
            args.target_lang = "en"
        elif args.direction == "en2zh":
            args.source_lang = "en"
            args.target_lang = "zh"
    
    return args

def detect_language(text: str) -> str:
    """
    自动检测文本的语言
    
    Args:
        text: 要检测的文本内容
        
    Returns:
        检测到的语言代码，如果无法确定，则返回"en"
    """
    # 首先检查特殊字符集
    for lang, pattern in LANG_PATTERN.items():
        if pattern.search(text):
            return lang
    
    # 如果没有特殊字符，尝试基于常见单词判断
    # 这里使用简单启发式，实际项目中可以使用更复杂的语言检测库
    
    # 检查是否主要为英文文本
    english_words = 0
    for word in text.split():
        if re.match(r'^[a-zA-Z]+$', word):
            english_words += 1
    
    if english_words > 10:  # 至少有10个英文单词
        return "en"
    
    # 默认返回英文
    return "en"

def is_markdown_file(filename: str) -> bool:
    """检查文件是否为Markdown或MDX文件"""
    return filename.endswith('.md') or filename.endswith('.mdx')

def get_markdown_extension(filename: str) -> str:
    """获取Markdown文件的扩展名(.md或.mdx)"""
    if filename.endswith('.mdx'):
        return '.mdx'
    elif filename.endswith('.md'):
        return '.md'
    return ''

def get_translation_system_prompt(provider: str, is_mdx: bool, source_lang: str, target_lang: str) -> str:
    """
    获取适合特定提供商的翻译系统提示
    
    Args:
        provider: LLM提供商
        is_mdx: 是否为MDX文件
        source_lang: 源语言代码
        target_lang: 目标语言代码
    
    Returns:
        适合该提供商的系统提示
    """
    source_lang_name = LANGUAGES.get(source_lang, source_lang)
    target_lang_name = LANGUAGES.get(target_lang, target_lang)
    
    # 通用翻译指令模板
    general_template = f"""
你是一个专业的{source_lang_name}到{target_lang_name}翻译专家。请将以下{source_lang_name}Markdown文档翻译成{target_lang_name}。遵循以下规则：

1. 保持Markdown格式和结构不变，只翻译文本内容
2. 保留代码块内容、链接URL、图片路径不变
3. 确保翻译准确、自然、符合{target_lang_name}表达习惯
4. 保持专业术语的准确性
5. 不要添加原文中不存在的内容
6. 不要省略原文中的任何内容
7. 技术术语尽可能使用业内通用的{target_lang_name}翻译
"""

    # 添加MDX特定说明
    if is_mdx:
        general_template += """
8. 这是一个MDX文件，包含JSX组件。请不要翻译JSX组件内的属性名称和组件名称
9. 保持所有JSX组件标签不变，例如 <Component>、</Component>、<Component />
10. 只翻译JSX组件中的文本内容和可翻译的属性值
11. 对于代码块中的内容，不进行翻译
12. 对于front matter (开头的---之间的YAML内容)，只翻译title、description等可翻译的值，不翻译键名或其他元数据
"""
    
    # Anthropic通常需要更简洁的指令
    if provider == "anthropic":
        anthropic_template = f"""
You are a professional translator specialized in translating from {source_lang} to {target_lang}. Translate the following Markdown document.
Maintain all Markdown formatting and structure. Only translate the text content.
Keep code blocks, link URLs, image paths, and other technical content unchanged.
Ensure accurate, natural translation that preserves professional terminology.
Use common {target_lang} translations for technical terms whenever possible.
"""
        
        if is_mdx:
            anthropic_template += """
This is an MDX file containing JSX components. Do not translate JSX component property names and component names.
Keep all JSX component tags unchanged, such as <Component>, </Component>, <Component />.
Only translate the text content within JSX components and translatable property values.
Do not translate content within code blocks.
For front matter (YAML content between --- at the beginning), only translate values for title, description, etc., not the keys or other metadata.
"""
        return anthropic_template
    
    # Gemini喜欢更详细的指令
    elif provider == "gemini":
        gemini_template = general_template + """
8. 遵循目标语言的标点符号规范
9. 保持标题、列表、表格等Markdown元素的格式
10. 对于代码注释，仅翻译注释部分，保持代码逻辑不变
"""
        if is_mdx:
            gemini_template += """
11. 对于MDX特有的JSX语法，请保持组件标签和属性名不变
12. 对于组件内的内容，只翻译可读的文本部分，保持代码和属性名原样
13. 特别注意front matter区域，只翻译内容值，而不是键名
"""
        return gemini_template
    
    # SiliconFlow模型可能与OpenAI兼容，但可能需要更具体的指令
    elif provider == "siliconflow":
        siliconflow_template = general_template + """
8. 如果模型的能力有限，请优先保证翻译的准确性和可读性，而不是追求完美的表达方式
9. 对于技术文档，保持专业术语的准确翻译
"""
        if is_mdx:
            siliconflow_template += """
10. 对于MDX中的JSX组件，例如<Component prop="value">内容</Component>，请只翻译"内容"部分和prop属性的value值(如果它是自然语言)
11. 不要改变任何JSX语法结构，确保组件可以正常工作
"""
        return siliconflow_template
    
    # 其他提供商使用基本指令
    return general_template + ("""
8. 对于MDX中的JSX组件，例如<Component prop="value">内容</Component>，请只翻译"内容"部分和prop属性的value值(如果它是自然语言)
9. 不要改变任何JSX语法结构，确保组件可以正常工作
10. 对于front matter区域(文件开头---之间的部分)，只翻译值，不翻译键名
""" if is_mdx else "")

def translate_content(content: str, provider: str, model: Optional[str] = None, 
                     is_mdx: bool = False, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    使用LLM将Markdown/MDX内容翻译成目标语言
    
    Args:
        content: 要翻译的内容
        provider: LLM提供商
        model: 要使用的模型（如果为None，将使用默认模型）
        is_mdx: 是否为MDX文件
        source_lang: 源语言代码
        target_lang: 目标语言代码
        
    Returns:
        翻译后的内容
    """
    # 自动检测语言
    if source_lang == "auto":
        detected_lang = detect_language(content)
        logger.info(f"自动检测到文档语言: {LANGUAGES.get(detected_lang, detected_lang)}")
        source_lang = detected_lang
    
    # 如果源语言和目标语言相同，则不需要翻译
    if source_lang == target_lang:
        logger.info("源语言和目标语言相同，跳过翻译")
        return content
    
    # 获取适合提供商的系统提示
    system_prompt = get_translation_system_prompt(provider, is_mdx, source_lang, target_lang)
    
    file_type = "MDX" if is_mdx else "Markdown"
    source_lang_name = LANGUAGES.get(source_lang, source_lang)
    target_lang_name = LANGUAGES.get(target_lang, target_lang)
    
    # 构建提示词
    prompt = f"""{system_prompt}

{source_lang_name}{file_type}文档：

{content}

{target_lang_name}翻译：
"""
    
    try:
        # 调用LLM进行翻译
        logger.info(f"正在使用 {provider} {'(' + model + ')' if model else ''} 将{source_lang_name}{file_type}翻译为{target_lang_name}...")
        translation = query_llm(prompt, provider=provider, model=model)
        time.sleep(1)  # 短暂延迟，避免过于频繁的API调用
        
        # 查看令牌使用情况（如果可用）
        token_tracker = get_token_tracker()
        if token_tracker and token_tracker.requests:
            last_request = token_tracker.requests[-1]
            logger.info(f"翻译使用了 {last_request.token_usage.total_tokens} 个令牌，" 
                       f"成本: ${last_request.cost:.6f}")
        
        return translation
    except Exception as e:
        logger.error(f"翻译失败: {str(e)}")
        return f"[TRANSLATION ERROR: {str(e)}]\n\n{content}"

def process_file(source_file: str, target_file: str, provider: str, model: Optional[str] = None, 
                delay: int = 3, source_lang: str = "auto", target_lang: str = "en") -> bool:
    """
    处理单个文件的翻译
    
    Args:
        source_file: 源文件路径
        target_file: 目标文件路径
        provider: LLM提供商
        model: 要使用的模型（如果为None，将使用默认模型）
        delay: 翻译后的延迟时间(秒)
        source_lang: 源语言代码
        target_lang: 目标语言代码
        
    Returns:
        是否成功翻译
    """
    try:
        # 确保目标文件夹存在
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        # 对于非Markdown/MDX文件，直接复制
        if not is_markdown_file(source_file):
            shutil.copy2(source_file, target_file)
            logger.info(f"复制文件: {source_file} -> {target_file}")
            return True
        
        # 是否为MDX文件
        is_mdx = source_file.endswith('.mdx')
        file_type = "MDX" if is_mdx else "Markdown"
        
        # 读取源文件内容
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 翻译内容
        source_lang_name = LANGUAGES.get(source_lang, source_lang)
        target_lang_name = LANGUAGES.get(target_lang, target_lang)
        
        logger.info(f"翻译{file_type}文件: {source_file} ({source_lang_name} -> {target_lang_name})")
        translated_content = translate_content(content, provider, model, is_mdx, source_lang, target_lang)
        
        # 检查翻译是否成功，判断错误标记
        if translated_content.startswith("[TRANSLATION ERROR:"):
            logger.error(f"翻译失败，不保存文件: {target_file}")
            return False
            
        # 写入目标文件
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
            
        logger.info(f"翻译完成: {target_file}")
        
        # 添加延迟，避免API速率限制
        time.sleep(delay)
        return True
    
    except Exception as e:
        logger.error(f"处理文件 {source_file} 时出错: {str(e)}")
        return False

def collect_files(directory: str) -> List[str]:
    """
    收集目录中的所有文件
    
    Args:
        directory: 要收集文件的目录
        
    Returns:
        文件路径列表
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def translate_directory(source_dir: str, target_dir: str, provider: str, model: Optional[str] = None, 
                       max_workers: int = 5, delay: int = 3, source_lang: str = "auto", target_lang: str = "en") -> bool:
    """
    翻译整个目录下的Markdown/MDX文件
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        provider: LLM提供商
        model: 要使用的模型（如果为None，将使用默认模型）
        max_workers: 并行工作线程数
        delay: 每个文件处理后的延迟时间(秒)
        source_lang: 源语言代码
        target_lang: 目标语言代码
        
    Returns:
        是否成功完成翻译
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        logger.error(f"源文件夹不存在: {source_dir}")
        return False
    
    # 确保目标目录存在
    os.makedirs(target_path, exist_ok=True)
    
    # 收集所有要处理的文件
    files_to_process = []
    for root, _, files in os.walk(source_path):
        for file in files:
            source_file = os.path.join(root, file)
            # 计算相对路径，保持目录结构
            rel_path = os.path.relpath(source_file, source_path)
            target_file = os.path.join(target_path, rel_path)
            files_to_process.append((source_file, target_file))
    
    total_files = len(files_to_process)
    markdown_files = sum(1 for src, _ in files_to_process if src.endswith('.md'))
    mdx_files = sum(1 for src, _ in files_to_process if src.endswith('.mdx'))
    
    source_lang_name = LANGUAGES.get(source_lang, source_lang)
    target_lang_name = LANGUAGES.get(target_lang, target_lang)
    
    if source_lang == "auto":
        source_lang_name = "自动检测"
    
    logger.info(f"发现 {total_files} 个文件需要处理，其中 {markdown_files} 个Markdown文件和 {mdx_files} 个MDX文件需要翻译 ({source_lang_name} -> {target_lang_name})")
    
    if markdown_files + mdx_files == 0:
        logger.warning("没有找到Markdown或MDX文件，将只进行文件复制")
    
    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, src, tgt, provider, model, delay, source_lang, target_lang) 
            for src, tgt in files_to_process
        ]
        
        # 监控处理进度
        completed = 0
        for future in futures:
            try:
                future.result()  # 等待任务完成
                completed += 1
                if completed % 5 == 0 or completed == total_files:
                    logger.info(f"进度: {completed}/{total_files} ({completed*100/total_files:.1f}%)")
            except Exception as e:
                logger.error(f"任务执行失败: {str(e)}")
    
    # 生成翻译报告
    report = {
        "总文件数": total_files,
        "Markdown文件数": markdown_files,
        "MDX文件数": mdx_files,
        "完成时间": time.strftime('%Y-%m-%d %H:%M:%S'),
        "提供商": provider,
        "模型": model or "默认",
        "源语言": source_lang_name,
        "目标语言": target_lang_name,
        "翻译方向": f"{source_lang_name} -> {target_lang_name}",
    }
    
    # 尝试添加令牌使用统计
    token_tracker = get_token_tracker()
    if token_tracker:
        report["令牌使用量"] = token_tracker.total_tokens
        report["估计成本"] = f"${token_tracker.total_cost:.6f}"
        report["请求次数"] = len(token_tracker.requests)
    
    # 将报告写入目标目录
    report_path = os.path.join(target_dir, "translation-report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"翻译完成，结果保存在 {target_dir}")
    logger.info(f"翻译报告保存在 {report_path}")
    return True

def main():
    """主函数"""
    args = parse_args()
    
    source_lang = args.source_lang
    target_lang = args.target_lang
    
    source_lang_name = LANGUAGES.get(source_lang, source_lang)
    target_lang_name = LANGUAGES.get(target_lang, target_lang)
    
    if source_lang == "auto":
        source_lang_name = "自动检测"
    
    if args.file:
        # 处理单个文件
        if not os.path.isfile(args.file):
            logger.error(f"源文件不存在: {args.file}")
            sys.exit(1)
            
        if not is_markdown_file(args.file):
            logger.error(f"不是Markdown或MDX文件: {args.file}")
            sys.exit(1)
            
        # 计算目标文件路径
        filename = os.path.basename(args.file)
        target_file = os.path.join(args.target, filename)
        
        logger.info(f"开始翻译单个文件，源文件: {args.file}，目标文件夹: {args.target}，翻译方向: {source_lang_name} -> {target_lang_name}")
        logger.info(f"LLM提供商: {args.provider}" + (f", 模型: {args.model}" if args.model else ""))
        
        # 确保目标目录存在
        os.makedirs(args.target, exist_ok=True)
        
        # 处理文件
        process_file(args.file, target_file, args.provider, args.model, args.delay, source_lang, target_lang)
        
        logger.info(f"翻译完成，结果保存到: {target_file}")
    else:
        # 处理整个目录
        logger.info(f"开始翻译目录，源文件夹: {args.source}，目标文件夹: {args.target}，翻译方向: {source_lang_name} -> {target_lang_name}")
        logger.info(f"LLM提供商: {args.provider}" + (f", 模型: {args.model}" if args.model else ""))
        logger.info(f"并行处理: {args.batch} 个文件, 延迟: {args.delay} 秒")
        
        success = translate_directory(
            args.source, 
            args.target, 
            args.provider, 
            args.model, 
            max_workers=args.batch, 
            delay=args.delay,
            source_lang=source_lang
        )
        
        if not success:
            logger.error("翻译任务失败")
            sys.exit(1)
    
    # 如果有令牌使用跟踪，打印统计信息
    token_tracker = get_token_tracker()
    if token_tracker:
        logger.info(f"令牌使用统计:")
        logger.info(f"总令牌: {token_tracker.total_tokens}")
        logger.info(f"估计成本: ${token_tracker.total_cost:.6f}")
        logger.info(f"请求数: {len(token_tracker.requests)}")
    
    logger.info("翻译任务成功完成")
    sys.exit(0)

if __name__ == "__main__":
    main() 