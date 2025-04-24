#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Markdown/MDX文件翻译工具辅助模块 - 为GUI界面提供支持

这个模块提供了一个修改版的translate_directory函数，支持进度回调，
以便在GUI界面中显示翻译进度。
"""

import os
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Optional, Callable

# 尝试导入翻译模块
try:
    from translate_md import (
        process_file, 
        LANGUAGES, 
        get_token_tracker, 
        is_markdown_file,
        logger
    )
except ImportError:
    # 如果导入失败，定义一个简单的模拟函数
    logger = logging.getLogger("translate_utils")
    def process_file(*args, **kwargs):
        logger.error("无法导入translate_md模块。请确保translate_md.py在同一目录下。")
        return None
    
    LANGUAGES = {"auto": "自动检测", "zh": "中文", "en": "英文"}
    def get_token_tracker(): return None
    def is_markdown_file(filename): return filename.endswith('.md') or filename.endswith('.mdx')

def translate_directory_with_progress(
    source_dir: str, 
    target_dir: str, 
    provider: str, 
    model: Optional[str] = None, 
    max_workers: int = 5, 
    delay: int = 3, 
    source_lang: str = "auto", 
    target_lang: str = "en",
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> bool:
    """
    翻译整个目录下的Markdown/MDX文件，支持进度回调
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        provider: LLM提供商
        model: 要使用的模型（如果为None，将使用默认模型）
        max_workers: 并行工作线程数
        delay: 每个文件处理后的延迟时间(秒)
        source_lang: 源语言代码
        target_lang: 目标语言代码
        progress_callback: 进度回调函数，接收(当前文件数, 总文件数)作为参数
        
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
    
    # 通知初始进度
    if progress_callback:
        progress_callback(0, total_files)
    
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
                
                # 更新进度条和日志
                if completed % 1 == 0 or completed == total_files:  # 每处理一个文件就更新进度
                    progress_message = f"进度: {completed}/{total_files} ({completed*100/total_files:.1f}%)"
                    logger.info(progress_message)
                    
                    # 调用进度回调函数
                    if progress_callback:
                        progress_callback(completed, total_files)
                        
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
    
    # 最终通知100%进度
    if progress_callback:
        progress_callback(total_files, total_files)
        
    return True 