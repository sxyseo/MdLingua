#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
翻译工具测试脚本 - 用于测试Markdown翻译功能

此脚本创建一个临时目录，包含几个测试用的中文Markdown文件，
然后调用translate_md.py进行翻译，最后检查结果是否符合预期。
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

# 测试用的中文Markdown内容
TEST_CONTENT = """
# 测试文档

这是一个用于测试翻译功能的文档。

## 基本功能

* 列表项1
* 列表项2
* 列表项3

## 代码示例

```python
def hello_world():
    print("你好，世界！")
```

## 链接和图片

[示例链接](https://example.com)

![示例图片](./images/example.png)

## 表格

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |
| 测试1 | 测试2 | 测试3 |
"""

def create_test_files(base_dir):
    """创建测试用的Markdown文件"""
    # 创建目录结构
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(os.path.join(docs_dir, "guide"), exist_ok=True)
    os.makedirs(os.path.join(docs_dir, "api"), exist_ok=True)
    
    # 创建测试文件
    with open(os.path.join(docs_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("# 文档首页\n\n这是文档的首页，包含项目概述和导航。")
    
    with open(os.path.join(docs_dir, "guide", "getting-started.md"), "w", encoding="utf-8") as f:
        f.write("# 入门指南\n\n本指南将帮助您快速上手我们的产品。")
    
    with open(os.path.join(docs_dir, "api", "reference.md"), "w", encoding="utf-8") as f:
        f.write(TEST_CONTENT)
    
    # 创建一个非Markdown文件用于测试复制功能
    with open(os.path.join(docs_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write('{"name": "测试配置", "version": "1.0.0"}')
    
    return docs_dir

def run_translation(source_dir, target_dir, provider="anthropic"):
    """运行翻译工具"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translate_md.py")
    
    # 确保脚本存在
    if not os.path.exists(script_path):
        print(f"错误: 找不到翻译脚本: {script_path}")
        return False
    
    # 运行翻译脚本
    cmd = [sys.executable, script_path, "--source", source_dir, "--target", target_dir, "--provider", provider]
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"翻译失败: {str(e)}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def main():
    """主函数"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"创建临时测试目录: {temp_dir}")
        
        # 创建测试文件
        source_dir = create_test_files(temp_dir)
        target_dir = os.path.join(temp_dir, "en")
        
        # 运行翻译
        success = run_translation(source_dir, target_dir)
        
        if success:
            print("测试成功完成！")
            print(f"翻译结果位于: {target_dir}")
        else:
            print("测试失败")
    
    finally:
        # 保留临时目录以便查看结果
        print(f"测试完成。临时目录 {temp_dir} 未删除，您可以检查结果后手动删除。")

if __name__ == "__main__":
    main() 