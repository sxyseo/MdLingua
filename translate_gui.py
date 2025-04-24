#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Markdown/MDX文件翻译工具GUI界面

这个程序提供了一个图形用户界面，允许用户：
1. 拖拽文件或文件夹进行翻译
2. 选择源语言和目标语言
3. 选择LLM提供商和模型
4. 选择输出文件夹
5. 查看翻译进度和结果

依赖:
- tkinter (Python内置库)
- tkinterdnd2 (拖放功能): pip install tkinterdnd2
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
from pathlib import Path

# 尝试导入拖放功能
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
except ImportError:
    messagebox.showerror("缺少依赖", "请安装tkinterdnd2库: pip install tkinterdnd2")
    sys.exit(1)

# 导入翻译功能
try:
    from translate_md import (
        LANGUAGES, 
        is_markdown_file, 
        process_file, 
        get_token_tracker
    )
    from translate_utils import translate_directory_with_progress
except ImportError:
    messagebox.showerror("缺少依赖", "无法导入翻译模块。请确保translate_md.py在同一目录下。")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("translate_gui.log")
    ]
)
logger = logging.getLogger("translate_gui")

# 支持的LLM提供商
PROVIDERS = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI GPT",
    "azure": "Azure OpenAI",
    "deepseek": "DeepSeek",
    "gemini": "Google Gemini",
    "siliconflow": "SiliconFlow",
    "local": "本地模型"
}

# 默认模型
DEFAULT_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1"],
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    "azure": ["gpt-4o-ms"],
    "deepseek": ["deepseek-chat"],
    "gemini": ["gemini-pro", "gemini-pro-vision", "gemini-1.5-flash"],
    "siliconflow": ["mixtral-8x7b", "qwen-7b", "chatglm2-6b", "deepseek-ai/DeepSeek-R1"],
    "local": ["Qwen/Qwen2.5-32B-Instruct-AWQ"]
}

class TranslateFrame(ttk.Frame):
    """翻译应用程序的主框架"""
    
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.translation_running = False
        self.source_path = None
        self.target_dir = None
        self.is_single_file = False
        
        self.setup_ui()
        self.setup_dnd()
        
    def setup_ui(self):
        """设置用户界面"""
        self.create_source_frame()
        self.create_target_frame()
        self.create_options_frame()
        self.create_progress_frame()
        self.create_buttons_frame()
        self.create_log_frame()
        
    def create_source_frame(self):
        """创建源文件/文件夹选择框架"""
        source_frame = ttk.LabelFrame(self, text="源文件/文件夹")
        source_frame.pack(fill="x", padx=10, pady=5, ipady=5)
        
        self.source_label = ttk.Label(source_frame, text="拖放文件或文件夹到这里，或点击浏览...")
        self.source_label.pack(side="left", expand=True, fill="x", padx=5)
        
        source_btn_frame = ttk.Frame(source_frame)
        source_btn_frame.pack(side="right", padx=5)
        
        self.source_file_btn = ttk.Button(source_btn_frame, text="选择文件", command=self.browse_source_file)
        self.source_file_btn.pack(side="left", padx=2)
        
        self.source_dir_btn = ttk.Button(source_btn_frame, text="选择文件夹", command=self.browse_source_dir)
        self.source_dir_btn.pack(side="left", padx=2)
        
    def create_target_frame(self):
        """创建目标文件夹选择框架"""
        target_frame = ttk.LabelFrame(self, text="目标文件夹")
        target_frame.pack(fill="x", padx=10, pady=5, ipady=5)
        
        self.target_label = ttk.Label(target_frame, text="选择翻译结果的保存位置...")
        self.target_label.pack(side="left", expand=True, fill="x", padx=5)
        
        self.target_btn = ttk.Button(target_frame, text="浏览...", command=self.browse_target_dir)
        self.target_btn.pack(side="right", padx=5)
        
    def create_options_frame(self):
        """创建翻译选项框架"""
        options_frame = ttk.LabelFrame(self, text="翻译选项")
        options_frame.pack(fill="x", padx=10, pady=5, ipady=5)
        
        # 创建网格布局
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(2, weight=1)
        options_frame.columnconfigure(3, weight=1)
        
        # 源语言选择
        ttk.Label(options_frame, text="源语言:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.source_lang_var = tk.StringVar(value="auto")
        self.source_lang_combobox = ttk.Combobox(options_frame, textvariable=self.source_lang_var)
        self.source_lang_combobox['values'] = ["auto"] + list(LANGUAGES.keys())
        self.source_lang_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # 目标语言选择
        ttk.Label(options_frame, text="目标语言:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.target_lang_var = tk.StringVar(value="en")
        self.target_lang_combobox = ttk.Combobox(options_frame, textvariable=self.target_lang_var)
        self.target_lang_combobox['values'] = list(LANGUAGES.keys())
        self.target_lang_combobox.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        
        # LLM提供商选择
        ttk.Label(options_frame, text="LLM提供商:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.provider_var = tk.StringVar(value="anthropic")
        self.provider_combobox = ttk.Combobox(options_frame, textvariable=self.provider_var)
        self.provider_combobox['values'] = list(PROVIDERS.keys())
        self.provider_combobox.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.provider_combobox.bind("<<ComboboxSelected>>", self.update_model_options)
        
        # 模型选择
        ttk.Label(options_frame, text="模型:").grid(row=1, column=2, sticky="e", padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(options_frame, textvariable=self.model_var)
        self.model_combobox.grid(row=1, column=3, sticky="w", padx=5, pady=5)
        self.update_model_options()
        
        # 批处理大小（仅目录模式）
        ttk.Label(options_frame, text="批处理大小:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.batch_var = tk.IntVar(value=5)
        batch_spinbox = ttk.Spinbox(options_frame, from_=1, to=20, textvariable=self.batch_var, width=5)
        batch_spinbox.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # 延迟时间
        ttk.Label(options_frame, text="延迟(秒):").grid(row=2, column=2, sticky="e", padx=5, pady=5)
        self.delay_var = tk.IntVar(value=3)
        delay_spinbox = ttk.Spinbox(options_frame, from_=0, to=10, textvariable=self.delay_var, width=5)
        delay_spinbox.grid(row=2, column=3, sticky="w", padx=5, pady=5)
    
    def create_progress_frame(self):
        """创建进度显示框架"""
        progress_frame = ttk.LabelFrame(self, text="翻译进度")
        progress_frame.pack(fill="x", padx=10, pady=5, ipady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", 
                                          length=100, mode="determinate",
                                          variable=self.progress_var)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        
        self.status_var = tk.StringVar(value="准备就绪")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(padx=10, pady=5)
    
    def create_buttons_frame(self):
        """创建按钮框架"""
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_btn = ttk.Button(buttons_frame, text="开始翻译", command=self.start_translation)
        self.start_btn.pack(side="left", padx=5)
        
        self.cancel_btn = ttk.Button(buttons_frame, text="取消", command=self.cancel_translation, state="disabled")
        self.cancel_btn.pack(side="left", padx=5)
        
        self.view_btn = ttk.Button(buttons_frame, text="查看结果文件夹", command=self.open_target_folder, state="disabled")
        self.view_btn.pack(side="right", padx=5)
        
    def create_log_frame(self):
        """创建日志显示框架"""
        log_frame = ttk.LabelFrame(self, text="翻译日志")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 创建文本框和滚动条
        self.log_text = tk.Text(log_frame, height=10, width=80, wrap="word")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        
        # 设置为只读
        self.log_text.configure(state="disabled")
        
    def setup_dnd(self):
        """设置拖放功能"""
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_drop)
        
    def handle_drop(self, event):
        """处理拖放事件"""
        file_path = event.data
        # 清理路径字符串（移除{}或""等）
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
            
        if os.path.isfile(file_path):
            self.set_source_file(file_path)
        elif os.path.isdir(file_path):
            self.set_source_dir(file_path)
        else:
            self.log_message(f"错误: 无效的文件或目录路径: {file_path}")
            
    def browse_source_file(self):
        """浏览并选择源文件"""
        file_path = filedialog.askopenfilename(
            title="选择Markdown/MDX文件",
            filetypes=[("Markdown文件", "*.md"), ("MDX文件", "*.mdx"), ("所有文件", "*.*")]
        )
        if file_path:
            self.set_source_file(file_path)
            
    def browse_source_dir(self):
        """浏览并选择源目录"""
        dir_path = filedialog.askdirectory(title="选择包含Markdown/MDX文件的文件夹")
        if dir_path:
            self.set_source_dir(dir_path)
            
    def browse_target_dir(self):
        """浏览并选择目标目录"""
        dir_path = filedialog.askdirectory(title="选择翻译结果保存位置")
        if dir_path:
            self.target_dir = dir_path
            self.target_label.config(text=dir_path)
            self.log_message(f"设置目标文件夹: {dir_path}")
            
    def set_source_file(self, file_path):
        """设置源文件"""
        if not is_markdown_file(file_path):
            messagebox.showwarning("文件类型错误", f"选择的文件 {file_path} 不是Markdown或MDX文件。")
            return
            
        self.source_path = file_path
        self.is_single_file = True
        self.source_label.config(text=file_path)
        self.log_message(f"设置源文件: {file_path}")
        
        # 自动设置默认目标文件夹
        if not self.target_dir:
            parent_dir = os.path.dirname(file_path)
            lang_code = self.target_lang_var.get()
            default_target = os.path.join(parent_dir, lang_code)
            self.target_dir = default_target
            self.target_label.config(text=default_target)
            
    def set_source_dir(self, dir_path):
        """设置源目录"""
        self.source_path = dir_path
        self.is_single_file = False
        self.source_label.config(text=dir_path)
        self.log_message(f"设置源文件夹: {dir_path}")
        
        # 自动设置默认目标文件夹
        if not self.target_dir:
            parent_dir = os.path.dirname(dir_path)
            dir_name = os.path.basename(dir_path)
            lang_code = self.target_lang_var.get()
            default_target = os.path.join(parent_dir, f"{lang_code}_{dir_name}")
            self.target_dir = default_target
            self.target_label.config(text=default_target)
            
    def update_model_options(self, event=None):
        """更新模型选项"""
        provider = self.provider_var.get()
        models = DEFAULT_MODELS.get(provider, [])
        self.model_combobox['values'] = models
        if models:
            self.model_var.set(models[0])
        else:
            self.model_var.set("")
            
    def log_message(self, message):
        """添加消息到日志窗口"""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        logger.info(message)
        
    def update_progress(self, current, total):
        """更新进度条"""
        progress = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(progress)
        self.status_var.set(f"进度: {current}/{total} ({progress:.1f}%)")
        
    def open_target_folder(self):
        """打开目标文件夹"""
        if self.target_dir and os.path.exists(self.target_dir):
            if sys.platform == 'win32':
                os.startfile(self.target_dir)
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{self.target_dir}"')
            else:  # Linux
                os.system(f'xdg-open "{self.target_dir}"')
        else:
            messagebox.showwarning("错误", "目标文件夹不存在")
            
    def start_translation(self):
        """开始翻译过程"""
        if not self.source_path:
            messagebox.showwarning("错误", "请先选择源文件或文件夹")
            return
            
        if not self.target_dir:
            messagebox.showwarning("错误", "请选择目标文件夹")
            return
            
        # 获取选项
        provider = self.provider_var.get()
        model = self.model_var.get() or None
        source_lang = self.source_lang_var.get()
        target_lang = self.target_lang_var.get()
        batch_size = self.batch_var.get()
        delay = self.delay_var.get()
        
        # 确保目标文件夹存在
        os.makedirs(self.target_dir, exist_ok=True)
        
        # 更新UI状态
        self.translation_running = True
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.view_btn.config(state="disabled")
        
        # 清空日志
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.configure(state="disabled")
        
        # 在新线程中启动翻译
        threading.Thread(target=self._run_translation, 
                        args=(provider, model, source_lang, target_lang, batch_size, delay),
                        daemon=True).start()
        
    def _run_translation(self, provider, model, source_lang, target_lang, batch_size, delay):
        """在后台线程中执行翻译"""
        try:
            source_lang_name = LANGUAGES.get(source_lang, source_lang)
            target_lang_name = LANGUAGES.get(target_lang, target_lang)
            
            if source_lang == "auto":
                source_lang_name = "自动检测"
                
            self.log_message(f"开始翻译 ({source_lang_name} -> {target_lang_name})")
            self.log_message(f"LLM提供商: {provider}" + (f", 模型: {model}" if model else ""))
            
            if self.is_single_file:
                # 单文件翻译
                filename = os.path.basename(self.source_path)
                target_file = os.path.join(self.target_dir, filename)
                
                self.log_message(f"翻译单个文件: {self.source_path} -> {target_file}")
                self.status_var.set("正在翻译文件...")
                self.progress_var.set(50)  # 设置进度条为50%
                
                process_file(self.source_path, target_file, provider, model, delay, source_lang, target_lang)
                
                self.progress_var.set(100)  # 设置进度条为100%
                self.log_message(f"翻译完成，结果保存到: {target_file}")
                
            else:
                # 目录翻译
                self.log_message(f"翻译目录: {self.source_path} -> {self.target_dir}")
                self.log_message(f"批处理大小: {batch_size}, 延迟: {delay}秒")
                
                # 创建一个进度回调函数
                def progress_callback(current, total):
                    # 在主线程中更新UI
                    self.master.after(0, lambda: self.update_progress(current, total))
                
                # 使用支持进度回调的translate_directory_with_progress函数
                success = translate_directory_with_progress(
                    self.source_path, 
                    self.target_dir, 
                    provider, 
                    model, 
                    max_workers=batch_size, 
                    delay=delay,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    progress_callback=progress_callback
                )
                
                if success:
                    self.log_message(f"翻译完成，结果保存在: {self.target_dir}")
                else:
                    self.log_message("翻译任务失败")
                    
            # 输出令牌使用统计（如果可用）
            token_tracker = get_token_tracker()
            if token_tracker:
                self.log_message(f"令牌使用统计:")
                self.log_message(f"总令牌: {token_tracker.total_tokens}")
                self.log_message(f"估计成本: ${token_tracker.total_cost:.6f}")
                self.log_message(f"请求数: {len(token_tracker.requests)}")
                
            self.master.after(0, self._translation_completed)
                
        except Exception as e:
            error_msg = f"翻译过程中出错: {str(e)}"
            logger.error(error_msg)
            self.master.after(0, lambda: self.log_message(error_msg))
            self.master.after(0, self._translation_failed)
            
    def _translation_completed(self):
        """翻译完成后更新UI"""
        self.translation_running = False
        self.start_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.view_btn.config(state="normal")
        self.status_var.set("翻译完成")
        messagebox.showinfo("完成", "翻译任务已完成")
        
    def _translation_failed(self):
        """翻译失败后更新UI"""
        self.translation_running = False
        self.start_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.status_var.set("翻译失败")
        
    def cancel_translation(self):
        """取消翻译过程"""
        if self.translation_running:
            self.translation_running = False
            self.log_message("正在取消翻译...")
            self.status_var.set("已取消")
            self.start_btn.config(state="normal")
            self.cancel_btn.config(state="disabled")
            
class TranslateApp(TkinterDnD.Tk):
    """翻译应用程序的主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Markdown/MDX文件翻译工具")
        self.geometry("800x700")
        self.minsize(800, 600)
        
        # 设置样式
        self.style = ttk.Style()
        try:
            # 尝试使用更现代的主题
            self.style.theme_use("clam")
        except tk.TclError:
            pass  # 使用默认主题
            
        # 创建主框架
        main_frame = TranslateFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 检查环境变量和依赖
        self.check_environment()
        
    def check_environment(self):
        """检查环境变量和依赖"""
        required_envs = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "siliconflow": "SILICONFLOW_API_KEY"
        }
        
        missing_envs = []
        for provider, env_var in required_envs.items():
            if not os.getenv(env_var):
                missing_envs.append(f"{provider} ({env_var})")
                
        if missing_envs:
            warning_msg = "以下LLM提供商的API密钥未设置:\n" + "\n".join(missing_envs)
            warning_msg += "\n\n可在.env文件中设置API密钥，或跳过这些提供商的选择。"
            messagebox.showwarning("API密钥缺失", warning_msg)
            
def main():
    """启动应用程序"""
    app = TranslateApp()
    app.mainloop()
    
if __name__ == "__main__":
    main() 