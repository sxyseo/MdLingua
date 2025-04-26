#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Markdown/MDX文件翻译工具GUI界面 (PySide6版本)

这个程序提供了一个图形用户界面，允许用户：
1. 选择文件或文件夹进行翻译
2. 选择源语言和目标语言
3. 选择LLM提供商和模型
4. 选择输出文件夹
5. 查看翻译进度和结果

依赖:
- PySide6: pip install pyside6
"""

import os
import sys
import threading
import logging
from pathlib import Path

# 添加调试信息
print("Python版本:", sys.version)
print("当前工作目录:", os.getcwd())
print("系统平台:", sys.platform)

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QPushButton, QProgressBar, QComboBox, QSpinBox,
        QTextEdit, QFileDialog, QMessageBox, QGroupBox, QFrame,
        QSplitter, QStyle, QGridLayout
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QMimeData
    from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon, QDesktopServices
    print("成功导入PySide6")
except ImportError as e:
    print(f"导入PySide6失败: {e}")
    print("请安装PySide6: pip install pyside6")
    sys.exit(1)

# 导入翻译功能
try:
    print("正在导入翻译模块...")
    from translate_md import (
        LANGUAGES, 
        is_markdown_file, 
        process_file, 
        get_token_tracker
    )
    from translate_utils import translate_directory_with_progress
    print("成功导入翻译模块")
except ImportError as e:
    print(f"导入翻译模块失败: {e}")
    if 'PySide6.QtWidgets' in sys.modules:
        QMessageBox.critical(None, "缺少依赖", "无法导入翻译模块。请确保translate_md.py在同一目录下。")
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
    "gemini": ["gemini-2.0-flash", "gemini-2.0-flash-vision", "gemini-1.5-flash"],
    "siliconflow": ["mixtral-8x7b", "qwen-7b", "chatglm2-6b", "deepseek-ai/DeepSeek-R1"],
    "local": ["Qwen/Qwen2.5-32B-Instruct-AWQ"]
}

class TranslationWorker(QThread):
    """翻译工作线程"""
    progressUpdated = Signal(int, int)  # 当前进度, 总数
    translationCompleted = Signal()     # 翻译完成信号
    translationFailed = Signal(str)     # 翻译失败信号
    messageLogged = Signal(str)         # 日志消息信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_path = None
        self.target_dir = None
        self.is_single_file = False
        self.provider = "anthropic"
        self.model = None
        self.source_lang = "auto"
        self.target_lang = "en"
        self.batch_size = 5
        self.delay = 3
        self.is_running = False
        
    def setup(self, source_path, target_dir, is_single_file, provider, model, 
             source_lang, target_lang, batch_size, delay):
        """设置翻译参数"""
        self.source_path = source_path
        self.target_dir = target_dir
        self.is_single_file = is_single_file
        self.provider = provider
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.delay = delay
        
    def run(self):
        """执行翻译任务"""
        self.is_running = True
        
        try:
            source_lang_name = LANGUAGES.get(self.source_lang, self.source_lang)
            target_lang_name = LANGUAGES.get(self.target_lang, self.target_lang)
            
            if self.source_lang == "auto":
                source_lang_name = "自动检测"
                
            self.messageLogged.emit(f"开始翻译 ({source_lang_name} -> {target_lang_name})")
            self.messageLogged.emit(f"LLM提供商: {self.provider}" + (f", 模型: {self.model}" if self.model else ""))
            
            if self.is_single_file:
                # 单文件翻译
                filename = os.path.basename(self.source_path)
                target_file = os.path.join(self.target_dir, filename)
                
                self.messageLogged.emit(f"翻译单个文件: {self.source_path} -> {target_file}")
                self.progressUpdated.emit(0, 1)  # 开始
                
                # 处理文件，返回布尔值表示是否成功
                success = process_file(self.source_path, target_file, self.provider, self.model, 
                          self.delay, self.source_lang, self.target_lang)
                
                self.progressUpdated.emit(1, 1)  # 完成
                
                if success:
                    self.messageLogged.emit(f"翻译完成，结果保存到: {target_file}")
                else:
                    error_msg = f"翻译失败，无法生成翻译文件: {target_file}"
                    self.messageLogged.emit(error_msg)
                    self.translationFailed.emit(error_msg)
                    return
                
            else:
                # 目录翻译
                self.messageLogged.emit(f"翻译目录: {self.source_path} -> {self.target_dir}")
                self.messageLogged.emit(f"批处理大小: {self.batch_size}, 延迟: {self.delay}秒")
                
                # 进度回调函数
                def progress_callback(current, total):
                    if self.is_running:  # 只有当任务运行时才发送信号
                        self.progressUpdated.emit(current, total)
                
                # 使用支持进度回调的目录翻译函数，现在返回元组(成功标志, 成功文件数, 失败文件数)
                success, success_count, failure_count = translate_directory_with_progress(
                    self.source_path, 
                    self.target_dir, 
                    self.provider, 
                    self.model, 
                    max_workers=self.batch_size, 
                    delay=self.delay,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    progress_callback=progress_callback
                )
                
                if success:
                    if failure_count > 0:
                        self.messageLogged.emit(f"翻译部分完成，成功: {success_count}个文件，失败: {failure_count}个文件")
                        self.messageLogged.emit(f"结果保存在: {self.target_dir}")
                    else:
                        self.messageLogged.emit(f"翻译完成，所有文件翻译成功，结果保存在: {self.target_dir}")
                else:
                    self.messageLogged.emit("翻译任务失败")
                    self.translationFailed.emit("翻译目录过程失败")
                    return
                    
            # 输出令牌使用统计（如果可用）
            token_tracker = get_token_tracker()
            if token_tracker:
                self.messageLogged.emit(f"令牌使用统计:")
                self.messageLogged.emit(f"总令牌: {token_tracker.total_tokens}")
                self.messageLogged.emit(f"估计成本: ${token_tracker.total_cost:.6f}")
                self.messageLogged.emit(f"请求数: {len(token_tracker.requests)}")
                
            # 在这里统一调用translationCompleted信号
            self.translationCompleted.emit()
                
        except Exception as e:
            error_msg = f"翻译过程中出错: {str(e)}"
            logger.error(error_msg)
            self.messageLogged.emit(error_msg)
            self.translationFailed.emit(str(e))
        
        finally:
            self.is_running = False
            
    def stop(self):
        """停止翻译任务"""
        self.is_running = False
        self.messageLogged.emit("正在停止翻译任务...")
        self.wait(1000)  # 等待1秒钟让线程停止

class TranslateWidget(QWidget):
    """翻译应用程序的主界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_path = None
        self.target_dir = None
        self.is_single_file = False
        self.translation_worker = TranslationWorker()
        
        # 连接信号
        self.translation_worker.progressUpdated.connect(self.update_progress)
        self.translation_worker.translationCompleted.connect(self.translation_completed)
        self.translation_worker.translationFailed.connect(self.translation_failed)
        self.translation_worker.messageLogged.connect(self.log_message)
        
        self.setup_ui()
        self.setup_drop_support()
        
    def setup_ui(self):
        """设置用户界面"""
        main_layout = QVBoxLayout(self)
        
        # 创建各个部分
        self.create_source_frame()
        self.create_target_frame()
        self.create_options_frame()
        self.create_progress_frame()
        self.create_buttons_frame()
        self.create_log_frame()
        
        # 添加到主布局
        main_layout.addWidget(self.source_frame)
        main_layout.addWidget(self.target_frame)
        main_layout.addWidget(self.options_frame)
        main_layout.addWidget(self.progress_frame)
        main_layout.addWidget(self.buttons_frame)
        main_layout.addWidget(self.log_frame, 1)  # 日志部分占据剩余空间
        
        self.setLayout(main_layout)
        
    def create_source_frame(self):
        """创建源文件/文件夹选择框架"""
        self.source_frame = QGroupBox("源文件/文件夹")
        layout = QHBoxLayout()
        
        self.source_label = QLabel("拖放文件或文件夹到这里，或点击浏览...")
        
        self.source_file_btn = QPushButton("选择文件")
        self.source_file_btn.clicked.connect(self.browse_source_file)
        
        self.source_dir_btn = QPushButton("选择文件夹")
        self.source_dir_btn.clicked.connect(self.browse_source_dir)
        
        layout.addWidget(self.source_label, 1)
        layout.addWidget(self.source_file_btn)
        layout.addWidget(self.source_dir_btn)
        
        self.source_frame.setLayout(layout)
        
    def create_target_frame(self):
        """创建目标文件夹选择框架"""
        self.target_frame = QGroupBox("目标文件夹")
        layout = QHBoxLayout()
        
        self.target_label = QLabel("拖放文件夹到这里，或点击浏览...")
        
        self.target_btn = QPushButton("浏览...")
        self.target_btn.clicked.connect(self.browse_target_dir)
        
        layout.addWidget(self.target_label, 1)
        layout.addWidget(self.target_btn)
        
        self.target_frame.setLayout(layout)
        
        # 为目标文件夹框架设置拖拽支持
        self.target_frame.setAcceptDrops(True)
        self.target_frame.dragEnterEvent = self.target_dragEnterEvent
        self.target_frame.dropEvent = self.target_dropEvent
        
    def target_dragEnterEvent(self, event: QDragEnterEvent):
        """目标文件夹拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        
    def target_dropEvent(self, event: QDropEvent):
        """目标文件夹放置事件"""
        urls = event.mimeData().urls()
        if urls:
            dir_path = urls[0].toLocalFile()
            if os.path.isdir(dir_path):
                self.target_dir = dir_path
                self.target_label.setText(dir_path)
                self.log_message(f"设置目标文件夹: {dir_path}")
            else:
                # 如果拖拽的是文件，则使用其所在目录作为目标目录
                dir_path = os.path.dirname(dir_path)
                self.target_dir = dir_path
                self.target_label.setText(dir_path)
                self.log_message(f"设置目标文件夹: {dir_path}（从文件路径推断）")
        
    def create_options_frame(self):
        """创建翻译选项框架"""
        self.options_frame = QGroupBox("翻译选项")
        layout = QGridLayout()
        
        # 源语言选择
        layout.addWidget(QLabel("源语言:"), 0, 0)
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItem("自动检测", "auto")
        for code, name in LANGUAGES.items():
            self.source_lang_combo.addItem(f"{name} ({code})", code)
        layout.addWidget(self.source_lang_combo, 0, 1)
        
        # 目标语言选择
        layout.addWidget(QLabel("目标语言:"), 0, 2)
        self.target_lang_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self.target_lang_combo.addItem(f"{name} ({code})", code)
        # 默认设置为英文
        self.target_lang_combo.setCurrentText(f"{LANGUAGES.get('en', '英文')} (en)")
        layout.addWidget(self.target_lang_combo, 0, 3)
        
        # LLM提供商选择
        layout.addWidget(QLabel("LLM提供商:"), 1, 0)
        self.provider_combo = QComboBox()
        for code, name in PROVIDERS.items():
            self.provider_combo.addItem(name, code)
        self.provider_combo.currentIndexChanged.connect(self.update_model_options)
        layout.addWidget(self.provider_combo, 1, 1)
        
        # 模型选择
        layout.addWidget(QLabel("模型:"), 1, 2)
        self.model_combo = QComboBox()
        layout.addWidget(self.model_combo, 1, 3)
        self.update_model_options()  # 初始化模型选项
        
        # 批处理大小（仅目录模式）
        layout.addWidget(QLabel("批处理大小:"), 2, 0)
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 20)
        self.batch_spinbox.setValue(5)
        layout.addWidget(self.batch_spinbox, 2, 1)
        
        # 延迟时间
        layout.addWidget(QLabel("延迟(秒):"), 2, 2)
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(0, 10)
        self.delay_spinbox.setValue(3)
        layout.addWidget(self.delay_spinbox, 2, 3)
        
        self.options_frame.setLayout(layout)
        
    def create_progress_frame(self):
        """创建进度显示框架"""
        self.progress_frame = QGroupBox("翻译进度")
        layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        self.progress_frame.setLayout(layout)
        
    def create_buttons_frame(self):
        """创建按钮框架"""
        self.buttons_frame = QFrame()
        layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始翻译")
        self.start_btn.clicked.connect(self.start_translation)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_translation)
        self.cancel_btn.setEnabled(False)
        
        self.view_btn = QPushButton("查看结果文件夹")
        self.view_btn.clicked.connect(self.open_target_folder)
        self.view_btn.setEnabled(False)
        
        layout.addWidget(self.start_btn)
        layout.addWidget(self.cancel_btn)
        layout.addStretch(1)
        layout.addWidget(self.view_btn)
        
        self.buttons_frame.setLayout(layout)
        
    def create_log_frame(self):
        """创建日志显示框架"""
        self.log_frame = QGroupBox("翻译日志")
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 10))
        
        layout.addWidget(self.log_text)
        
        self.log_frame.setLayout(layout)
        
    def setup_drop_support(self):
        """设置拖放支持"""
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        """放置事件"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path):
                self.set_source_file(file_path)
            elif os.path.isdir(file_path):
                self.set_source_dir(file_path)
        
    def browse_source_file(self):
        """浏览并选择源文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择Markdown/MDX文件", 
            "", 
            "Markdown文件 (*.md);;MDX文件 (*.mdx);;所有文件 (*.*)"
        )
        if file_path:
            self.set_source_file(file_path)
            
    def browse_source_dir(self):
        """浏览并选择源目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择包含Markdown/MDX文件的文件夹")
        if dir_path:
            self.set_source_dir(dir_path)
            
    def browse_target_dir(self):
        """浏览并选择目标目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择翻译结果保存位置")
        if dir_path:
            self.target_dir = dir_path
            self.target_label.setText(dir_path)
            self.log_message(f"设置目标文件夹: {dir_path}")
            
    def set_source_file(self, file_path):
        """设置源文件"""
        if not is_markdown_file(file_path):
            QMessageBox.warning(self, "文件类型错误", f"选择的文件 {file_path} 不是Markdown或MDX文件。")
            return
            
        self.source_path = file_path
        self.is_single_file = True
        self.source_label.setText(file_path)
        self.log_message(f"设置源文件: {file_path}")
        
        # 自动设置默认目标文件夹
        if not self.target_dir:
            parent_dir = os.path.dirname(file_path)
            lang_code = self.target_lang_combo.currentData()
            default_target = os.path.join(parent_dir, lang_code)
            self.target_dir = default_target
            self.target_label.setText(default_target)
            
    def set_source_dir(self, dir_path):
        """设置源目录"""
        self.source_path = dir_path
        self.is_single_file = False
        self.source_label.setText(dir_path)
        self.log_message(f"设置源文件夹: {dir_path}")
        
        # 自动设置默认目标文件夹
        if not self.target_dir:
            parent_dir = os.path.dirname(dir_path)
            dir_name = os.path.basename(dir_path)
            lang_code = self.target_lang_combo.currentData()
            default_target = os.path.join(parent_dir, f"{lang_code}_{dir_name}")
            self.target_dir = default_target
            self.target_label.setText(default_target)
            
    def update_model_options(self):
        """更新模型选项"""
        provider = self.provider_combo.currentData()
        self.model_combo.clear()
        models = DEFAULT_MODELS.get(provider, [])
        for model in models:
            self.model_combo.addItem(model)
            
    def log_message(self, message):
        """添加消息到日志窗口"""
        self.log_text.append(message)
        logger.info(message)
        
    def update_progress(self, current, total):
        """更新进度条"""
        progress = (current / total) * 100 if total > 0 else 0
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(f"进度: {current}/{total} ({progress:.1f}%)")
        
    def open_target_folder(self):
        """打开目标文件夹"""
        if self.target_dir and os.path.exists(self.target_dir):
            url = QUrl.fromLocalFile(self.target_dir)
            try:
                # 尝试使用QDesktopServices打开
                success = QDesktopServices.openUrl(url)
                if not success:
                    # 如果失败，尝试使用系统命令
                    if sys.platform == 'win32':
                        os.startfile(self.target_dir)
                    elif sys.platform == 'darwin':  # macOS
                        os.system(f'open "{self.target_dir}"')
                    else:  # Linux
                        os.system(f'xdg-open "{self.target_dir}"')
                    self.log_message(f"使用系统命令打开文件夹: {self.target_dir}")
            except Exception as e:
                self.log_message(f"打开文件夹时出错: {str(e)}")
                QMessageBox.warning(self, "错误", f"无法打开目标文件夹: {str(e)}")
        else:
            QMessageBox.warning(self, "错误", "目标文件夹不存在")
            
    def start_translation(self):
        """开始翻译过程"""
        if not self.source_path:
            QMessageBox.warning(self, "错误", "请先选择源文件或文件夹")
            return
            
        if not self.target_dir:
            QMessageBox.warning(self, "错误", "请选择目标文件夹")
            return
            
        # 获取选项
        provider = self.provider_combo.currentData()
        model = self.model_combo.currentText() if self.model_combo.count() > 0 else None
        source_lang = self.source_lang_combo.currentData()
        target_lang = self.target_lang_combo.currentData()
        batch_size = self.batch_spinbox.value()
        delay = self.delay_spinbox.value()
        
        # 确保目标文件夹存在
        os.makedirs(self.target_dir, exist_ok=True)
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.view_btn.setEnabled(False)
        
        # 清空日志
        self.log_text.clear()
        
        # 设置进度条
        self.progress_bar.setValue(0)
        self.status_label.setText("正在准备翻译...")
        
        # 设置并启动翻译工作线程
        self.translation_worker.setup(
            self.source_path, self.target_dir, self.is_single_file,
            provider, model, source_lang, target_lang, batch_size, delay
        )
        self.translation_worker.start()
        
    def translation_completed(self):
        """翻译完成后更新UI"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.view_btn.setEnabled(True)
        self.status_label.setText("翻译完成")
        
        # 检查是否为目录翻译，并且有失败的文件
        if not self.translation_worker.is_single_file:
            # 从日志中提取失败文件数量信息
            log_text = self.log_text.toPlainText()
            if "失败:" in log_text and "个文件" in log_text:
                # 如果有失败的文件，在弹窗中显示失败信息
                failure_info = ""
                for line in log_text.split("\n"):
                    if "失败:" in line and "个文件" in line:
                        failure_info = line.strip()
                        break
                
                if failure_info:
                    QMessageBox.information(self, "完成", f"翻译任务已完成，{failure_info}")
                    return
        
        # 如果没有失败信息或者是单文件翻译，显示普通完成消息
        QMessageBox.information(self, "完成", "翻译任务已完成")
        
    def translation_failed(self, error_msg):
        """翻译失败后更新UI"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("翻译失败")
        QMessageBox.critical(self, "错误", f"翻译失败: {error_msg}")
        
    def cancel_translation(self):
        """取消翻译过程"""
        if self.translation_worker.isRunning():
            self.translation_worker.stop()
            self.log_message("已取消翻译任务")
            self.status_label.setText("已取消")
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

class TranslateApp(QMainWindow):
    """翻译应用程序的主窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("Markdown/MDX文件翻译工具")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        
        # 设置应用图标
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MdLingua.png")
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
            # 为整个应用程序设置图标，这样会在任务栏和标题栏显示
            QApplication.setWindowIcon(app_icon)
            print(f"设置应用图标: {icon_path}")
        else:
            print(f"图标文件不存在: {icon_path}")
        
        self.central_widget = TranslateWidget(self)
        self.setCentralWidget(self.central_widget)
        
        # 检查环境变量
        QTimer.singleShot(100, self.check_environment)
        
    def check_environment(self):
        """检查环境变量和依赖"""
        # 如果启用了模拟翻译，则不检查API密钥
        if os.getenv("MOCK_TRANSLATION", "").lower() == "true":
            print("已启用模拟翻译模式，跳过API密钥检查")
            return
        
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
            QMessageBox.warning(self, "API密钥缺失", warning_msg)

def main():
    """主函数"""
    try:
        print("开始初始化应用程序...")
        app = QApplication(sys.argv)
        
        # 设置样式
        app.setStyle("Fusion")
        
        # 设置应用图标
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MdLingua.png")
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            app.setWindowIcon(app_icon)
            print(f"全局设置应用图标: {icon_path}")
        
        # 创建和显示主窗口
        main_window = TranslateApp()
        main_window.show()
        
        print("初始化应用程序成功，开始主循环")
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"应用程序启动错误: {e}")
        import traceback
        traceback.print_exc()
        if 'app' in locals():
            QMessageBox.critical(None, "启动错误", f"应用程序启动失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 