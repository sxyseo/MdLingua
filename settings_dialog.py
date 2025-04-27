#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MdLingua设置对话框 - 用于配置翻译工具选项
"""

import os
import sys
import time
from pathlib import Path

try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
        QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
        QPushButton, QGroupBox, QFormLayout, QFileDialog,
        QMessageBox, QGridLayout, QWidget, QApplication
    )
    from PySide6.QtCore import Qt, QSize
    from PySide6.QtGui import QFont, QClipboard
except ImportError as e:
    print(f"导入PySide6失败: {str(e)}")
    sys.exit(1)

class SettingsDialog(QDialog):
    """设置对话框"""
    
    def __init__(self, config_manager, parent=None):
        """
        初始化设置对话框
        
        Args:
            config_manager: 配置管理器实例
            parent: 父窗口
        """
        super().__init__(parent)
        self.config_manager = config_manager
        self.api_key_fields = {}
        self.provider_env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY", 
            "gemini": "GOOGLE_API_KEY",
            "siliconflow": "SILICONFLOW_API_KEY"
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("MdLingua设置")
        self.setMinimumSize(550, 400)
        
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_general_tab(), "常规设置")
        self.tabs.addTab(self.create_api_tab(), "API设置")
        self.tabs.addTab(self.create_appearance_tab(), "外观设置")
        self.tabs.addTab(self.create_advanced_tab(), "高级设置")
        
        layout.addWidget(self.tabs)
        
        # 底部按钮
        buttons_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("重置为默认")
        self.reset_btn.clicked.connect(self.reset_to_default)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_settings)
        self.save_btn.setDefault(True)
        
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.cancel_btn)
        buttons_layout.addWidget(self.save_btn)
        
        layout.addLayout(buttons_layout)
        
        # 加载当前设置
        self.load_settings()
        
    def create_general_tab(self):
        """创建常规设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 翻译设置组
        translation_group = QGroupBox("翻译设置")
        translation_layout = QFormLayout()
        
        # 源语言设置
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItem("自动检测", "auto")
        # 目标语言设置
        self.target_lang_combo = QComboBox()
        
        # 从translate_md模块获取语言列表的代码将在load_settings中处理
        
        translation_layout.addRow("默认源语言:", self.source_lang_combo)
        translation_layout.addRow("默认目标语言:", self.target_lang_combo)
        
        # LLM提供商设置
        self.provider_combo = QComboBox()
        # PROVIDERS变量将在load_settings中填充
        
        translation_layout.addRow("默认LLM提供商:", self.provider_combo)
        
        # 批处理设置
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 20)
        self.batch_spinbox.setValue(5)
        translation_layout.addRow("默认批处理大小:", self.batch_spinbox)
        
        # 延迟设置
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(0, 10)
        self.delay_spinbox.setValue(3)
        translation_layout.addRow("默认延迟(秒):", self.delay_spinbox)
        
        translation_group.setLayout(translation_layout)
        layout.addWidget(translation_group)
        
        # 最近使用项目设置
        history_group = QGroupBox("历史记录")
        history_layout = QFormLayout()
        
        self.max_recent_spinbox = QSpinBox()
        self.max_recent_spinbox.setRange(0, 20)
        self.max_recent_spinbox.setValue(5)
        history_layout.addRow("最近使用项目数量:", self.max_recent_spinbox)
        
        self.clear_history_btn = QPushButton("清除历史记录")
        self.clear_history_btn.clicked.connect(self.clear_history)
        history_layout.addRow("", self.clear_history_btn)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        layout.addStretch()
        return tab
        
    def create_api_tab(self):
        """创建API设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("设置各LLM提供商的API密钥。这些密钥将安全地保存在本地配置文件中。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.api_keys_group = QGroupBox("API密钥")
        api_keys_layout = QFormLayout()
        
        # 为每个提供商创建输入字段
        for provider, env_var in self.provider_env_vars.items():
            line_edit = QLineEdit()
            line_edit.setEchoMode(QLineEdit.Password)
            line_edit.setPlaceholderText(f"从环境变量 {env_var} 读取")
            
            # 存储引用以便后续访问
            self.api_key_fields[provider] = line_edit
            
            # 使用更友好的名称
            provider_names = {
                "openai": "OpenAI",
                "anthropic": "Anthropic",
                "azure": "Azure OpenAI",
                "deepseek": "DeepSeek",
                "gemini": "Google Gemini",
                "siliconflow": "SiliconFlow"
            }
            
            api_keys_layout.addRow(f"{provider_names.get(provider, provider)}:", line_edit)
        
        self.api_keys_group.setLayout(api_keys_layout)
        layout.addWidget(self.api_keys_group)
        
        # 环境变量说明
        env_info = QLabel("注意: 也可以通过环境变量或.env文件设置API密钥。应用程序会优先使用环境变量中的密钥。")
        env_info.setWordWrap(True)
        layout.addWidget(env_info)
        
        layout.addStretch()
        return tab
        
    def create_appearance_tab(self):
        """创建外观设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        appearance_group = QGroupBox("界面外观")
        appearance_layout = QFormLayout()
        
        # 主题设置
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("跟随系统", "system")
        self.theme_combo.addItem("浅色", "light")
        self.theme_combo.addItem("深色", "dark")
        appearance_layout.addRow("主题:", self.theme_combo)
        
        # 字体大小设置
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(8, 18)
        self.font_size_spinbox.setValue(10)
        appearance_layout.addRow("字体大小:", self.font_size_spinbox)
        
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        
        layout.addStretch()
        return tab
        
    def create_advanced_tab(self):
        """创建高级设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 日志设置
        log_group = QGroupBox("日志设置")
        log_layout = QFormLayout()
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItem("调试", "DEBUG")
        self.log_level_combo.addItem("信息", "INFO")
        self.log_level_combo.addItem("警告", "WARNING")
        self.log_level_combo.addItem("错误", "ERROR")
        log_layout.addRow("日志级别:", self.log_level_combo)
        
        self.auto_save_logs_check = QCheckBox("自动保存日志到文件")
        log_layout.addRow("", self.auto_save_logs_check)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 配置文件信息
        config_group = QGroupBox("配置信息")
        config_layout = QVBoxLayout()
        
        config_path_layout = QHBoxLayout()
        config_path_label = QLabel("配置文件位置:")
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setReadOnly(True)
        self.config_path_edit.setText(self.config_manager.config_file)
        
        # 添加复制按钮
        self.copy_path_btn = QPushButton("复制路径")
        self.copy_path_btn.clicked.connect(self.copy_config_path)
        
        self.open_config_folder_btn = QPushButton("打开文件夹")
        self.open_config_folder_btn.clicked.connect(self.open_config_folder)
        
        config_path_layout.addWidget(config_path_label)
        config_path_layout.addWidget(self.config_path_edit, 1)
        config_path_layout.addWidget(self.copy_path_btn)
        config_path_layout.addWidget(self.open_config_folder_btn)
        
        config_layout.addLayout(config_path_layout)
        
        # 添加配置文件状态信息
        config_status_layout = QHBoxLayout()
        config_status_label = QLabel("配置文件状态:")
        
        if os.path.exists(self.config_manager.config_file):
            config_size = os.path.getsize(self.config_manager.config_file)
            config_time = os.path.getmtime(self.config_manager.config_file)
            config_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(config_time))
            status_text = f"文件大小: {config_size} 字节, 最后修改时间: {config_time_str}"
        else:
            status_text = "配置文件尚未创建，将在保存设置时创建"
            
        config_status_value = QLabel(status_text)
        config_status_value.setWordWrap(True)
        
        config_status_layout.addWidget(config_status_label)
        config_status_layout.addWidget(config_status_value, 1)
        
        config_layout.addLayout(config_status_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        layout.addStretch()
        return tab
        
    def load_settings(self):
        """从配置管理器加载设置"""
        try:
            # 从父窗口获取语言和提供商列表
            try:
                from translate_md import LANGUAGES
            except ImportError:
                # 如果无法导入，使用默认语言列表
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
                    "ru": "俄语"
                }
                
            try:
                from translate_gui import PROVIDERS
            except ImportError:
                # 如果无法导入，使用默认提供商列表
                PROVIDERS = {
                    "anthropic": "Anthropic Claude",
                    "openai": "OpenAI GPT",
                    "azure": "Azure OpenAI",
                    "deepseek": "DeepSeek",
                    "gemini": "Google Gemini",
                    "siliconflow": "SiliconFlow",
                    "local": "本地模型"
                }
            
            # 填充语言列表
            for code, name in LANGUAGES.items():
                if code != "auto":  # 自动检测已经添加到源语言中
                    self.source_lang_combo.addItem(f"{name} ({code})", code)
                self.target_lang_combo.addItem(f"{name} ({code})", code)
            
            # 填充提供商列表
            for code, name in PROVIDERS.items():
                self.provider_combo.addItem(name, code)
            
            # 设置默认值
            source_lang = self.config_manager.get("source_lang", "auto")
            target_lang = self.config_manager.get("target_lang", "en")
            provider = self.config_manager.get("provider", "anthropic")
            
            # 设置源语言
            index = self.source_lang_combo.findData(source_lang)
            if index >= 0:
                self.source_lang_combo.setCurrentIndex(index)
                
            # 设置目标语言
            index = self.target_lang_combo.findData(target_lang)
            if index >= 0:
                self.target_lang_combo.setCurrentIndex(index)
                
            # 设置提供商
            index = self.provider_combo.findData(provider)
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)
                
            # 设置批处理和延迟
            self.batch_spinbox.setValue(self.config_manager.get("batch_size", 5))
            self.delay_spinbox.setValue(self.config_manager.get("delay", 3))
            
            # 设置最近使用项目数量
            self.max_recent_spinbox.setValue(self.config_manager.get("max_recent_items", 5))
            
            # 设置API密钥
            for provider in self.api_key_fields:
                self.api_key_fields[provider].setText(self.config_manager.get_api_key(provider))
                
            # 设置外观选项
            theme = self.config_manager.get("appearance.theme", "system")
            index = self.theme_combo.findData(theme)
            if index >= 0:
                self.theme_combo.setCurrentIndex(index)
                
            self.font_size_spinbox.setValue(self.config_manager.get("appearance.font_size", 10))
            
            # 设置高级选项
            log_level = self.config_manager.get("log_level", "INFO")
            index = self.log_level_combo.findData(log_level)
            if index >= 0:
                self.log_level_combo.setCurrentIndex(index)
                
            self.auto_save_logs_check.setChecked(self.config_manager.get("auto_save_logs", True))
            
        except Exception as e:
            QMessageBox.warning(self, "加载设置错误", f"加载设置时出错: {str(e)}")
    
    def save_settings(self):
        """保存设置到配置管理器"""
        try:
            # 保存常规设置
            self.config_manager.set("source_lang", self.source_lang_combo.currentData())
            self.config_manager.set("target_lang", self.target_lang_combo.currentData())
            self.config_manager.set("provider", self.provider_combo.currentData())
            self.config_manager.set("batch_size", self.batch_spinbox.value())
            self.config_manager.set("delay", self.delay_spinbox.value())
            self.config_manager.set("max_recent_items", self.max_recent_spinbox.value())
            
            # 保存API密钥
            for provider, field in self.api_key_fields.items():
                api_key = field.text().strip()
                if api_key:
                    self.config_manager.set_api_key(provider, api_key)
            
            # 保存外观设置
            self.config_manager.set("appearance.theme", self.theme_combo.currentData())
            self.config_manager.set("appearance.font_size", self.font_size_spinbox.value())
            
            # 保存高级设置
            self.config_manager.set("log_level", self.log_level_combo.currentData())
            self.config_manager.set("auto_save_logs", self.auto_save_logs_check.isChecked())
            
            # 写入配置文件
            self.config_manager.save_config()
            
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "保存设置错误", f"保存设置时出错: {str(e)}")
    
    def reset_to_default(self):
        """重置设置为默认值"""
        reply = QMessageBox.question(
            self, 
            "确认重置", 
            "确定要将所有设置重置为默认值吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config_manager.reset_to_default()
            self.load_settings()
    
    def clear_history(self):
        """清除历史记录"""
        reply = QMessageBox.question(
            self, 
            "确认清除", 
            "确定要清除所有历史记录吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config_manager.set("recent_sources", [])
            self.config_manager.set("recent_targets", [])
            self.config_manager.save_config()
            QMessageBox.information(self, "已清除", "历史记录已清除")
    
    def open_config_folder(self):
        """打开配置文件所在文件夹"""
        try:
            folder = os.path.dirname(os.path.abspath(self.config_manager.config_file))
            if os.path.exists(folder):
                # 根据系统打开文件夹
                if sys.platform == 'win32':
                    os.startfile(folder)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open "{folder}"')
                else:  # Linux
                    os.system(f'xdg-open "{folder}"')
            else:
                QMessageBox.warning(self, "文件夹不存在", f"配置文件夹 {folder} 不存在。\n将在保存设置时自动创建。")
        except Exception as e:
            QMessageBox.warning(self, "打开文件夹错误", f"无法打开配置文件夹: {str(e)}")
    
    def copy_config_path(self):
        """复制配置文件路径到剪贴板"""
        try:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.config_manager.config_file)
            QMessageBox.information(self, "已复制", "配置文件路径已复制到剪贴板")
        except Exception as e:
            QMessageBox.warning(self, "复制失败", f"无法复制路径: {str(e)}") 