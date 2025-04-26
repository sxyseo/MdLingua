# Markdown/MDX 文件翻译工具

这个工具用于自动将 Markdown/MDX 文件翻译成其他语言，并保持原始文件结构。它利用大型语言模型（LLM）进行高质量翻译，适合文档国际化。

## 功能特点

- 批量翻译整个目录的 Markdown/MDX 文件
- 支持单个文件翻译
- 保持原始目录结构
- 保留 Markdown/MDX 格式（标题、列表、代码块、JSX组件等）
- 支持多种 LLM 提供商（OpenAI、Anthropic、Azure OpenAI、SiliconFlow 等）
- 支持指定特定模型（如 GPT-4o、Claude 3.5 Sonnet、Mixtral 等）
- 并行处理多个文件，提高效率
- 自动跳过非 Markdown/MDX 文件（直接复制）
- 详细的日志记录
- 翻译过程进度跟踪
- 令牌使用统计和成本估算
- 自动生成翻译报告
- 支持多语言互译（中文、英文、日语、韩语、法语、德语等）
- 特殊处理 MDX 文件中的 JSX 组件
- **提供图形用户界面(GUI)，支持拖放操作**

## 使用方法

### 图形界面(GUI)

最简单的使用方式是启动图形界面：

```bash
python translate_gui_tk.py
```

GUI界面提供以下功能：
- 拖放文件或文件夹到界面中进行翻译
- 选择源语言和目标语言
- 选择LLM提供商和模型
- 设置批处理大小和延迟时间
- 实时查看翻译进度和日志
- 翻译完成后直接打开目标文件夹

注：GUI模式需要安装额外的依赖：`pip install tkinterdnd2`

### 命令行基本用法

```bash
# 翻译目录
python translate_md.py --source <源文件夹> --target <目标文件夹> [--provider <LLM提供商>] [--model <模型名称>] 
                       [--source-lang <源语言>] [--target-lang <目标语言>]

# 翻译单个文件
python translate_md.py --file <源文件> --target <目标文件夹> [--provider <LLM提供商>] [--model <模型名称>]
                       [--source-lang <源语言>] [--target-lang <目标语言>]
```

### Windows 批处理脚本

```batch
# 翻译目录
translate.bat <源文件夹> <目标文件夹> [LLM提供商] [模型名称] [批处理大小] [延迟秒数] [源语言] [目标语言]

# 翻译单个文件
translate.bat --file <源文件> <目标文件夹> [LLM提供商] [模型名称] [延迟秒数] [源语言] [目标语言]
```

### Linux/macOS 脚本

```bash
# 翻译目录
./translate.sh <源文件夹> <目标文件夹> [LLM提供商] [模型名称] [批处理大小] [延迟秒数] [源语言] [目标语言]

# 翻译单个文件
./translate.sh --file <源文件> <目标文件夹> [LLM提供商] [模型名称] [延迟秒数] [源语言] [目标语言]
```

### 参数说明

- `--source`：源文件夹路径，包含需要翻译的 Markdown/MDX 文件
- `--file`：单个源文件路径，指定要翻译的 Markdown/MDX 文件
- `--target`：目标文件夹路径，用于保存翻译后的 Markdown/MDX 文件
- `--provider`：LLM 提供商，默认为 "anthropic"，可选值:
  - "openai"：使用 OpenAI 的 GPT 模型
  - "anthropic"：使用 Anthropic 的 Claude 模型
  - "azure"：使用 Azure OpenAI 服务
  - "deepseek"：使用 DeepSeek 模型
  - "gemini"：使用 Google 的 Gemini 模型
  - "siliconflow"：使用 SiliconFlow 的模型（中国本地化服务）
  - "local"：使用本地部署的模型
- `--model`：要使用的特定模型名称（可选）
- `--source-lang`：源语言代码，默认为 "auto"（自动检测），可指定如 "zh"（中文）, "en"（英文）, "ja"（日语）等
- `--target-lang`：目标语言代码，默认为 "en"（英文），可指定如 "zh"（中文）, "fr"（法语）, "de"（德语）等
- `--batch`：批处理大小，同时处理的文件数，默认为 5（仅目录模式适用）
- `--delay`：每个文件处理后的延迟时间(秒)，默认为 3 秒
- `--direction`：兼容旧版本的翻译方向，可选 "zh2en"(中译英) 或 "en2zh"(英译中)

### 支持的语言

工具支持多种语言互译，包括但不限于：
- 中文 (zh)
- 英文 (en)
- 日语 (ja)
- 韩语 (ko)
- 法语 (fr)
- 德语 (de)
- 西班牙语 (es)
- 意大利语 (it)
- 俄语 (ru)
- 葡萄牙语 (pt)
- 荷兰语 (nl)
- 阿拉伯语 (ar)
- 印地语 (hi)
- 越南语 (vi)
- 泰语 (th)
- 更多语言...

### 示例

```bash
# 使用 Anthropic Claude 翻译目录（中译英）
python translate_md.py --source ./docs --target ./en/docs --provider anthropic --source-lang zh --target-lang en

# 使用 OpenAI GPT-4o 翻译单个文件（中译英）
python translate_md.py --file ./docs/guide.md --target ./en/docs --provider openai --model gpt-4o --source-lang zh --target-lang en

# 使用 Azure OpenAI 服务翻译目录，自定义批处理大小和延迟（英译中）
python translate_md.py --source ./en/docs --target ./docs --provider azure --batch 3 --delay 5 --source-lang en --target-lang zh

# 使用 Google Gemini 翻译单个文件（英译中）
python translate_md.py --file ./en/content/intro.md --target ./content --provider gemini --model gemini-2.0-flash --source-lang en --target-lang zh

# 使用 Windows 批处理脚本翻译目录（中译英）
translate.bat docs en\docs openai gpt-4o 3 5 zh en

# 使用 Windows 批处理脚本翻译目录（日译韩）
translate.bat docs ko\docs anthropic claude-3-5-sonnet-20241022 3 5 ja ko

# 使用 Windows 批处理脚本翻译单个MDX文件（英译中）
translate.bat --file docs\intro.mdx zh\docs siliconflow deepseek-ai/DeepSeek-R1 3 en zh

# 使用 Linux/macOS 脚本翻译目录（中译英）
./translate.sh docs en/docs openai gpt-4o 5 3 zh en

# 使用 Linux/macOS 脚本翻译单个文件（英译法）
./translate.sh --file docs/getting-started.md fr/docs anthropic claude-3-5-sonnet-20241022 3 en fr

# 自动检测语言并翻译成西班牙语
./translate.sh --file article.md es/docs openai gpt-4o 3 auto es
```

### 测试
```bash
# 使用 Windows 命令行翻译单个文件（英译法）
./translate.bat --file test/entest.md test/frtest siliconflow deepseek-ai/DeepSeek-R1 3 en fr

# 使用 Linux/macOS 脚本翻译单个文件（英译法）
./translate.sh --file test/entest.md test/frtest anthropic claude-3-5-sonnet-20241022 3 en fr

# 自动检测语言并翻译成西班牙语
./translate.sh --file article.md es/docs openai gpt-4o 3 auto es
```

## 支持的模型

### OpenAI
- `gpt-4o`（默认）
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- `o1`（最强大的模型，但成本较高）

### Anthropic
- `claude-3-5-sonnet-20241022`（默认）
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-2.1`

### Azure OpenAI
- 根据您的 Azure 部署设置（默认：`gpt-4o-ms`）

### Google Gemini
- `gemini-2.0-flash`（默认）
- `gemini-2.0-flash-vision`（支持图像）
- `gemini-2.0-flash-exp`
- `gemini-1.5-flash`

### DeepSeek
- `deepseek-chat`（默认）

### SiliconFlow
- `mixtral-8x7b`（默认）
- `qwen-7b`
- `chatglm2-6b` 
- `deepseek-ai/DeepSeek-R1`
- 其他支持的开源模型

### 本地部署模型
- 任何兼容 OpenAI API 的本地部署模型

## 环境设置

为了使用此工具，您需要配置相应 LLM 提供商的 API 密钥。可以通过环境变量或 `.env` 文件设置：

```
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_ENDPOINT=https://...openai.azure.com/
AZURE_OPENAI_MODEL_DEPLOYMENT=gpt-4o-ms

# Google Gemini
GOOGLE_API_KEY=...

# DeepSeek
DEEPSEEK_API_KEY=...

# SiliconFlow
SILICONFLOW_API_KEY=...
SILICONFLOW_API_URL=https://api.siliconflow.cn/v1

# 本地模型
LOCAL_LLM_URL=http://localhost:8000/v1
```

支持以下位置的 `.env` 文件（按优先顺序）：
1. `.env.local`（用户特定的覆盖）
2. `.env`（项目默认值）
3. `.env.example`（示例配置）

## 翻译报告

翻译完成后，会在目标目录中生成一个 `translation-report.json` 文件，包含以下信息：
- 总文件数和 Markdown/MDX 文件数
- 完成时间
- 使用的提供商和模型
- 令牌使用量
- 估计成本
- 请求次数

## 测试工具

为方便测试，提供了一个测试脚本：

```bash
python test_translate.py
```

这将创建一些测试用的中文 Markdown 文件并尝试翻译它们，以验证翻译工具是否正常工作。

## 调试模式

在开发和测试阶段，可以设置环境变量 `MOCK_TRANSLATION=true` 启用模拟翻译，避免实际调用 LLM API。

## 注意事项

- 翻译大量文件可能需要较长时间，取决于 LLM 服务的响应速度
- 为避免 API 调用限制，工具在文件之间添加了延迟
- 确保您的 API 密钥有足够的配额/信用额度
- 翻译后建议人工检查重要文档，特别是专业术语和技术内容
- 不同的 LLM 提供商可能对某些类型的文档有不同的表现，建议尝试不同提供商找到最适合您需求的

## 依赖项

- Python 3.7+
- 根据所选 LLM 提供商可能需要安装额外的依赖：
  - OpenAI: `pip install openai`
  - Anthropic: `pip install anthropic`
  - Azure: `pip install azure-ai-openai`
  - Google Gemini: `pip install google-generativeai`
  - 对于 .env 文件支持: `pip install python-dotenv` 

