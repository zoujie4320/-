# RAG 数据预处理系统

一个可扩展的多格式文档向量化预处理系统，将 Word、PDF、Excel、图片等文档经过 **解析 → 语义分块 → 向量化 → 存储** 流水线处理，为 RAG（检索增强生成）系统的向量数据库提供格式化数据。

> **快速开始**: 将文档放入 `knowledge_base/` 文件夹，运行 `python main.py` 即可。

## 项目结构

```
RAG数据处理/
├── main.py                      # 主入口，自动扫描 knowledge_base/ 并处理
├── settings.py                  # ★ 部署配置（路径、密钥、数据库连接等）
├── config.py                    # 算法参数（分块阈值、chunk大小等）
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
├── knowledge_base/              # ★ 知识库源文件目录（文档放这里）
│   └── .gitkeep
├── output/                      # 向量化结果输出目录
├── processors/                  # 文档处理器
│   ├── base_processor.py        # 抽象基类（扩展入口）
│   ├── word_processor.py        # Word (.docx) — 双栏检测
│   ├── pdf_processor.py         # PDF OCR
│   ├── excel_processor.py       # Excel (.xlsx/.xls)
│   └── image_processor.py       # 图片 OCR
├── chunker/
│   └── semantic_chunker.py      # 语义分块算法
├── embedder/
│   └── embedder.py              # 双模式：TF-IDF离线 + Sentence-Transformer
├── storage/
│   ├── file_storage.py          # CSV/Excel 存储
│   └── vector_db_interface.py   # 向量数据库接口（预留）
└── utils/
    └── helpers.py               # 工具函数
```

## 配置文件说明

| 文件 | 用途 | 修改频率 |
|------|------|----------|
| **`settings.py`** | 部署配置：路径、API密钥、数据库连接、模型选择 | 部署时修改 |
| **`config.py`** | 算法参数：分块阈值、chunk大小、重叠比例 | 调优时修改 |

### settings.py 主要内容

```python
# 路径
KNOWLEDGE_BASE_DIR = "./knowledge_base"   # 文档存放目录
OUTPUT_DIR = "./output"                   # 向量输出目录
OUTPUT_FORMAT = "csv"                     # csv 或 excel

# 嵌入模式
USE_TFIDF = True                          # True=离线TF-IDF, False=在线模型
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TFIDF_MAX_FEATURES = 512                  # TF-IDF 向量维度

# OCR
OCR_LANG = "chi_sim+eng"                  # Tesseract 语言
PDF_DPI = 300

# 数据库连接（预留）
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
VECTOR_DB_TYPE = "file"                   # "file" | "milvus" | "chroma"

# 大模型 API（预留）
LLM_API_KEY = "your-api-key-here"
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4o"
```

## 已实现功能

### 文档解析
- **Word (.docx)** — 段落/表格/页眉页脚提取，**双栏/多栏布局检测**
- **PDF (.pdf)** — OCR 文字识别（基于 Tesseract），支持中英文混合
- **Excel (.xlsx/.xls)** — 多 Sheet 遍历，保留行列结构信息
- **图片 (.png/.jpg/.bmp/.tiff)** — OCR 文字识别，含图片预处理增强

### 语义分块
- 基于句子嵌入的 **语义相似度自适应分块**，在知识边界处切分
- 自然段落/句子边界优先，保持知识点完整性
- 相邻 chunk 文本重叠，避免边界信息丢失
- 参数集中在 `config.py` 中调优

### 向量化
- **双模式嵌入**：TF-IDF 离线模式（默认，无需联网） / Sentence-Transformer 在线模式
- 通过 `settings.py` 中 `USE_TFIDF` 一键切换
- 支持中英文混合文本，批量编码 + L2归一化

### 存储
- **CSV** 输出（默认）或 **Excel** 输出
- 含 chunk_id、原文、embedding JSON、元数据
- 支持多文件批量处理后自动合并
- 预留 **向量数据库接口** (`VectorDBInterface`)，后续可接入 Milvus/Chroma 等

### 扩展性
- 新增文档类型：继承 `BaseProcessor` → 实现 `can_handle()` + `extract_text()` → 在 `processors/__init__.py` 注册
- 新增向量数据库：实现 `VectorDBInterface` 抽象类 → 修改 `settings.py` 中 `VECTOR_DB_TYPE`
- 检索模块预留位置，可在 `storage/` 下扩展

## 环境准备

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 Tesseract OCR（PDF/图片处理必需）

- **Windows**: 下载安装 [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract tesseract-lang`
- **Linux**: `sudo apt install tesseract-ocr tesseract-ocr-chi-sim`

确保安装中文语言包（`chi_sim`）。

### 3. 安装 Poppler（PDF 处理必需）

- **Windows**: 下载 [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)，将 `bin/` 目录添加到 PATH
- **macOS**: `brew install poppler`
- **Linux**: `sudo apt install poppler-utils`

### 4. 安装 PyTorch（可选，GPU 加速）

```bash
# CUDA 版本
pip install torch --index-url https://download.pytorch.org/whl/cu118
# 或 CPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 使用方式

### 第一步：放入文档

将需要处理的文档放入 `knowledge_base/` 文件夹：

```
knowledge_base/
├── 产品手册.docx
├── 技术报告.pdf
├── 数据表格.xlsx
└── 截图.png
```

### 第二步：修改配置（可选）

编辑 `settings.py`，按需修改：
- **嵌入模式**: `USE_TFIDF = True` (离线) 或 `False` (在线模型)
- **输出路径**: `OUTPUT_DIR = "./output"`
- **输出格式**: `OUTPUT_FORMAT = "csv"` 或 `"excel"`
- **OCR 语言**: `OCR_LANG = "chi_sim+eng"`
- **数据库连接**: 接入向量数据库时填写

### 第三步：运行

```bash
# 默认模式：自动扫描 knowledge_base/ 文件夹
python main.py

# 查看当前配置
python main.py --show-settings

# 指定其他输入目录
python main.py --input ./my_docs/

# 指定输出目录和格式
python main.py --output ./my_vectors/ --format excel

# 调整分块参数
python main.py --similarity 0.7 --max-chars 800

# 查看支持的文件格式
python main.py --list-formats
```

### 完整参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `settings.KNOWLEDGE_BASE_DIR` | 输入目录路径 |
| `--output` | `-o` | `settings.OUTPUT_DIR` | 输出目录 |
| `--format` | `-f` | `settings.OUTPUT_FORMAT` | 输出格式: `csv` 或 `excel` |
| `--similarity` | `-s` | `0.6` | 语义分块相似度阈值 (0~1) |
| `--max-chars` | | `1000` | 单个 chunk 最大字符数 |
| `--min-chars` | | `80` | 单个 chunk 最小字符数 |
| `--overlap` | | `0.15` | chunk 间重叠比例 (0~0.3) |
| `--show-settings` | | | 查看当前 settings.py 配置 |
| `--list-formats` | | | 列出所有支持的文件格式 |
| `--verbose` | `-v` | | 详细日志 |

## 输出文件格式

生成的 CSV 文件包含以下列：

| 列名 | 说明 |
|------|------|
| `chunk_id` | 唯一 chunk ID（源文件名_序号） |
| `source_file` | 源文件名 |
| `chunk_text` | chunk 文本内容 |
| `embedding_json` | 向量 JSON 数组（384维） |
| `char_count` | chunk 字符数 |
| `metadata_json` | 元数据 JSON（来源、页数等） |
| `created_at` | 处理时间 |

## 扩展指南

### 新增文档类型

1. 在 `processors/` 下创建新文件，继承 `BaseProcessor`
2. 实现 `can_handle()` — 基于扩展名判断
3. 实现 `extract_text()` — 提取纯文本
4. 在 `processors/__init__.py` 的 `get_all_processors()` 中注册

```python
# processors/my_processor.py
from .base_processor import BaseProcessor

class MyProcessor(BaseProcessor):
    processor_name = "my_format"

    def can_handle(self, file_path: str) -> bool:
        return file_path.endswith(".myext")

    def extract_text(self, file_path: str) -> str:
        # 自定义提取逻辑
        return extracted_text
```

### 接入向量数据库

实现 `storage/vector_db_interface.py` 中的 `VectorDBInterface` 抽象类，然后在 `main.py` 中切换存储后端即可。

## 项目状态

- [x] Word 文档解析（含双栏检测）
- [x] PDF OCR 识别
- [x] Excel 表格解析
- [x] 图片 OCR 识别
- [x] 语义分块
- [x] 文本向量化
- [x] CSV/Excel 存储
- [x] 向量数据库接口占位
- [ ] 检索模块（后续开发）
- [ ] 向量数据库实际接入（后续开发）

## License

MIT
