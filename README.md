# GTO-RAG: Poker Strategy Retrieval System

## 项目简介 (Introduction)

GTO-RAG 是一个专注于扑克 GTO (Game Theory Optimal) 策略的本地化检索增强 (Retrieval-Augmented) 系统。该项目旨在帮助用户通过上传扑克策略相关的 PDF 文档，构建一个私有的知识库，并通过语义搜索快速找到相关的策略建议。

目前的 **主程序 (`main.py`)** 运行在 **"Local / No-LLM" (纯本地/无大模型)** 模式下，作为一个强大的语义搜索引擎，直接返回文档中最相关的片段，无需依赖外部 API，确保了数据的隐私和检索的高效性。

此外，项目中还包含了一个可选的 RAG 实现 (`src/rag.py`)，如果配置了 OpenAI API Key，可以启用基于 GPT-4o 的智能问答功能。

## 主要特性 (Features)

*   **本地化 Embeddings**: 使用本地 HuggingFace 模型进行向量化，无需 API 调用成本。
*   **高效向量检索**: 基于 ChromaDB 构建向量数据库，支持快速相似度搜索。
*   **自动文档处理**: 支持批量导入 PDF 文档，自动进行文本分割和索引。
*   **交互式 CLI**: 提供简洁的命令行界面进行交互式搜索。

## 项目结构 (Project Structure)

*   `main.py`: 程序的入口点，提供交互式搜索命令行界面。
*   `src/ingest.py`: 数据处理脚本，负责加载 `data/` 目录下的 PDF，生成向量并存入 `db/`。
*   `src/retrieve.py`: 检索逻辑的核心实现，负责加载向量库并执行相似度搜索。
*   `src/rag.py`: (可选) 基于 OpenAI 的 RAG 实现，可进行生成式问答。
*   `db/`: 存放生成的 ChromaDB 向量数据库文件。
*   `data/`: (需自行创建) 存放 PDF 源文件的目录。
*   `requirements.txt`: 项目依赖列表。

## 环境要求 (Prerequisites)

*   Python 3.10 或更高版本
*   推荐使用虚拟环境 (Virtualenv / Conda)

## 安装指南 (Installation)

1.  **克隆项目**
    ```bash
    git clone <repository-url>
    cd GTO-RAG
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

## 配置说明 (Configuration)

### 1. 准备数据
在项目根目录下创建一个名为 `data` 的文件夹，并将你的扑克策略 PDF 文件放入其中。
```bash
mkdir data
# 将 .pdf 文件复制到 data/ 目录中
```

### 2. 模型路径配置 (重要)
目前代码中硬编码了本地 Embedding 模型的路径。在运行前，请务必检查并修改以下文件中的 `LOCAL_MODEL_PATH` 变量：

*   `src/ingest.py`
*   `src/retrieve.py`

默认配置为：
```python
LOCAL_MODEL_PATH = r"G:\new_projects\GTO-RAG\model"
```
如果你没有该路径或模型，请将其修改为你本地存放 Sentence-Transformer 模型文件的路径，或者修改代码以使用 HuggingFace Hub 在线加载模型（例如 `model_name="sentence-transformers/all-MiniLM-L6-v2"`）。

## 使用指南 (Usage)

### 第一步：构建知识库 (Ingestion)
运行以下命令处理 PDF 文件并生成向量数据库：
```bash
python src/ingest.py
```
*注意：首次运行可能需要下载模型（如果未配置本地路径），且处理大量 PDF 可能需要一些时间。*

### 第二步：执行搜索 (Search)
构建完成后，运行主程序开始搜索：
```bash
python main.py
```
在提示符后输入你的问题（例如："How to play flush draws?"），系统将返回相关性最高的文档片段。

---

## 进阶功能：启用 OpenAI RAG
如果你希望系统直接回答问题而不仅仅是检索片段，可以使用 `src/rag.py`。
1. 创建 `.env` 文件并设置 `OPENAI_API_KEY`。
2. 修改代码或直接调用 `src/rag.py` 中的逻辑来使用 GPT-4o 进行生成。
