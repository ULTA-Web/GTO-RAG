# GTO-RAG 整合版

本项目整合了三个核心功能：
1. 文本RAG检索
2. 基于board计算的例子检索
3. 处理翻译情况

每个功能都提供了两个函数：
- 建库函数：用于构建或加载数据库
- 查询函数：用于查询检索结果

## 目录结构

```
GTO-RAG-整合版/
├── data/             # 存放PDF文件
├── db/               # 向量数据库存储
├── model/            # 嵌入模型
├── src/              # 源代码
│   ├── rag/          # 文本RAG检索模块
│   │   ├── ingest.py    # 建库函数
│   │   └── retrieve.py  # 查询函数
│   ├── board/        # 基于board计算的例子检索模块
│   │   ├── ingest.py    # 建库函数
│   │   └── retrieve.py  # 查询函数
│   └── translation/  # 处理翻译情况模块
│       ├── ingest.py    # 建库函数
│       └── retrieve.py  # 查询函数
├── main.py           # 主入口文件
└── requirements.txt  # 依赖文件
```

## 功能说明

### 1. 文本RAG检索
- **建库函数**：`build_text_rag_database()`
  - 加载data目录下的PDF文件
  - 分割文本并生成嵌入向量
  - 构建向量数据库并存储到db目录

- **查询函数**：`query_text_rag(query, k=5)`
  - 从向量数据库中检索与查询最相似的文本片段
  - 返回top-k个结果

### 2. 基于board计算的例子检索
- **建库函数**：`build_board_rag_database()`
  - 加载rag_metadata.json文件中的历史策略数据

- **查询函数**：`query_board_rag(query, k=3)`
  - 基于13维特征向量计算牌面相似度
  - 应用硬过滤（位置、轮次、行动线）
  - 返回最相似的top-k个牌面策略

### 3. 处理翻译情况
- **建库函数**：`build_translation_database()`
  - 加载德扑专业术语翻译词典

- **查询函数**：`query_translation(english_text)`
  - 识别文本中的德扑专业术语（支持n-gram匹配）
  - 使用embedding模型计算相似度，补充匹配（相似度阈值：0.7）
  - 生成包含术语翻译标准的翻译提示

## 安装步骤

1. 克隆项目到本地
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 准备数据：
   - 将PDF文件放入data目录
   - 确保rag_metadata.json文件在项目根目录

## 使用方法

### 方法一：使用主入口文件

运行主入口文件：
```bash
python main.py
```

根据提示选择功能：
1. 文本RAG检索
2. 基于board计算的例子检索
3. 处理翻译情况
4. 退出

### 方法二：直接调用函数

```python
# 文本RAG检索
from src.rag.ingest import build_text_rag_database
from src.rag.retrieve import query_text_rag

# 构建数据库
build_text_rag_database()

# 查询检索
results = query_text_rag("如何在翻牌前玩口袋A", k=3)

# 基于board计算的例子检索
from src.board.ingest import build_board_rag_database
from src.board.retrieve import query_board_rag

# 加载数据库
build_board_rag_database()

# 查询检索
query = {
    "meta_data": {
        "line": "preflop=BTN_2.5bb_BB_Call&flop=BB_Check&turn=&river=",
        "board": "Qd8s3c",
        "position": "BTN vs BB",
        "street": "flop",
        "texture": {
            "pairing_texture": "Unpaired board",
            "high_card_texture": "Qhigh",
            "suit_texture": "Rainbow flop, no made flush, no flush draw",
            "straight_texture": "No straight draws and no made straight"
        }
    }
}
results = query_board_rag(query, k=3)

# 处理翻译情况
from src.translation.ingest import build_translation_database
from src.translation.retrieve import query_translation

# 构建术语数据库
build_translation_database()

# 查询翻译
prompt = query_translation("You should consider making these kind of bets on the river in re-raised pots when you suspect both of you have A K")
print(prompt)
```

## 依赖说明

- langchain：核心框架
- langchain-community：社区集成
- langchain-text-splitters：文本分割
- langchain-huggingface：HuggingFace集成
- langchain-chroma：Chroma向量数据库集成
- chromadb：向量数据库
- transformers：嵌入模型
- torch：深度学习框架
- pypdf：PDF处理
- numpy：数值计算
- pandas：数据处理

## 注意事项

1. **文本RAG检索的嵌入模型**：
   - 默认使用本地模型，存储在model目录
   - 推荐使用sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2模型
   - 下载方法：
     ```bash
     # 使用Hugging Face CLI下载
     huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir model
     ```
   - 首次运行建库函数时，会自动加载模型，可能需要几分钟时间

2. **基于board计算的例子检索**：
   - 需要rag_metadata.json文件，包含历史策略数据
   - 文件格式：JSON数组，每个元素包含meta_data和data字段

3. **处理翻译情况**：
   - 使用内置的德扑专业术语词典
   - 词典存储在src/translation/ingest.py文件中
   - 使用与文本RAG检索相同的embedding模型计算术语相似度
   - 支持n-gram匹配和基于相似度的补充匹配

4. **文件夹组织**：
   - data/：存放PDF文件，用于文本RAG检索
   - db/：存储向量数据库，由文本RAG检索自动生成
   - model/：存储嵌入模型，需要手动下载
   - src/：源代码目录，包含三个功能模块

5. **GitHub仓库包含内容**：
   - 包含所有源代码文件
   - 包含rag_metadata.json文件
   - 包含data、db和model目录结构
   - 模型文件较大，可能需要使用Git LFS存储

## 示例

### 文本RAG检索示例

输入："如何在翻牌前玩口袋A"
输出：
```
检索到 3 个结果:

结果 1 (相似度: 0.95):
内容: 口袋A是扑克中最强的起手牌，翻牌前应该加注...
来源: data/poker_strategy.pdf
```

### 基于board计算的例子检索示例

输入：
```python
query = {
    "meta_data": {
        "line": "preflop=BTN_2.5bb_BB_Call&flop=BB_Check&turn=&river=",
        "board": "Qd8s3c",
        "position": "BTN vs BB",
        "street": "flop",
        "texture": {
            "pairing_texture": "Unpaired board",
            "high_card_texture": "Qhigh",
            "suit_texture": "Rainbow flop, no made flush, no flush draw",
            "straight_texture": "No straight draws and no made straight"
        }
    }
}
```

输出：
```
检索到 3 个结果:

结果 1 (距离: 0.1234):
牌面: Qh8d3s
策略: 在这种彩虹面且无顺子听牌的情况下，BTN位置应该持续下注...
```

### 处理翻译情况示例

输入："You should consider making these kind of bets on the river in re-raised pots when you suspect both of you have A K"

输出：
```
请严格按照以下术语翻译标准，准确翻译德扑相关英文文本：

【德扑专业术语翻译标准】
river → 河牌圈
re-raised pots → 重注底池

需要翻译的文本：
You should consider making these kind of bets on the river in re-raised pots when you suspect both of you have A K
```

## 许可证

本项目采用MIT许可证。
