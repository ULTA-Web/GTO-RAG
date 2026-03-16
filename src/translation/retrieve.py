import re
import os
import numpy as np
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
# 注意：请确保 POKER_TERMINOLOGY、STOP_WORDS 能正常导入
from src.translation.ingest import POKER_TERMINOLOGY, STOP_WORDS

# 模型配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "model")

# 简单检查路径
if os.path.exists(LOCAL_MODEL_PATH):
    EMBEDDING_MODEL_PATH = LOCAL_MODEL_PATH
else:
    EMBEDDING_MODEL_PATH = LOCAL_MODEL_PATH

# 加载嵌入模型（设置 normalize_embeddings=True 简化余弦相似度计算）
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={"device": "cpu"},  # 可根据需要改为 "cuda"
    encode_kwargs={"normalize_embeddings": True}  # 归一化向量，余弦相似度=点积
)

def preprocess_text(text):
    """文本预处理：小写、去标点、分词、去停用词"""
    # 1. 转为小写
    text = text.lower()
    # 2. 移除多余标点（保留括号、斜杠、连字符）
    text = re.sub(r'[^\w\s()/\-]', '', text)
    # 3. 按空格分词
    tokens = text.split()
    # 4. 去停用词 + 去空字符串
    tokens = [token for token in tokens if token not in STOP_WORDS and token.strip()]
    return tokens

def generate_ngrams(tokens, n):
    """生成 n-gram 短语，包含起止索引"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i+n])
        ngrams.append({
            "phrase": ngram,
            "start_idx": i,
            "end_idx": i + n - 1
        })
    return ngrams

def lemmatize_word(word):
    """德扑术语词形还原（处理时态/单复数变体）"""
    # 德扑术语常见变体映射
    variant_map = {
        "bets": "bet",
        "betting": "bet",
        "betted": "bet",
        "raises": "raise",
        "raising": "raise",
        "raised": "raise",
        "calls": "call",
        "calling": "call",
        "called": "call",
        "folds": "fold",
        "folding": "fold",
        "folded": "fold",
        "draws": "draw",
        "drawing": "draw",
        "flushes": "flush",
        "flushing": "flush",
        "straights": "straight",
        "straightening": "straight",
        "bluffs": "bluff",
        "bluffing": "bluff",
        "ranges": "range",
        "ranging": "range",
        "stacks": "stack",
        "stacking": "stack",
        "pots": "pot",
        "potting": "pot",
        "flops": "flop",
        "flopping": "flop",
        "turns": "turn",
        "turning": "turn",
        "rivers": "river",
        "rivering": "river",
        "semi-bluffs": "semi-bluff",
        "blockers": "blocker",
        "outs": "outs",  # 单复数同形
        "out": "outs"
    }
    # 优先匹配变体映射，无匹配则返回原词
    return variant_map.get(word.lower(), word.lower())

def calculate_similarity_matrix(terms_list, tokens_list):
    """
    矩阵化计算术语列表与文本单词列表的相似度
    :param terms_list: 德扑术语列表
    :param tokens_list: 文本预处理后的单词列表
    :return: 相似度矩阵 (n_terms, n_tokens)，值为余弦相似度
    """
    if not terms_list or not tokens_list:
        return np.array([])
    

    terms_embeddings = embedding_model.embed_documents(terms_list)  # (n_terms, dim)
    tokens_embeddings = embedding_model.embed_documents(tokens_list)  # (n_tokens, dim)
    
    terms_matrix = np.array(terms_embeddings)  # (n_terms, dim)
    tokens_matrix = np.array(tokens_embeddings)  # (n_tokens, dim)
    
    similarity_matrix = np.dot(terms_matrix, tokens_matrix.T)  # (n_terms, n_tokens)
    
    return similarity_matrix

def match_terminology(text):
    """
    核心匹配逻辑：n-gram精准匹配 + 单词级embedding语义匹配
    """
    tokens = preprocess_text(text)
    if not tokens:
        return {}
    
    matched_indices = set()
    hit_terms = defaultdict(str)
    
    # 匹配3gram
    trigrams = generate_ngrams(tokens, 3)
    for trigram in trigrams:
        phrase = trigram["phrase"]
        if phrase in POKER_TERMINOLOGY:
            hit_terms[phrase] = POKER_TERMINOLOGY[phrase]
            for idx in range(trigram["start_idx"], trigram["end_idx"] + 1):
                matched_indices.add(idx)
    
    # 匹配2gram
    bigrams = generate_ngrams(tokens, 2)
    for bigram in bigrams:
        if bigram["start_idx"] in matched_indices or bigram["end_idx"] in matched_indices:
            continue
        phrase = bigram["phrase"]
        if phrase in POKER_TERMINOLOGY:
            hit_terms[phrase] = POKER_TERMINOLOGY[phrase]
            for idx in range(bigram["start_idx"], bigram["end_idx"] + 1):
                matched_indices.add(idx)
    
    # 匹配单个单词（词形还原后）
    for idx, token in enumerate(tokens):
        if idx in matched_indices:
            continue
        lemmatized_token = lemmatize_word(token)
        if lemmatized_token in POKER_TERMINOLOGY:
            hit_terms[lemmatized_token] = POKER_TERMINOLOGY[lemmatized_token]
            matched_indices.add(idx)
        elif token in POKER_TERMINOLOGY:
            hit_terms[token] = POKER_TERMINOLOGY[token]
            matched_indices.add(idx)
    

    similarity_threshold = 0.7  # 相似度阈值，可调整
    unmatched_terms = [term for term in POKER_TERMINOLOGY if term not in hit_terms]
    unmatched_tokens = [token for idx, token in enumerate(tokens) if idx not in matched_indices]
    
    if unmatched_terms and unmatched_tokens:
     
        similarity_matrix = calculate_similarity_matrix(unmatched_terms, unmatched_tokens)

  
        for term_idx, term in enumerate(unmatched_terms):
         
            term_similarities = similarity_matrix[term_idx]
            matched_token_indices = np.where(term_similarities >= similarity_threshold)[0]
            if len(matched_token_indices) > 0:
                hit_terms[term] = POKER_TERMINOLOGY[term]
    
    return dict(hit_terms)

def query_translation(english_text):
    """
    处理翻译情况的查询函数
    :param english_text: 英文文本
    :return: 翻译提示
    """
    hit_terms = match_terminology(english_text)
    
    if not hit_terms:
        prompt = f"""请准确翻译以下德扑相关英文文本，使用德扑专业术语：
{english_text}
"""
    else:
        term_pairs = "\n".join([f"{en} → {zh}" for en, zh in hit_terms.items()])
        prompt = f"""请严格按照以下术语翻译标准，准确翻译德扑相关英文文本：

【德扑专业术语翻译标准】
{term_pairs}

需要翻译的文本：
{english_text}
"""
    return prompt
