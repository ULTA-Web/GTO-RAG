import re
from collections import defaultdict
from src.translation.ingest import POKER_TERMINOLOGY, STOP_WORDS

def preprocess_text(text):
    # 1. 转为小写
    text = text.lower()
    # 2. 移除多余标点
    text = re.sub(r'[^\w\s()/\-]', '', text)
    # 3. 按空格分词
    tokens = text.split()
    # 4. 去停用词 + 去空字符串
    tokens = [token for token in tokens if token not in STOP_WORDS and token.strip()]
    return tokens

def generate_ngrams(tokens, n):
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

def match_terminology(text):
    """
    核心匹配逻辑
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
    
    # 匹配单个单词
    for idx, token in enumerate(tokens):
        if idx in matched_indices:
            continue
        lemmatized_token = lemmatize_word(token)
        if lemmatized_token in POKER_TERMINOLOGY:
            hit_terms[lemmatized_token] = POKER_TERMINOLOGY[lemmatized_token]
        elif token in POKER_TERMINOLOGY:
            hit_terms[token] = POKER_TERMINOLOGY[token]
    
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