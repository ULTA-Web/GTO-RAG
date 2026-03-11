import json
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOARD_DATA_PATH = os.path.join(BASE_DIR, "rag_metadata.json")

def get_rank_value(rank_char):
    """将牌面字符转换为数值（A=14, K=13...2=2）"""
    rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, 
                '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    return rank_map.get(rank_char.upper(), 0)

def encode_board_to_vector_13d(board_str, texture_dict):
    """
    将牌面转换为13维特征向量：
    [card0, card1, card2,  # 牌值（降序）
     suit0, suit1, suit2,  # 花色One-Hot：彩虹/双色/单色
     straight0, straight1, straight2, straight3,  # 顺子One-Hot：无/卡顺/两头顺/成顺
     pair0, pair1, pair2]  # 成对One-Hot：无对/单对/强成牌
    """
    # 1. 牌值维度（card0-2）：降序排列的三张牌数值
    ranks = sorted([get_rank_value(board_str[i]) for i in range(0, 6, 2)], reverse=True)
    # 兜底：防止board_str格式错误（不足3张牌）
    card0 = ranks[0] if len(ranks)>=1 else 0
    card1 = ranks[1] if len(ranks)>=2 else 0
    card2 = ranks[2] if len(ranks)>=3 else 0

    # 2. 花色结构One-Hot（suit0-2）
    suit0 = suit1 = suit2 = 0
    suit_texture = texture_dict.get("suit_texture", "")
    if "Rainbow" in suit_texture:
        suit0 = 1
    elif "Two-tone" in suit_texture:
        suit1 = 1
    elif "Monotone" in suit_texture:
        suit2 = 1

    # 3. 顺子结构One-Hot（straight0-3）
    straight0 = straight1 = straight2 = straight3 = 0
    straight_texture = texture_dict.get("straight_texture", "")
    if "No straight draws and no made straight" in straight_texture:
        straight0 = 1
    elif "Gutshot" in straight_texture or "Inside straight draw" in straight_texture:
        straight1 = 1
    elif "OESD" in straight_texture or "Open-ended straight draw" in straight_texture:
        straight2 = 1
    elif "Made straight" in straight_texture:
        straight3 = 1

    # 4. 成对结构One-Hot（pair0-2）
    pair0 = pair1 = pair2 = 0
    pairing_texture = texture_dict.get("pairing_texture", "")
    if "Unpaired board" in pairing_texture:
        pair0 = 1
    elif "One pair board" in pairing_texture:
        pair1 = 1
    elif any(key in pairing_texture for key in ["Two pair", "Trips", "Set", "Full house", "Quads"]):
        pair2 = 1

    # 组合13维向量
    return [
        card0, card1, card2,
        suit0, suit1, suit2,
        straight0, straight1, straight2, straight3,
        pair0, pair1, pair2
    ]

def calculate_poker_distance_13d(v1, v2):
    """
    计算13维特征向量的加权欧氏距离（使用自定义权重）
    权重：[3.0/12, 1.5/12, 0.5/12, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3, 2.0/3]
    """
    # 预计算权重为浮点数，避免重复计算
    weights = [
        3.0/12,    # card0: 0.25
        1.5/12,    # card1: 0.125
        0.5/12,    # card2: ≈0.0417
        2.0/3,     # suit0: ≈0.6667
        2.0/3,     # suit1: ≈0.6667
        2.0/3,     # suit2: ≈0.6667
        2.0/3,     # straight0: ≈0.6667
        2.0/3,     # straight1: ≈0.6667
        2.0/3,     # straight2: ≈0.6667
        2.0/3,     # straight3: ≈0.6667
        2.0/3,     # pair0: ≈0.6667
        2.0/3,     # pair1: ≈0.6667
        2.0/3      # pair2: ≈0.6667
    ]
    
    squared_dist = 0.0
    for i in range(13):
        diff = v1[i] - v2[i]
        squared_dist += weights[i] * (diff ** 2)
        
    return math.sqrt(squared_dist)

def query_board_rag(query, k=3):
    """
    从数据库中检索最相似的 Top-K 牌面策略（使用13维向量+自定义权重）
    """
    # 1. 加载本地 JSON 数据
    try:
        with open(BOARD_DATA_PATH, 'r', encoding='utf-8') as f:
            rag_database = json.load(f)
    except FileNotFoundError:
        print(f"[错误] 找不到文件：{BOARD_DATA_PATH}。请检查路径是否正确！")
        return []
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 格式解析失败，请检查文件内容是否合规：{e}")
        return []
    
    query_meta = query.get("meta_data", {})
    query_texture = query_meta.get("texture", {})
    
    # 第一步：计算Query的13维特征向量
    query_vector = encode_board_to_vector_13d(query_meta.get("board", ""), query_texture)
    
    candidates = []
    
    # 第二步：遍历数据库进行匹配
    for item in rag_database:
        item_meta = item.get("meta_data", {})
        item_texture = item_meta.get("texture", {})
        
        # --- 硬过滤（Hard Filters）阶段 ---
        if item_meta.get("street") != query_meta.get("street"):
            continue
        if item_meta.get("position") != query_meta.get("position"):
            continue
        if item_meta.get("line") != query_meta.get("line"):
            continue
        
        # --- 软检索（Soft Match）阶段 ---
        item_vector = encode_board_to_vector_13d(item_meta.get("board", ""), item_texture)
        distance = calculate_poker_distance_13d(query_vector, item_vector)
        
        candidates.append({
            "distance": distance,
            "board": item_meta.get("board"),
            "data": item.get("data"),
            "full_meta": item_meta
        })
    
    # 第三步：按距离从小到大排序（距离越小越相似）
    candidates.sort(key=lambda x: x["distance"])
    
    # 返回 Top K
    return candidates[:k]