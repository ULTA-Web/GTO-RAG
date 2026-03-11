import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOARD_DATA_PATH = os.path.join(BASE_DIR, "rag_metadata.json")

def build_board_rag_database():
    """
    构建基于board计算的例子检索数据库
    """
    print(f"尝试加载本地数据文件: {BOARD_DATA_PATH}...")
    
    # 1. 检查文件是否存在
    if not os.path.exists(BOARD_DATA_PATH):
        print(f"[错误] 找不到文件：{BOARD_DATA_PATH}。请检查路径是否正确！")
        return False
    
    # 2. 加载本地 JSON 数据
    try:
        with open(BOARD_DATA_PATH, 'r', encoding='utf-8') as f:
            rag_database = json.load(f)
        print(f"成功加载数据库！共包含 {len(rag_database)} 条历史策略数据。")
        return True
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 格式解析失败，请检查文件内容是否合规：{e}")
        return False

if __name__ == "__main__":
    build_board_rag_database()