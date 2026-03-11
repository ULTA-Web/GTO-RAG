#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GTO-RAG 整合版主入口文件

本文件整合了三个功能：
1. 文本RAG检索
2. 基于board计算的例子检索
3. 处理翻译情况

每个功能都有两个函数：
- 建库函数：用于构建或加载数据库
- 查询函数：用于查询检索结果
"""

from src.rag.ingest import build_text_rag_database
from src.rag.retrieve import query_text_rag
from src.board.ingest import build_board_rag_database
from src.board.retrieve import query_board_rag
from src.translation.ingest import build_translation_database
from src.translation.retrieve import query_translation


def main():
    """
    主函数，展示各个功能的使用方法
    """
    print("=== GTO-RAG 整合版 ===")
    print("1. 文本RAG检索")
    print("2. 基于board计算的例子检索")
    print("3. 处理翻译情况")
    print("4. 退出")
    
    while True:
        choice = input("请选择功能编号: ")
        
        if choice == "1":
            print("\n=== 文本RAG检索 ===")
            sub_choice = input("1. 构建数据库\n2. 查询检索\n请选择: ")
            if sub_choice == "1":
                build_text_rag_database()
            elif sub_choice == "2":
                query = input("请输入查询内容: ")
                results = query_text_rag(query, k=3)
                print(f"\n检索到 {len(results)} 个结果:")
                for i, (doc, score) in enumerate(results):
                    print(f"\n结果 {i+1} (相似度: {1-score:.4f}):")
                    print(f"内容: {doc.page_content}")
                    print(f"来源: {doc.metadata.get('source', '未知')}")
            else:
                print("无效选择")
        
        elif choice == "2":
            print("\n=== 基于board计算的例子检索 ===")
            sub_choice = input("1. 加载数据库\n2. 查询检索\n请选择: ")
            if sub_choice == "1":
                build_board_rag_database()
            elif sub_choice == "2":
                # 示例查询
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
                print(f"\n检索到 {len(results)} 个结果:")
                for i, res in enumerate(results):
                    print(f"\n结果 {i+1} (距离: {res['distance']:.4f}):")
                    print(f"牌面: {res['board']}")
                    print(f"策略: {res['data']}")
            else:
                print("无效选择")
        
        elif choice == "3":
            print("\n=== 处理翻译情况 ===")
            sub_choice = input("1. 构建术语数据库\n2. 查询翻译\n请选择: ")
            if sub_choice == "1":
                build_translation_database()
            elif sub_choice == "2":
                english_text = input("请输入英文文本: ")
                prompt = query_translation(english_text)
                print("\n生成的翻译提示:")
                print(prompt)
            else:
                print("无效选择")
        
        elif choice == "4":
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")
        
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()