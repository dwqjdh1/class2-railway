import os
import json
import numpy as np
from docx import Document
from openai import OpenAI

# 模型配置
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"

# API Key 配置
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    api_key = "sk-dc4d54e3c1ba4558be3e4fabcea7669e"

client = OpenAI(
    api_key=api_key,
    base_url=QWEN_BASE_URL,
)


def get_embedding(text, model=EMBEDDING_MODEL):
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def parse_docx(file_path):
    """解析Word文档，提取段落"""
    doc = Document(file_path)
    paragraphs = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    
    return paragraphs


def build_knowledge_base(docx_path, output_json="knowledge_base.json"):
    """构建知识库"""
    print("解析文档中...")
    paragraphs = parse_docx(docx_path)
    print(f"提取到 {len(paragraphs)} 个段落")
    
    knowledge_base = []
    
    print("生成嵌入向量中...")
    for i, text in enumerate(paragraphs):
        if i % 5 == 0:
            print(f"处理中: {i+1}/{len(paragraphs)}")
        
        try:
            embedding = get_embedding(text)
            knowledge_base.append({
                "id": i,
                "text": text,
                "embedding": embedding
            })
        except Exception as e:
            print(f"处理段落 {i+1} 时出错: {e}")
    
    # 保存知识库
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    
    print(f"知识库构建完成，保存到 {output_json}")
    print(f"知识库包含 {len(knowledge_base)} 个条目")


def search_knowledge_base(query, knowledge_base_path="knowledge_base.json", top_k=3):
    """检索知识库"""
    # 加载知识库
    with open(knowledge_base_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
    
    # 获取查询向量
    query_embedding = get_embedding(query)
    
    # 计算相似度
    similarities = []
    for item in knowledge_base:
        sim = cosine_similarity(query_embedding, item["embedding"])
        # 只存储相似度和文本，避免字典比较问题
        similarities.append((sim, item["text"], item["id"]))
    
    # 排序并返回前k个
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_items = similarities[:top_k]
    
    return top_items


if __name__ == "__main__":
    # 构建知识库
    docx_path = r"c:\Users\dwqjdh\Desktop\NLP\自然语言处理实践\家护家电产品说明书.docx"
    # build_knowledge_base(docx_path)  # 已经构建过了，注释掉避免重复
    
    # 测试检索
    test_query = "产品如何使用"
    results = search_knowledge_base(test_query)
    
    print("\n测试检索结果:")
    for i, (sim, text, _) in enumerate(results):
        print(f"\n第 {i+1} 条 (相似度: {sim:.4f}):")
        print(text)
