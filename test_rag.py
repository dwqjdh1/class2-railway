import os
import dashscope
import numpy as np
from pathlib import Path

# 设置API Key
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    api_key = "sk-dc4d54e3c1ba4558be3e4fabcea7669e"  # 备用API Key
dashscope.api_key = api_key

# 知识库文件路径
KNOWLEDGE_FILE = Path(__file__).parent / "customer-service-bot" / "knowledge.txt"

# 加载知识库
def load_knowledge():
    """加载知识库并切分成片段"""
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    chunks = []
    # 按照标题和内容切分
    lines = text.strip().split('\n')
    current_chunk = []
    for line in lines:
        line = line.strip()
        if line.startswith('【') and line.endswith('】'):
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
        else:
            if line:
                current_chunk.append(line)
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    print(f"加载的知识库片段: {chunks}")
    return chunks

# 获取文本的嵌入向量
def get_embeddings(texts):
    """获取文本的嵌入向量"""
    # 调用Embedding API
    response = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=texts,
    )
    
    if response.status_code == 200:
        return [item['embedding'] for item in response.output['embeddings']]
    else:
        print(f"调用失败: {response.status_code}, {response.message}")
        return []

# 计算两个向量的余弦相似度
def cosine_similarity(vector_a, vector_b):
    """计算两个向量的余弦相似度"""
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

# 检索最相关的知识片段
def search_knowledge(query, chunks, chunk_vectors, top_k=3):
    """检索最相关的知识片段"""
    # 获取查询向量
    query_vector = get_embeddings([query])[0]
    
    # 计算相似度并排序
    scored_chunks = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((float(score), chunk))
    
    scored_chunks.sort(reverse=True, key=lambda item: item[0])
    print(f"查询: {query}")
    print(f"搜索结果: {scored_chunks}")
    return scored_chunks[:top_k]

# 使用大模型基于检索结果回答问题
def answer_with_rag(query, search_results):
    """使用大模型基于检索结果回答问题"""
    # 构建上下文
    context_parts = []
    for index, (score, chunk) in enumerate(search_results, start=1):
        context_parts.append(f"[资料{index}]\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    # 调用大模型
    system_prompt = """
你是企业智能客服，专门回答用户关于星河科技有限公司产品和服务的问题。
请严格按照以下要求回答：
1. 只根据提供的企业知识库资料回答问题
2. 回答要直接、准确，不要添加任何引言或开场白
3. 如果资料中没有相关信息，明确说明"知识库中暂未找到相关信息，需要人工客服进一步处理"
4. 使用简洁专业的语言
"""
    
    user_prompt = f"""
企业知识库资料：
{context}

用户问题：
{query}

请根据上述资料回答用户问题，直接给出答案，不要有任何引言。
"""
    
    response = dashscope.Generation.call(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        top_p=0.9,
    )
    
    if response.status_code == 200:
        return response.output.text
    else:
        return f"调用失败: {response.status_code}, {response.message}"

# 测试RAG功能
def test_rag():
    # 加载知识库
    chunks = load_knowledge()
    print(f"知识库片段数量: {len(chunks)}")
    
    # 生成知识库向量
    chunk_vectors = get_embeddings(chunks)
    if not chunk_vectors:
        print("生成向量失败")
        return
    print(f"向量维度: {len(chunk_vectors[0])}")
    
    # 测试问题
    test_questions = [
        "专业版多少钱？",
        "如何申请发票？",
        "超过 7 天还能退款吗？",
        "人工客服什么时候在线？",
    ]
    
    for question in test_questions:
        print(f"\n用户问题: {question}")
        
        # 检索相关知识
        search_results = search_knowledge(question, chunks, chunk_vectors)
        print("检索到的相关资料:")
        for score, chunk in search_results:
            print(f"相似度: {score:.4f}\n{chunk}\n")
        
        # 生成回答
        answer = answer_with_rag(question, search_results)
        print(f"回答: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    test_rag()
