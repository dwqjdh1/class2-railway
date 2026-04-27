import os
import dashscope
import numpy as np
from pathlib import Path
from openai import OpenAI
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型配置
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-plus"

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"

# API Key 配置
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    api_key = "sk-dc4d54e3c1ba4558be3e4fabcea7669e"

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    deepseek_api_key = "sk-5e5da217bc7247119fff4c5cc7ddd8cd"

# 客户端配置
qwen_client = OpenAI(
    api_key=api_key,
    base_url=QWEN_BASE_URL,
)

deepseek_client = OpenAI(
    api_key=deepseek_api_key,
    base_url=DEEPSEEK_BASE_URL,
)

# RAG相关功能
KNOWLEDGE_FILE = Path("customer-service-bot/knowledge.txt")

# 全局变量，用于存储知识库和向量
chunks = []
chunk_vectors = []

# 加载知识库
def load_knowledge():
    """加载知识库并切分成片段"""
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    chunks = []
    for block in text.split("\n\n"):
        chunk = block.strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# 获取文本的嵌入向量
def get_embeddings(texts):
    """获取文本的嵌入向量"""
    # 设置API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = "sk-dc4d54e3c1ba4558be3e4fabcea7669e"  # 备用API Key
    
    dashscope.api_key = api_key
    
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
def search_knowledge(query, top_k=3):
    """检索最相关的知识片段"""
    # 获取查询向量
    query_vector = get_embeddings([query])[0]
    
    # 计算相似度并排序
    scored_chunks = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((float(score), chunk))
    
    scored_chunks.sort(reverse=True, key=lambda item: item[0])
    return scored_chunks[:top_k]

# 使用大模型基于检索结果回答问题
def answer_with_rag(query, search_results):
    """使用大模型基于检索结果回答问题"""
    # 构建上下文
    context_parts = []
    for index, (score, chunk) in enumerate(search_results, start=1):
        context_parts.append(f"[资料{index}，相似度 {score:.4f}]\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    # 设置API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = "sk-dc4d54e3c1ba4558be3e4fabcea7669e"  # 备用API Key
    
    dashscope.api_key = api_key
    
    # 调用大模型
    system_prompt = """
你是企业智能客服。
你必须根据企业知识库资料回答用户问题。
如果资料中没有答案，请说明"知识库中暂未找到相关信息，需要人工客服进一步处理"。
回答要礼貌、简洁、专业，不要编造资料中没有的信息。
"""
    
    user_prompt = f"""
请根据下面的企业知识库资料回答用户问题。

企业知识库资料：
{context}

用户问题：
{query}
"""
    
    response = dashscope.Generation.call(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    
    if response.status_code == 200:
        return response.output.text
    else:
        return f"调用失败: {response.status_code}, {response.message}"

# 初始化知识库
if KNOWLEDGE_FILE.exists():
    chunks = load_knowledge()
    print(f"知识库片段数量: {len(chunks)}")
    
    # 生成知识库向量
    chunk_vectors = get_embeddings(chunks)
    if chunk_vectors:
        print(f"向量维度: {len(chunk_vectors[0])}")
    else:
        print("生成向量失败")
else:
    print(f"知识库文件不存在: {KNOWLEDGE_FILE}")


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    model = data.get("model", "qwen")
    messages = data.get("messages", [])

    if not user_message:
        return {"error": "消息不能为空"}

    try:
        # 构建对话历史
        chat_messages = []
        for msg in messages:
            chat_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 添加系统提示
        if model == "qwen":
            chat_messages.insert(0, {
                "role": "system",
                "content": "你是自然语言处理课程助教，回答要准确、简洁。",
            })
            client = qwen_client
            model_name = QWEN_CHAT_MODEL
        else:
            chat_messages.insert(0, {
                "role": "system",
                "content": "你是自然语言处理课程助教，回答要准确、简洁。",
            })
            client = deepseek_client
            model_name = DEEPSEEK_CHAT_MODEL
        
        # 添加当前用户消息
        chat_messages.append({
            "role": "user",
            "content": user_message,
        })

        completion = client.chat.completions.create(
            model=model_name,
            messages=chat_messages,
            temperature=0.3,
        )
        answer = completion.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.post("/rag")
async def rag(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    if not user_message:
        return {"error": "消息不能为空"}

    try:
        # 检索相关知识
        search_results = search_knowledge(user_message)
        
        # 生成回答
        answer = answer_with_rag(user_message, search_results)
        
        # 准备返回结果
        result = {
            "answer": answer,
            "search_results": [
                {"score": score, "content": chunk} for score, chunk in search_results
            ]
        }
        return result
    except Exception as e:
        return {"error": str(e)}

