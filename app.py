import os
import json
import numpy as np
from docx import Document
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
QWEN_EMBEDDING_MODEL = "text-embedding-v4"

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

# 知识库检索功能
def get_embedding(text, model=QWEN_EMBEDDING_MODEL):
    response = qwen_client.embeddings.create(
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

def search_knowledge_base(query, knowledge_base_path="knowledge_base.json", top_k=3):
    """检索知识库"""
    try:
        # 加载知识库
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        
        # 获取查询向量
        query_embedding = get_embedding(query)
        
        # 计算相似度
        similarities = []
        for item in knowledge_base:
            sim = cosine_similarity(query_embedding, item["embedding"])
            similarities.append((sim, item["text"], item["id"]))
        
        # 排序并返回前k个
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_items = similarities[:top_k]
        
        return top_items
    except Exception as e:
        print(f"知识库检索错误: {e}")
        return []


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
        # 检索知识库
        knowledge_results = search_knowledge_base(user_message)
        
        # 构建知识库上下文
        knowledge_context = ""
        if knowledge_results:
            knowledge_context = "根据产品说明书的相关信息：\n"
            for i, (sim, text, _) in enumerate(knowledge_results):
                if sim > 0.5:  # 只使用相似度高于0.5的结果
                    knowledge_context += f"{i+1}. {text}\n\n"
        
        # 构建对话历史
        chat_messages = []
        for msg in messages:
            chat_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 添加系统提示和知识库上下文
        if model == "qwen":
            system_prompt = "你是家护家电产品的智能客服，根据产品说明书回答用户问题。"
            if knowledge_context:
                system_prompt += "\n\n" + knowledge_context
            chat_messages.insert(0, {
                "role": "system",
                "content": system_prompt,
            })
            client = qwen_client
            model_name = QWEN_CHAT_MODEL
        else:
            system_prompt = "你是家护家电产品的智能客服，根据产品说明书回答用户问题。"
            if knowledge_context:
                system_prompt += "\n\n" + knowledge_context
            chat_messages.insert(0, {
                "role": "system",
                "content": system_prompt,
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
        return {"answer": answer, "knowledge_used": bool(knowledge_context)}
    except Exception as e:
        return {"error": str(e)}