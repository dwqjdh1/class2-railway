import os
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

