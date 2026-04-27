import os
import dashscope
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 根路径返回前端页面
@app.get("/")
async def root():
    return app.send_file("static/index.html")

# 聊天接口
@app.post("/chat")
async def chat(request: Request):
    try:
        # 获取请求数据
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return JSONResponse({"error": "请输入消息"})
        
        # 设置API Key
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return JSONResponse({"error": "没有读取到环境变量 DASHSCOPE_API_KEY。请在 Railway Variables 中配置它。"})
        
        dashscope.api_key = api_key
        
        # 调用千问大模型
        response = dashscope.Generation.call(
            model="qwen-plus",
            messages=[
                {
                    "role": "system",
                    "content": "你是智能客服助手，回答要准确、简洁、友好。",
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            temperature=0.3,
        )
        
        # 处理响应
        if response.status_code == 200:
            answer = response.output.text
            return JSONResponse({"answer": answer})
        else:
            return JSONResponse({"error": f"调用失败: {response.status_code}, {response.message}"})
    
    except Exception as e:
        return JSONResponse({"error": f"发生错误: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
