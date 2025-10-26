# 新的 API 的调用模板 - 正确的图像识别格式

import requests
import base64

url = "https://llm.q2zy.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer MUSO_API_KEY_REMOVED"
}

def image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 读取测试图片
image_path = "../frames/包/frame_013.jpg"
image_base64 = image_to_base64(image_path)

prompt = """进行推理判断 分析这个字最大概率是什么  该字可能未写完 字严格按照汉字标准书写 
回答格式如下：
Answer:"字" """

# 正确的API调用格式 - OpenAI标准多模态格式
data = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "system",
            "content": "你是一个汉字识别助手"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    "stream": False,
    "max_tokens": 60,
    "temperature": 0.7
}

# 发起请求
response = requests.post(url, headers=headers, json=data)

# 解析结果
if response.status_code == 200:
    result = response.json()
    print("完整返回：", result)
    print("\nAI 回复：", result["choices"][0]["message"]["content"])
    print(f"\nTokens使用情况:")
    print(f"  输入tokens: {result['usage']['prompt_tokens']}")
    print(f"  输出tokens: {result['usage']['completion_tokens']}")
    print(f"  总计tokens: {result['usage']['total_tokens']}")
else:
    print(f"API调用失败，状态码: {response.status_code}")
    print("错误信息:", response.text)