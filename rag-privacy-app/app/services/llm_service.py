from flask import current_app
import requests
import os

class LLMService:
    def __init__(self):
        self.api_url = current_app.config['LLM_API_URL']
        self.api_key = current_app.config['LLM_API_KEY']

    def query_llm(self, prompt):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'prompt': prompt,
            'max_tokens': 150
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            return f"Error: {response.status_code} - {response.text}"

def query_llm_api(prompt: str):
    """
    针对火山引擎 API 的简化调用方式。
    """
    # 从 .env 读取配置
    api_key = os.getenv('LLM_SECRET_ACCESS_KEY')
    model_id = os.getenv('LLM_MODEL_ID') # 这是推理终端 ID (EP ID)
    api_url = os.getenv('LLM_API_URL') or "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    if not api_key or not model_id:
        return f"错误：.env 配置不完整。Key: {'已设置' if api_key else '未设置'}, Model: {'已设置' if model_id else '未设置'}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        print(f"DEBUG: Using KEY: {api_key[:10]}...")
        print(f"DEBUG: Using MODEL ID: {model_id}")
        print(f"DEBUG: Using URL: {api_url}")
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        # 强制打印所有返回内容，方便我们调试
        print(f"DEBUG: Status Code: {response.status_code}")
        print(f"DEBUG: Raw Response: {response.text}")
        
        if response.status_code != 200:
            return f"API 报错: {response.status_code} - {response.text}"
            
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        return f"系统异常: {str(e)}"

def get_llm_response(prompt: str):
    """
    Gets a response from the LLM API.
    """
    api_key = os.getenv('LLM_SECRET_ACCESS_KEY')

    # This is a placeholder for the actual API call.
    # You will need to replace the URL and the request structure
    # with the actual ones from your LLM provider's documentation.
    
    # For example, if using a service like OpenAI:
    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json"
    # }
    # data = {
    #     "model": "some-model-name",
    #     "prompt": prompt,
    #     "max_tokens": 150
    # }
    # response = requests.post("https://api.example.com/v1/completions", headers=headers, json=data)
    
    # For now, we'll just return a simulated response.
    print(f"--- Making API call with key: ...{api_key[-4:] if api_key else 'None'}")
    
    if not api_key:
        return "Error: LLM_API_KEY not configured."
        
    # Simulate a successful response
    simulated_response = f"This is a simulated response for the prompt: '{prompt}'"
    return simulated_response