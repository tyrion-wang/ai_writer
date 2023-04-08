import openai
import json

def set_openai_key(key):
    """
    设置OpenAI API密钥
    """
    openai.api_key = key

def gpt3_turbo(prompt, model, temperature=0.5, max_tokens=2048, n=1):
    """
    使用GPT-3.5-turbo模型生成文本

    prompt: 输入的文本
    model: 要使用的模型的名称或ID
    temperature: 温度参数，控制生成文本的随机性
    max_tokens: 最大令牌数，控制生成文本的长度
    n: 生成文本的数量
    """

    response = openai.Completion.create(
      engine=model,
      prompt=prompt,
      max_tokens=max_tokens,
      n=n,
      temperature=temperature
    )

    if response['choices']:
        return response['choices'][0]['text'].strip()
    else:
        return ''

def get_models():
    """
    获取可用的GPT-3模型列表
    """
    models = openai.Model.list()
    return [model.id for model in models['data']]

def create_model(name, model_type='text', language='en', max_tokens=2048, temperature=0.5):
    """
    创建一个新的GPT-3模型
    """
    model = openai.Model.create(
      id=name,
      model_type=model_type,
      language=language,
      max_tokens=max_tokens,
      temperature=temperature
    )
    return model.id

def delete_model(name):
    """
    删除一个GPT-3模型
    """
    try:
        openai.Model.delete(name)
        return True
    except:
        return False

def get_model_details(name):
    """
    获取一个GPT-3模型的详细信息
    """
    try:
        model = openai.Model.retrieve(name)
        return json.loads(str(model))
    except:
        return {}

def update_model(name, max_tokens=None, temperature=None):
    """
    更新一个GPT-3模型的参数
    """
    try:
        model = openai.Model.retrieve(name)
        if max_tokens:
            model.max_tokens = max_tokens
        if temperature:
            model.temperature = temperature
        model.save()
        return True
    except:
        return False
