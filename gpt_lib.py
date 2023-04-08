import openai
import json
import os

def set_openai_key():
    """
    设置OpenAI API密钥
    """
    openai.api_key = os.environ.get('OPENAI_API_KEY')


def create_completion(prompt, model, temperature=0.5, max_tokens=1024, n=1, stop=None, presence_penalty=0.0,
                      frequency_penalty=0.0, echo=False):
    """
    使用GPT-3.5-Turbo模型生成文本。

    Args:
        prompt (str): 模型生成文本的起始语句。
        model (str): 使用的模型名称。
        temperature (float): 控制随机性的温度值。默认值为0.5。
        max_tokens (int): 生成文本的最大长度。默认值为1024。
        n (int): 生成多少个响应。默认值为1。
        stop (str): 用于停止生成的文本的标记符。默认值为None。
        presence_penalty (float): 控制生成文本中是否存在特定单词的罚分。默认值为0.0。
        frequency_penalty (float): 控制生成文本中特定单词的使用频率的罚分。默认值为0.0。
        echo (bool): 是否将用户的输入添加到生成的文本中。默认值为False。

    Returns:
        str: 生成的文本响应。
    """
    prompt = f"{prompt.strip()}{{}}"
    messages = [{"type": "input", "text": prompt}]
    if echo:
        messages.append({"type": "input", "text": "{0}"})
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        model=model,
        messages=messages,
    )
    return response.choices[0].text.strip()


def list_models():
    """
    列出当前支持的所有模型名称。

    Returns:
        list: 所有模型名称的列表。
    """
    models = []
    for model in openai.Model.list():
        models.append(model.id)
    return models


def get_model_info(model):
    """
    获取指定模型的详细信息。

    Args:
        model (str): 模型名称。

    Returns:
        dict: 包含模型详细信息的字典。
    """
    return openai.Model.retrieve(model).to_dict()


def list_engines():
    """
    列出当前支持的所有引擎名称。

    Returns:
        list: 所有引擎名称的列表。
    """
    engines = []
    for engine in openai.Engine.list():
        engines.append(engine.id)
    return engines


def get_engine_info(engine):
    """
    获取指定引擎的详细信息。

    Args:
        engine (str): 引擎名称。

    Returns:
        dict: 包含引擎详细信息的字典。
    """
    return openai.Engine.retrieve(engine).to_dict()
