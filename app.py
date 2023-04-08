from flask import Flask, render_template, request, jsonify
import openai
import os
import json
import gpt_lib

# 配置openai的API Key
gpt_lib.set_openai_key()
# 初始化Flask
app = Flask(__name__)



# 定义首页
@app.route('/')
def index():
    return render_template('index.html', text="123123")


# 定义转写函数
@app.route('/transcribe', methods=['POST'])
def transcribe():
    # 获取用户输入的文字
    text = request.form['text']
    # 获取用户选择的相似度
    similarity = request.form['similarity']
    temperature = 1.0 - float(similarity) / 10.0

    # 使用openai的GPT-3模型进行文本加工
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        temperature=temperature,
        max_tokens=1024,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )

    # 提取GPT-3的输出文本
    transcription = response.choices[0].text

    # transcription = gpt_lib.create_completion(text, "gpt-3.5-turbo")
    # 返回json格式的结果
    return jsonify({'transcription': transcription.strip()})

if __name__ == '__main__':
    app.run(debug=True)
