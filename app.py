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
    test()
    return render_template('index.html')


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



def test():
    text = "ChatGPT（全名：Chat Generative Pre - trained Transformer），美国OpenAI[1]研发的聊天机器人程序[12]  ，于2022年11月30日发布[2 - 3]  。ChatGPT是人工智能技术驱动的自然语言处理工具，它能够通过理解和学习人类的语言来进行对话，还能根据聊天的上下文进行互动，真正像人类一样来聊天交流，甚至能完成撰写邮件、视频脚本、文案、翻译、代码，写论文[21]等任务。"
    return jsonify({'text': text})



if __name__ == '__main__':
    app.run(debug=True)
