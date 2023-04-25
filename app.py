from flask import Flask, render_template, request, jsonify, Response
import openai
import os
import json
import gpt_lib
import logging
from flask_socketio import SocketIO, emit
# import eventlet
# eventlet.monkey_patch()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI

# import unstructured
from dotenv import load_dotenv


# 配置openai的API Key
gpt_lib.set_openai_key()
# 初始化Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')
# socketio = SocketIO(app)

logging.basicConfig(level=logging.DEBUG)
# logging.disable()
load_dotenv()

# embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
# embeddings = OpenAIEmbeddings(openai_api_key=str(os.environ.get('OPENAI_API_KEY')))
# persist_directory = '/Users/bytedance/Downloads/database/'
# docsearch = Chroma(embedding_function=embeddings, persist_directory=persist_directory)


# 定义首页
@app.route('/')
def index():
    return render_template('index.html')

def test():
    print('test')


# 定义转写函数
@app.route('/transcribe', methods=['POST'])
def transcribe():
    # 获取用户输入的文字
    text = request.form['text']
    # 获取用户选择的相似度
    similarity = request.form['similarity']
    temperature = 1.0 - float(similarity) / 10.0
    #transcription = gpt_lib.chat(text, "围绕这个命题，生成一个800字的作文：", temperature)
    # transcription = gpt_lib.chat(text, "总结这段文本，10个字以内：", temperature)
    #gpt_lib.chat_stream(text, "围绕这个命题，生成一个800字的作文：", temperature, socketio)
    gpt_lib.chat_stream(text, "总结这段文本", temperature, socketio)
    # transcription = "123"
    # 返回json格式的结果
    # return jsonify({'transcription': transcription.strip()})


@socketio.on('my event')
def handle_my_custom_event(data):
    print('received data: ' + str(data))
    emit('my response', data, broadcast=True)


def gen_prompt(docs, query) -> str:
    return f"""To answer the question please only use the Context given, nothing else. Do not make up answer, simply say 'I don't know' if you are not sure.
Question: {query}
Context: {[doc.page_content for doc in docs]}
Answer:
"""

def prompt(query):
    # print(query)
    # docs = docsearch.similarity_search(query, k=4)
    # print(docs)
    # prompt = gen_prompt(docs, query)
    prompt = query
    prompt = "今天天气如何？"
    return prompt


def stream(input_text):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": "You're an assistant."},
            {"role": "user", "content": f"{prompt(input_text)}"},
        ], stream=True, max_tokens=500, temperature=0)
        for line in completion:
            if 'content' in line['choices'][0]['delta']:
                yield line['choices'][0]['delta']['content']

@app.route('/completion', methods=['GET', 'POST'])
def completion_api():
    # print("111")
    # return "111222333"
    if request.method == "POST":
        data = request.form
        input_text = data['input_text']
        return Response(stream(input_text), mimetype='text/event-stream')
    else:
        return Response(None, mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
    # socketio.run(app, host='127.0.0.1', port=5000, server='eventlet')
    # socketio.run(app)
