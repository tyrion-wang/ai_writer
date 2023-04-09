from flask import Flask, render_template, request, jsonify
import openai
import os
import json
import gpt_lib
import logging
from flask_socketio import SocketIO, emit

# 配置openai的API Key
gpt_lib.set_openai_key()
# 初始化Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

logging.basicConfig(level=logging.DEBUG)
#logging.disable()

# 定义首页
@app.route('/')
def index():
    return render_template('index.html')


# 定义转写函数
@app.route('/transcribe', methods=['POST'])
def transcribe():
    # 获取用户输入的文字
    text = request.form['text']
    # 获取用户选择的相似度
    similarity = request.form['similarity']
    temperature = 1.0 - float(similarity) / 10.0
    #transcription = gpt_lib.chat(text, "围绕这个命题，生成一个800字的作文：", temperature)
    transcription = gpt_lib.chat(text, "总结这段文本，10个字以内：", temperature)
    #gpt_lib.chat_stream(text, "总结这段文本，10个字以内：", temperature)

    # 返回json格式的结果
    return jsonify({'transcription': transcription.strip()})


@socketio.on('my event')
def handle_my_custom_event(data):
    print('received data: ' + str(data))
    emit('my response', data, broadcast=True)

if __name__ == '__main__':
    app.run(debug=True)
