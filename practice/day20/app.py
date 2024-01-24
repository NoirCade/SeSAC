# app.py
from flask import Flask, render_template

#Flask 객체 인스턴스 생성
app = Flask(__name__)

@app.route('/') # 접속하는 url
def index():
  datas=[
    {'name': '반원', 'label': 60, 'point': 360, 'exp': 450000},
    {'name': '반원2', 'label': 2, 'point': 20, 'exp': 200},
    {'name': '반원3', 'label': 3, 'point': 30, 'exp': 300}
  ]
  return render_template('index.html', datas=datas)

@app.route('/index_table')
def index_table():
  datas=[
    {'name': '반원', 'label': 60, 'point': 360, 'exp': 450000},
    {'name': '반원2', 'label': 2, 'point': 20, 'exp': 200},
    {'name': '반원3', 'label': 3, 'point': 30, 'exp': 300}
  ]
  return render_template('index_table.html', datas=datas)

if __name__=="__main__":
  app.run(debug=True)
  # host 등을 직접 지정하고 싶다면
  # app.run(host="127.0.0.1", port="5000", debug=True)