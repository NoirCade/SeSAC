from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "GET":
       return render_template('stock.html')
    
    if request.method == "POST":
      stock_num = (request.form['stock_num'])
      url = 'http://companyinfo.stock.naver.com/v1/company/c1010001.aspx?cmp_cd=' + stock_num
      datas = pd.read_html(url, encoding='utf-8')
      header = datas[0].iloc[:,0][0]
      datas = datas[12]
      datas = datas.to_dict(orient='records')

    return render_template('stock.html', datas=datas, header=header)

if __name__=="__main__":
  app.run(debug=True)