from flask import Flask, render_template

#Flask 객체 인스턴스 생성
app = Flask(__name__)

@app.route('/') # 접속하는 url
def index():
    # 이미지 파일 경로 리스트 (실제 파일 경로로 대체해야 합니다.)
    images_data = [
        {'path': 'images/image01.jpg', 'description': '이미지 설명1'},
        {'path': 'images/image02.jpg', 'description': '이미지 설명2'},
        {'path': 'images/image03.jpg', 'description': '이미지 설명3'},
        {'path': 'images/image04.jpg', 'description': '이미지 설명4'},
        {'path': 'images/image05.jpg', 'description': '이미지 설명5'},
        {'path': 'images/image06.jpg', 'description': '이미지 설명6'},
    ]

    return render_template('img.html', images_data=images_data)

if __name__ == "__main__":
    app.run(debug=True)