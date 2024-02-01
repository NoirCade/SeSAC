# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import os


#Flask 객체 인스턴스 생성
app = Flask(__name__)
# 세션 활성화
app.secret_key = 'session_secret_keeeeeey'
# 모델을 불러오거나 초기화
model = tf.keras.models.load_model("./static/model/keras_model.h5")
try:
    print(model)
except:
    print('Model load failed')
class_names = open("./static/model/labels.txt", "r").readlines()


def preprocess_image(image):
    # 이미지 전처리 로직을 추가 (크기 조정, 정규화 등)
    image = image.resize((224, 224))

    # 채널 수가 4개라면 알파 채널을 무시하고 RGB로 변환
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    return image


# 대문 페이지
@app.route('/')
def home():
  return render_template('index.html')


# 이미지 업로드 페이지
@app.route('/image_input')
def image_input():
#   server_image_path = 'static/images/image_input.jpeg'
    return render_template('image_input.html')

@app.route('/predict')
def predicting():
    # 여기에 결과 페이지에 보여줄 이미지 경로를 전달하는 로직을 추가할 수 있습니다.
    image_path = url_for('static', filename='images/A_headon_shot_of_the_goalkeeper_in_front_of_the.jpg')
    return render_template('predicting.html', image_path=image_path)


# 중간 애니메이션 페이지
@app.route('/next_page')
def next_page():
    return render_template('next_page.html')


@app.route('/final_page')
def final_page():
    return render_template('final_page.html')


@app.route('/keeperprediction')
def keeperprediction():
    return render_template('keeperprediction.html')


# 예측 기능 구현 페이지
@app.route('/prediction', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # 업로드된 이미지 저장
        upload_folder = './static/uploads'
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        image = Image.open(file.stream)
        processed_image = preprocess_image(image)

        # 모델에 이미지 전달하여 예측
        predictions = model.predict(processed_image)
        index = np.argmax(predictions)
        class_name = class_names[index]
        confidence_score = predictions[0][index]

        # 예측 결과를 result 변수에 저장하고, 페이지 전환
        # result = {'prediction': predictions.tolist()}
        result = {
            "redirect_url": '/predict',
            "file_name": file.filename,
            "prediction": {
                "Class": class_name[2:],
                "Confidence Score": str(np.round(confidence_score * 100))[:-2]
            }
        }

        # result값 확인
        print(result)
        session['result'] = result

        return jsonify(session['result'])


# 분석 페이지
@app.route('/report')
def report():
   result = session.get('result', {})
   data = {"file_name": result.get('file_name', ''),
           "prediction": result.get('prediction', {})}
   user_image_path = 'static/uploads/' + data['file_name']
   print(user_image_path)
   return render_template('report.html', user_image_path=user_image_path, data=data)


if __name__=="__main__":
  app.run(debug=True)
  # host 등을 직접 지정하고 싶다면
  # app.run(host="127.0.0.1", port="5000", debug=True)