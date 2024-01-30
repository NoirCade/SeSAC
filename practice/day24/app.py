# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import os

global result

#Flask 객체 인스턴스 생성
app = Flask(__name__)
# 모델을 불러오거나 초기화
model = tf.keras.models.load_model(r"C:\Users\bluecom015\Desktop\SeSAC\practice\day24\static\model\keras_model.h5")
class_names = open(r"C:\Users\bluecom015\Desktop\SeSAC\practice\day24\static\model\labels.txt", "r").readlines()


def preprocess_image(image):
    # 이미지 전처리 로직을 추가 (크기 조정, 정규화 등)
    # 예제로 간단히 이미지 크기를 224x224로 조정하고 정규화합니다.
    # image = image.resize((224, 224))
    # image = np.array(image) / 255.0
    # image = np.expand_dims(image, axis=0)
    image = image.resize((224, 224))
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    return image


# 대문 페이지
@app.route('/')
def index():
  server_image_path = 'static/images/image_input.jpeg'
  return render_template('image_input.html', server_image_path=server_image_path)

@app.route('/next')
def animate():
   return render_template('animation.html')

# 예측 페이지
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # 업로드된 이미지 저장
        upload_folder = r'C:\Users\bluecom015\Desktop\SeSAC\practice\day24\static\uploads'
        file_path = os.path.join(upload_folder, file.filename)
        # file.save(file_path)

        image = Image.open(file.stream)
        processed_image = preprocess_image(image)

        # 모델에 이미지 전달하여 예측
        predictions = model.predict(processed_image)
        index = np.argmax(predictions)
        class_name = class_names[index]
        confidence_score = predictions[0][index]

        # 예측 결과를 클라이언트로 전송
        # result = {'prediction': predictions.tolist()}
        result = {"Class": class_name[2:],
                  "Confidence Score": str(np.round(confidence_score * 100))[:-2]}
        
        return redirect(url_for('animate'))


if __name__=="__main__":
  app.run(debug=True)
  # host 등을 직접 지정하고 싶다면
  # app.run(host="127.0.0.1", port="5000", debug=True)