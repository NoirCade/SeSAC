import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("../day24/static/model/keras_model.h5")

if __name__ == "__main__":
    image = Image.open('../practice/day24/static/images/image_input.jpeg')
    
    for l in model.layers:
        print("layer: ", l.name, ", expects input of shape: ", l.input_shape)