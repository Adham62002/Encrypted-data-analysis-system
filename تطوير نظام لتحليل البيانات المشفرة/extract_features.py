from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import numpy as np
from utils.preprocess import load_and_preprocess_image

model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_feature(image_path):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)[0]
