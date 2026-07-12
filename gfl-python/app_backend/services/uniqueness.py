import numpy as np
import tensorflow as tf
import os

from app_backend.utils.helpers import preprocess_image
from app_backend.config import Uniqueness_MODEL_PATH, THRESHOLD

# Register and load model (same as detect.py)
@tf.keras.utils.register_keras_serializable()
def l2_normalize(x):
    return tf.keras.backend.l2_normalize(x, axis=1)

embedding_model = tf.keras.models.load_model(
    Uniqueness_MODEL_PATH,
    custom_objects={'l2_normalize': l2_normalize}
)

def compare_two_images(input_img_path, db_img_path):
    input_img = preprocess_image(input_img_path)
    db_img = preprocess_image(db_img_path)

    if input_img is None or db_img is None:
        return {
            'error': 'Failed to preprocess one or both images.',
            'similar': False
        }

    input_embedding = embedding_model.predict(input_img)[0]
    db_embedding = embedding_model.predict(db_img)[0]

    distance = np.linalg.norm(input_embedding - db_embedding)
    is_similar = bool(distance < THRESHOLD) 
    return {
        'similar': is_similar,
        'distance': float(distance),
        'threshold': THRESHOLD
    }
