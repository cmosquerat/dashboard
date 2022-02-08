import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


model = tf.keras.models.load_model('Scripts/dogs_cats_model.h5')

#### Para que funcionen, tiene que estar el archivo model.h5 en el mismo directorio, no se incluye en el repositorio por el peso

def predict(X):
    """Predict wheter an RGB image contains a dor or a cat

    Args:
        X (np.ndarray): array of dimentions [width, height, 3]
    
    Returns:
        str: either 'dog' or 'cat'
    """
    X = np.array(X)
    X = X/255.
    X = cv2.resize(X, (224,224))
    X = np.expand_dims(X, axis = 0)
    
    Y = model.predict(X)
    
    if Y > 0.5:
        return "dog"
    else:
        return "cat"
    
    
def predict_from_file(path):
    """Predict wheter an RGB image contains a dor or a cat

    Args:
        path (str): path of the image (jpg or jpeg)

    Returns:
        str: either 'dog' or 'cat'
    """
    X = plt.imread(path)
    return predict(X)