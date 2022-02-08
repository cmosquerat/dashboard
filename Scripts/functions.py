import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

def Erode_Dilate(image_path, kernel, operation="Erode", size=3):
    """Erode or Dilate an image

    Args:
        image_path (str): path of the image
        kernel (int): 0: square array of (sizexsize), 1: (1 x size) kernel 2: (size,) kernel
        operation (str, optional): Operation to apply ["Erode", "Dilate"]. Defaults to "Erode".
        size (int, optional): Size of the kernel. Defaults to 3.

    Returns:
        np.array? : Eroded or Dilated image
    """

    img = cv2.imread(image_path)
    morph_rect = np.ones((size,size))
    morph_row = np.ones((1,size))
    morph_column = np.ones(size)
    kernels = [morph_rect, morph_row, morph_column]

    if operation == "Erode":
        new_image = cv2.erode(img, kernels[kernel], iterations=1)

    if operation == "Dilate":
        new_image = cv2.dilate(img, kernels[kernel], iterations=1)

    return new_image


def CNN(input_image, method):
    """Apply a specific layer to an image

    Args:
        image_path (str): path of the image
        method (str): Layer to apply. ["Conv", "MaxPool", "AvgPool", "All"]

    Returns:
        tf.tensor: New image
    """
    images = np.array(input_image)
    img = tf.convert_to_tensor(images, dtype=tf.float32)
    img /= 255.0 
    images = np.array([img])

    

    if method == 'Conv':
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                padding="SAME", activation="relu")
        output_image = conv(images)

    if method == 'MaxPool':
        max_pool = tf.keras.layers.MaxPool2D(pool_size=2,dtype='float32')
        output_image = max_pool(images)

    if method == "AvgPool":
        avg_pool = tf.keras.layers.AvgPool2D(pool_size=2,dtype='float64')
        output_image = avg_pool(images)

    if method == "All":
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                padding="SAME", activation="relu")
        max_pool = tf.keras.layers.MaxPool2D(pool_size=2,dtype='float32')
        avg_pool = tf.keras.layers.AvgPool2D(pool_size=2,dtype='float64')

        image = conv(images)
        image = max_pool(image)
        output_image = avg_pool(image)

    return Image.fromarray(np.uint8(output_image.numpy()[0, :, :, 1]*255)).convert('RGB')
