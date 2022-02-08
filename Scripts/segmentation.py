import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from matplotlib import cm
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection)/(union), axis=0)
    return dice

def iou_coef(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection) / (union), axis=0)
    return iou


def binary_segmentation(input_img, mode, threshold = None):
    """Apply binary segmentation to a grayscale image using one of the avaible modes
    

    Args:
        input_path (str): relative path to the input image file
        output_path (str, optional): relative path of the output image. Defaults to './output.png'
        mode (str): One of the following: 'kmeans', 'threshold', 'growing_splitting' or 'split_merge'
        threshold (float, optional): threshold to by applied if threshold mode is selected

        If no threshold is defined when using threshold mode, it will be automatically 
        selected using the mean value of the image.
        
    Returns
        (np.array): array of the generated segmentation mask
    """
    
    input_img = np.array(input_img)/255
    if mode == 'kmeans':
        X = input_img.reshape(-1,1)
        kmeans = KMeans(n_clusters = 2).fit(X)
        output_img = kmeans.cluster_centers_[kmeans.labels_].reshape(input_img.shape)
    elif mode == 'threshold':
        if threshold == None:
            threshold = np.mean(input_img.reshape(-1))    #compute threshold
            # print(f'Image median: {np.median(input_img.reshape(-1))}')
            # print(f'Image mean: {np.mean(input_img.reshape(-1))}')
        output_img = np.where(input_img > threshold,1, 0)
        pass
    elif mode == 'growing_splitting':
        pass
    elif mode == 'split_merge':
        pass
    
    im = Image.fromarray(np.uint8(cm.gist_earth(output_img)*255))
    return im
    
    

def multiclass_segmentation(input_img, mode, classes_number = 2):
    """Apply semantic segmentation to a given image and save it to output

    Args:
        input_path (str): relative path to the input image file
        mode (str): one of the following segmentation modes: 'kmeans' or 'deep'
        classes_number (int, optional): Total number of classes to segmentate. Defaults to 2
        output_path (str, optional): relative path to save the segmentation mask. Defaults to './output.png'.
        
    The deep mode is experimental and only works well with thermal images of feet
    """
    input_img = np.array(input_img)/255. # Read and scale input image
    
    if mode == 'kmeans':
        X = input_img.reshape(-1,3)
        kmeans = KMeans(n_clusters = classes_number).fit(X)
        
        output_img = kmeans.cluster_centers_[kmeans.labels_].reshape(input_img.shape)
    elif mode == 'deep':
        img_size = 224
        X = tf.convert_to_tensor(input_img)
        X = tf.image.resize(X,(img_size,img_size))
        X = tf.expand_dims(X,0)
        
        model_path = 'Scripts/feet_model.h5'
        model = tf.keras.models.load_model(model_path, custom_objects = {'dice_coef':dice_coef, 'iou_coef':iou_coef})
        
        threshold = 0.5
        Y = model.predict(X)   
        Y = Y/Y.max()
        Y = np.where(Y>=threshold,1,0)

        Y = tf.image.resize(Y, (input_img.shape[0],input_img.shape[1])) # Resize the prediction to have the same dimensions as the input
        output_img = np.array(Y[0,:,:,0])

    
        
    
    return Image.fromarray(np.uint8(output_img*255)).convert('RGB')



# Tests

