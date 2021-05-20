import tensorflow as tf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb


IMG_WIDTH = 256
IMG_HEIGHT = 256


# https://stackoverflow.com/questions/62544528/tensorflow-decodejpeg-expected-image-jpeg-png-or-gif-got-unknown-format-st
# remove invalid images
def is_valid_image(filename, verbose=False):

    data = open(filename,'rb').read(10)

    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True

    # check if file is PNG
    if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        if verbose == True:
             print(filename+" is: PNG.")
        return True

    return False

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image,channels=3)
 
    img_gray = tf.image.rgb_to_grayscale(image)
    img_rgb = image 

    img_gray = tf.cast(img_gray, tf.float32)
    img_rgb = tf.cast(img_rgb, tf.float32)

    return img_gray, img_rgb

def load_lab_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image,channels=3)
    #Resize to 256x256
    image = tf.image.resize(image, size=[IMG_HEIGHT,IMG_WIDTH])
    img_lab = rgb2lab(image/255.0)

    img_l = tf.cast(np.expand_dims(img_lab[:,:,0], axis=-1), tf.float32)
    img_ab = tf.cast(img_lab[:,:,1:], tf.float32)

    # normalizing the images to [-1, 1]
    img_l = img_l/50.0 - 1      # L:0-100
    img_ab = img_ab/128         # a:-127 to 128, b:-128 to 127

    return img_l, img_ab

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


def load_image_train(image_file):
  input_image, real_image = load_image(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load_image(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image
  
def to_rgb(l, ab):
    '''rgb image from grayscale and ab channels'''
    l = (l + 1)*50
    ab = ab*128
    lab_img = tf.concat((l,ab),axis=-1)
    rgb_img = lab2rgb(lab_img)
    # getting the pixel values between [-1, 1]
    rgb_img = (rgb_img*255)/127.5 - 1
    return rgb_img

def generate_images(model, test_input, tar, lab=True):
  start = time.time()
  print(test_input.shape, tar.shape)
  prediction = model(test_input, training=False)        #training=False for Lab images only or RGB w/o dropout
  end = time.time()
  print('Inference Time: ',end-start)
  plt.figure(figsize=(15,15))
  
  # for Lab image
  if lab:
      display_list = [tf.squeeze(test_input[0]), to_rgb(test_input[0],tar[0]),
                  to_rgb(test_input[0],prediction[0])]
  else:
      # for RGB image
      display_list = [tf.squeeze(test_input[0]), tar[0], prediction[0]]
  
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5,cmap='gray')
    plt.axis('off')
  plt.show()