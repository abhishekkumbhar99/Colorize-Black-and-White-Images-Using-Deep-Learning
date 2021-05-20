import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

OUTPUT_CHANNELS = 2

def downsample(filters, size, apply_batchnorm=True):
  ''' Downsampling block'''  
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
    # result.add(tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
    #                                                  gamma_initializer=tf.random_normal_initializer(0., 0.02)))

  result.add(tf.keras.layers.LeakyReLU(0.2))

  return result

def upsample(filters, size, apply_dropout=False):
  ''' Upsamling block'''
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
#   # Upsampling2d + Conv 
#   result.add(tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear'))
#   result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=1,
#                                     padding='same',
#                                     kernel_initializer=initializer,
#                                     use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())
#   result.add(tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True,
                                                    #  gamma_initializer=tf.random_normal_initializer(0., 0.02)))

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def conv_1x1_bn_relu(filters):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2D(filters, 1, strides=1,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())

  return result


# pix2pix with bacth_sixe=32 100 epochs
def Generator():
  ''' U-net generator with 8 downsampling and 8 upsampling
  blocks like original Pix2pix model'''
  inputs = tf.keras.layers.Input(shape=[256,256,1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# pix2pix2 model with batch_sixe=8 for 100 epochs
def Generator2():
  ''' U-net generator with 4 downsampling and 4 upsampling
  blocks'''
  inputs = tf.keras.layers.Input(shape=[256,256,1])

  initializer = tf.random_normal_initializer(0., 0.02)

  first = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 4, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.LeakyReLU()
            ]) # (bs, 256, 256, 64)

  down_stack = [
    downsample(128, 4), # (bs, 128, 128, 128)     
    downsample(256, 4), # (bs, 64, 64, 256)     
    downsample(512, 4), # (bs, 32, 32, 512)     
    downsample(512, 4), # (bs, 16, 16, 512)       
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 32, 32, 1024)
    upsample(256, 4, apply_dropout=True), # (bs, 64, 64, 512)
    upsample(128, 4), # (bs, 128, 128, 256)
    upsample(64, 4),# (bs, 256, 256, 128)
        ]

  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, kernel_size=1,
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs


  # Downsampling through the model
  skips = []
  
  x = first(x)
  skips.append(x)

  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def Generator3():
  ''' U-net generator with pretrained MobileNetV2 as encoder 
  (frozen) and 5 upsampling blocks in decoder.'''
  inputs = tf.keras.layers.Input(shape=[256,256,1])

  initializer = tf.random_normal_initializer(0., 0.02)
  
  # convert grayscale to RGB by concatenating along last axis
  x = tf.concat([inputs]*3,axis=3)
  
  # Pretrained models accept RGB inputs only
  backbone = tf.keras.applications.MobileNetV2(input_shape=(256,256,3), include_top=False,weights='imagenet')

  # Use the activations of these layers
  layer_names = [
        'block_1_expand_relu',   # 128x128x96
        'block_3_expand_relu',   # 64x64x144
        'block_6_expand_relu',   # 32x32x192
        'block_13_expand_relu',  # 16x16x576
        'block_16_project',      # 8x8x320
  ]

  layers = [backbone.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=backbone.input, outputs=layers)
  down_stack.trainable = False

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 16, 16,)
    upsample(512, 4, apply_dropout=True), # (bs, 32, 32,)
    upsample(256, 4), # (bs, 64, 64,)
    upsample(128, 4),# (bs, 128, 128,)
        ]
  
  last_upsample = upsample(64,4) #(bs, 256, 256,)

  # if we use upsampling in this layer, output is blurry (Maybe)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, kernel_size=1,                        # Conv2DTranspose for other models except upsampling              
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)
  
  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last_upsample(x)
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# pix2pix3_resnet50 using pretrained encoder layers, freexed encoder, train decoder layer only
def Generator4():
  ''' U-net generator with pretrained Resnet50 as encoder 
  (frozen) and 4 upsampling blocks in decoder with 1x1 conv layers 
  between each block to reduce feature depth.'''
  inputs = tf.keras.layers.Input(shape=[256,256,1])

  initializer = tf.random_normal_initializer(0., 0.02)
  
  # convert grayscale to RGB by concatenating along last axis
  x = tf.concat([inputs]*3,axis=3)
  
  # Pretrained models accept RGB inputs only
  backbone = tf.keras.applications.ResNet50(input_shape=(256,256,3), include_top=False,weights='imagenet')
  # Use the activations of these layers

  layer_names = ['conv1_relu', #128x128x64
    'conv2_block3_out', #64x64x256
    'conv3_block4_out', #32x32x512
    'conv4_block6_out', # 16x16x1024
    'conv5_block3_out', # 8x8x2048
    ]

  layers = [backbone.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=backbone.input, outputs=layers)
  down_stack.trainable = False

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 16, 16, 512)
    upsample(512, 4, apply_dropout=True), # (bs, 32, 32, 512)
    upsample(256, 4), # (bs, 64, 64, 256)
    upsample(128, 4),# (bs, 128, 128, 128)
        ]
  conv_1x1_list =[
    conv_1x1_bn_relu(1024), # (bs, 8, 8, 512)
    conv_1x1_bn_relu(512), # (bs, 16, 16, 512)
    conv_1x1_bn_relu(512), # (bs, 32, 32, 512)
    conv_1x1_bn_relu(256) # (bs, 64, 64, 256)
  ]

  last_upsample = upsample(64,4) #(bs, 256, 256,)

  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, kernel_size=1,
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)
  
  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for c_layer, up, skip in zip(conv_1x1_list, up_stack, skips):
    x = c_layer(x)
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last_upsample(x)
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# pix2pix3_densenet121 using pretrained encoder layers, freexed encoder, train decoder layer only
def Generator5():
  inputs = tf.keras.layers.Input(shape=[256,256,1])

  initializer = tf.random_normal_initializer(0., 0.02)
  
  # convert grayscale to RGB by concatenating along last axis
  x = tf.concat([inputs]*3,axis=3)
  
  # Pretrained models accept RGB inputs only
  backbone = tf.keras.applications.DenseNet121(input_shape=(256,256,3), include_top=False,weights='imagenet')
  # Use the activations of these layers
# 'densenet121': (311, 139, 51, 4)
  layer_names = ['conv1/relu', #128x128x64  layer4
    'pool2_conv', #64x64x128                layer51
    'pool3_conv', #32x32x256                layer139
    'pool4_conv', # 16x16x512               layer311
    'relu', # 8x8x1024                      layer426(last)
    ]

  layers = [backbone.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=backbone.input, outputs=layers)
  down_stack.trainable = False

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 16, 16, 512)
    upsample(512, 4, apply_dropout=True), # (bs, 32, 32, 512)
    upsample(256, 4), # (bs, 64, 64, 256)
    upsample(128, 4),# (bs, 128, 128, 128)
        ]

  last_upsample = upsample(64,4) #(bs, 256, 256,)

  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, kernel_size=1,
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)
  
  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last_upsample(x)
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
  
  
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')          # grayscale image | L-channel
  tar = tf.keras.layers.Input(shape=[256, 256, OUTPUT_CHANNELS], name='target_image')         # (256, 256, 3) for RGB | (256, 256, 2) for ab channel

  x = tf.keras.layers.concatenate([inp, tar]) 

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
#   instnorm1 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
#                                                      gamma_initializer=tf.random_normal_initializer(0., 0.02))(conv)

  leaky_relu = tf.keras.layers.LeakyReLU(0.2)(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)