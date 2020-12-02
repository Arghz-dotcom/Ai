import os
import re
from scipy import ndimage, misc
from skimage.transform import resize, rescale
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
print(tf.__version__)

import models


encoder = models.encoder
autoencoder = models.autoencoder
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

def train_batches(just_load_dataset=False):

    batches = 256 

    batch = 0 
    batch_nb = 0 
    max_batches = -1 
    
    ep = 4 

    images = []
    x_train_n = []
    x_train_down = []
    
    x_train_n2 = [] 
    x_train_down2 = []
    
    for root, dirnames, filenames in os.walk("../Datasets/cars_train"):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
                if batch_nb == max_batches: 
                    return x_train_n2, x_train_down2
                filepath = os.path.join(root, filename)
                image = pyplot.imread(filepath)
                if len(image.shape) > 2:
                        
                    image_resized = resize(image, (256, 256))
                    x_train_n.append(image_resized)
                    x_train_down.append(rescale(rescale(image_resized, 0.5, multichannel=True), 2.0, multichannel=True))
                    batch += 1
                    if batch == batches:
                        batch_nb += 1

                        x_train_n2 = np.array(x_train_n)
                        x_train_down2 = np.array(x_train_down)
                        
                        if just_load_dataset:
                            return x_train_n2, x_train_down2
                        
                        print('Training batch', batch_nb, '(', batches, ')')

                        autoencoder.fit(x_train_down2, x_train_n2,
                            epochs=ep,
                            batch_size=10,
                            shuffle=True,
                            validation_split=0.15)
                    
                        x_train_n = []
                        x_train_down = []
                    
                        batch = 0

    return x_train_n2, x_train_down2

x_train_n, x_train_down = train_batches(just_load_dataset=True)

autoencoder.load_weights('data/sr.img_net.mse.final_model5.no_patch.weights.best.hdf5')
encoder.load_weights('data/encoder_weights.hdf5')


encoded_imgs = encoder.predict(x_train_down)
print(encoded_imgs.shape)

sr1 = np.clip(autoencoder.predict(x_train_down), 0.0, 1.0)
image_index = np.random.randint(0, 256)

plt.figure(figsize=(128,128))
i = 1
ax = plt.subplot(10,10,i)
plt.imshow(x_train_down[image_index])
i += 1
ax = plt.subplot(10,10,i)
plt.imshow(x_train_down[image_index], interpolation='bicubic')
i += 1
ax = plt.subplot(10,10,i)
plt.imshow(encoded_imgs[image_index].reshape((64*64, 256)))
i += 1
ax = plt.subplot(10,10,i)
plt.imshow(sr1[image_index])
i += 1
ax = plt.subplot(10,10,i)
plt.imshow(x_train_n[image_index])
plt.show()