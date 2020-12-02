import os
from skimage.transform import resize, rescale
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

import models
encoder = models.encoder
autoencoder = models.autoencoder

autoencoder.load_weights('data/sr.img_net.mse.final_model5.no_patch.weights.best.hdf5')
encoder.load_weights('data/encoder_weights.hdf5')

image_index = np.random.randint(0, 256)
count = 0
image_resized2 = None
image_down2 = None
for root, dirnames, filenames in os.walk("../Datasets/cars_train"):
    for filename in filenames:
        if count == image_index:
            filepath = os.path.join(root, filename)
            image = pyplot.imread(filepath)
            image_resized = np.array(list(resize(image, (256, 256))))
            image_resized2 = np.expand_dims(image_resized, axis=0)
            image_down = np.array(list(rescale(rescale(image_resized, 0.5, multichannel=True), 2.0, multichannel=True)))
            image_down2 = np.expand_dims(image_down, axis=0)
        count += 1

encoded_imgs = encoder.predict(image_down2)
print(encoded_imgs.shape)

sr1 = np.clip(autoencoder.predict(image_down2), 0.0, 1.0)

image_down2 = np.squeeze(image_down2, axis=0)

plt.figure(figsize=(128,128))
i = 1
ax = plt.subplot(10,10,i)
plt.imshow(image_down2)
i += 1
ax = plt.subplot(10,10,i)
plt.imshow(image_down2, interpolation='bicubic')
i += 1
ax = plt.subplot(10,10,i)
plt.imshow(encoded_imgs[0].reshape((64*64, 256)))
i += 1
ax = plt.subplot(10,10,i)
plt.imshow(sr1[0])
i += 1
ax = plt.subplot(10,10,i)
plt.imshow(image_down2)
plt.show()