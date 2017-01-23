from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Dropout

from keras.callbacks import TensorBoard

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


encoding_dim = 2
n_epoch = 50
input_img = Input(shape=(1, 28, 28))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = BatchNormalization()(x)
x = Flatten()(x)
encoded = Dense(encoding_dim)(x)
# at this point the representation is (8, 4, 4) i.e. 128-dimensional
encoded_input = Input(shape=(encoding_dim,))
x = Dense(128)(encoded_input)
x = Reshape((8, 4, 4))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
#x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)


encoder = Model(input_img, encoded)
decoder = Model(encoded_input, decoded)
#encoder.summary()
#decoder.summary()
autoencoder = Model(input_img, decoder(encoder(input_img)))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


(x_train, x_test), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
print(x_train.shape)
print(x_test.shape)

#autoencoder.load_weights("CNN_epoch_50.weights")
history = autoencoder.fit(x_train, x_train,
                          nb_epoch=n_epoch,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

#autoencoder.save_weights("CNN_epoch_50.weights")

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

plt.hist(encoded_imgs[:, 0])
plt.show()
plt.hist(encoded_imgs[:, 1])
plt.show()

print("Correlation between the latent variables :")
print(np.corrcoef(encoded_imgs[:, 0], encoded_imgs[:, 1]))

# display a 2D plot of the digit classes in the latent space
plt.figure(figsize=(6, 6))
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# build a digit generator that can sample from the learned distribution

generator = decoder

for gen in [generator]:
    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through
    # the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent
    # space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    grid_x = grid_x * np.std(encoded_imgs[:, 0])
    grid_x = grid_x + np.mean(encoded_imgs[:, 0])

    grid_y = grid_y * np.std(encoded_imgs[:, 1])
    grid_y = grid_y + np.mean(encoded_imgs[:, 1])

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = gen.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.colorbar()
    plt.show()

plt.plot(history.history["loss"], color='blue')
plt.plot(history.history["val_loss"], color='red')
plt.show()
