from keras.layers import Input, Dense
from keras.models import Model

from keras.callbacks import TensorBoard

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


encoding_dim = 2
n_epoch = 200
n_hidden = 256

input_img = Input(shape=(784,))
x = Dense(n_hidden, activation="relu")(input_img)
encoded = Dense(encoding_dim)(x)

encoded_input = Input(shape=(encoding_dim,))
x = Dense(n_hidden, activation="relu")(encoded_input)
decoded = Dense(784, activation='sigmoid')(x)

encoder = Model(input=input_img, output=encoded)
decoder = Model(encoded_input, decoded)
# encoder.summary()
# decoder.summary()
autoencoder = Model(input_img, decoder(encoder(input_img)))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

history = autoencoder.fit(x_train, x_train,
                          nb_epoch=n_epoch,
                          batch_size=100,
                          shuffle=True,
                          validation_data=(x_test, x_test))

autoencoder.save_weights("FCN_200.weights")

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

plt.hist(encoded_imgs[:, 0])
plt.show()
plt.hist(encoded_imgs[:, 1])
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
plt.show()


# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=32)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution

generator = decoder

for gen in [generator]:
    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through
    # the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the
    # latent space is Gaussian
    grid_x = np.linspace(np.min(encoded_imgs[:, 0]), np.max(encoded_imgs[:, 0]), n)
    grid_y = np.linspace(np.min(encoded_imgs[:, 1]), np.max(encoded_imgs[:, 1]), n)

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


