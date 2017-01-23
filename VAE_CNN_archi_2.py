'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
Should yield 160
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, merge, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.activations import relu
from keras.optimizers import Adadelta

batch_size = 100
original_dim = (1, 28, 28)
full_size = int(np.prod(original_dim))
print(type(full_size))
latent_dim = 2
intermediate_dim = 256
nb_epoch = 30
epsilon_std = 1.0
decoder_type = "Bernoulli"  # Can be Gaussian or Bernoulli


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def affine_relu(x):
    return 0.01 + relu(1 + x)

# Add Batch norm at some point
x = Input(batch_shape=(batch_size,) + original_dim)
y = Convolution2D(16, 3, 3, activation='sigmoid', border_mode='same')(x)
y = MaxPooling2D((2, 2), border_mode='same')(y)
y = BatchNormalization()(y)
#(8,14,14)
y = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(y)
y = MaxPooling2D((2, 2), border_mode='same')(y)
#y = BatchNormalization()(y)
#(8,7,7)
#y = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(y)
#y = MaxPooling2D((2, 2), border_mode='same')(y)
#y = BatchNormalization()(y)
#(8, 4, 4)
y = Flatten()(y)

z_mean = Dense(latent_dim)(y)
z_log_var = Dense(latent_dim)(y)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model(x, z_mean)
encoder_full = Model(x, z)
encoder_full.summary()

z_input = Input(batch_shape=(batch_size, latent_dim))
y = Dense(8*7*7)(z_input)
y = Reshape((8, 7, 7))(y)
#y = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(y)
#y = UpSampling2D((2, 2))(y)
#y = BatchNormalization()(y)
y = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(y)
y = UpSampling2D((2, 2))(y)
#y = BatchNormalization()(y)
y = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(y)
y = UpSampling2D((2, 2))(y)
y = BatchNormalization()(y)
x_decoded_mean = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(y)

decoder_h = Model(z_input, x_decoded_mean)
decoder_h.summary()

def vae_loss(x, x_decoded_full):
    if decoder_type == "Bernoulli":
        x_true = x
        x_pred = x_decoded_full
        x_true = K.reshape(x_true, (batch_size, full_size))
        x_pred = K.reshape(x_pred, (batch_size, full_size))
        x_loss = full_size * objectives.binary_crossentropy(x_true, x_pred)
    else:
        x_loss = K.log(x_decoded_sigma)
        x_loss += 0.5 * K.square((x - x_decoded_mean) / x_decoded_sigma)
        x_loss = K.sum(x_loss, axis=-1)
    kl_loss_mat = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = - 0.5 * K.sum(kl_loss_mat, axis=-1)
    return x_loss + kl_loss

x_mean = decoder_h(encoder_full(x))
vae = Model(x, x_mean)
vae.compile(optimizer=Adadelta(), loss=vae_loss)
vae.summary()


# train the VAE on MNIST digits
print("Loading data ...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train),) + original_dim)
x_test = x_test.reshape((len(x_test),) + original_dim)
print("Done !")

print(np.min(x_train))

history = vae.fit(x_train, x_train,
                  shuffle=True,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
#vae.save_weights("VAE_CNN_100_epochs.weights")

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
x_test_encoded = encoder.predict(x_train, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_train)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_x_decoded_mean = decoder_h(decoder_input)
# _x_decoded_sigma = decoder_sigma(decoder_sigma_raw(_h_decoded))
generator = Model(decoder_input, _x_decoded_mean)
# generator_sigma = Model(decoder_input, _x_decoded_sigma)
gens = [generator]#, generator_sigma]
for gen in gens:
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
