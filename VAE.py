'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
Should give ~155 score
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, merge
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.activations import relu
# from .. import train_viz.trainViz

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
nb_epoch = 100
epsilon_std = 1.0
decoder_type = "Bernoulli"  # Can be Gaussian or Bernoulli


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def affine_relu(x):
    return 0.01 + relu(1 + x)


x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')

if decoder_type == "Bernoulli":
    decoder_mean = Dense(original_dim, activation='sigmoid')
else:
    decoder_mean = Dense(original_dim, activation='relu')

decoder_sigma_raw = Dense(original_dim)  # Not used if Bern
decoder_sigma = Lambda(affine_relu, output_shape=(original_dim,))

h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
x_decoded_sigma = decoder_sigma(decoder_sigma_raw(h_decoded))

if decoder_type == "Bernoulli":
    x_decoded_full = x_decoded_mean
else:
    x_decoded_full = merge([x_decoded_mean, x_decoded_sigma],
                           mode='concat', concat_axis=1)


def vae_loss(x, x_decoded_full):
    if decoder_type == "Bernoulli":
        x_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        x_loss = original_dim * x_loss
    else:
        x_loss = K.log(x_decoded_sigma)
        x_loss += 0.5 * K.square((x - x_decoded_mean) / x_decoded_sigma)
        x_loss = K.sum(x_loss, axis=-1)
    kl_loss_mat = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = - 0.5 * K.sum(kl_loss_mat, axis=-1)
    return x_loss + kl_loss


vae = Model(input=x, output=x_decoded_full)
vae.compile(optimizer='adadelta', loss=vae_loss,
            metrics=["binary_crossentropy"])
# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.load_weights("VAE_100_epochs.weights")

if True:
    history = vae.fit(x_train, x_train,
                      shuffle=True,
                      nb_epoch=nb_epoch,
                      batch_size=batch_size,
                      validation_data=(x_test, x_test))

vae.save_weights("VAE_200_epochs.weights")


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

plt.hist(x_test_encoded[:, 0])
plt.show()
plt.hist(x_test_encoded[:, 1])
plt.show()

print("Correlation between the latent variables :")
print(np.corrcoef(x_test_encoded[:, 0], x_test_encoded[:, 1]))

decoded_imgs = vae.predict(x_test, batch_size=batch_size)
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
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_sigma = decoder_sigma(decoder_sigma_raw(_h_decoded))
generator = Model(decoder_input, _x_decoded_mean)
generator_sigma = Model(decoder_input, _x_decoded_sigma)

for gen in [generator, generator_sigma]:
    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through
    # the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z,
    # since the prior of the latent
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
