import chainer
import numpy as np
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda
from chainer import serializers
import numpy as np
from sklearn.model_selection import train_test_split
import net


model = net.VAE(784, 20, 500,250)
chainer.serializers.load_npz('sample.npz', model)

train, test = chainer.datasets.get_mnist(withlabel=False)

result = 'result'
def save_images(x, filename):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
        for ai, xi in zip(ax.flatten(), x):
            ai.imshow(xi.reshape(28, 28))
        fig.savefig(filename)

train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
x = chainer.Variable(np.asarray(train[train_ind]))

with chainer.using_config('train', False), chainer.no_backprop_mode():
    x1 = model(x)
save_images(x.array, os.path.join(result, 'train'))
save_images(x1.array, os.path.join(result, 'train_reconstructed'))

test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
x = chainer.Variable(np.asarray(test[test_ind]))
with chainer.using_config('train', False), chainer.no_backprop_mode():
    x1 = model(x)
save_images(x.array, os.path.join(result, 'test'))
save_images(x1.array, os.path.join(result, 'test_reconstructed'))

# draw images from randomly sampled z
z = chainer.Variable(
    np.random.normal(0, 1, (9, 20)).astype(np.float32))
x = model.decode(z)
save_images(x.array, os.path.join(result, 'sampled'))

