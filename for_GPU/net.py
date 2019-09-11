#!/usr/bin/env python
import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h, n_h_2):
        super(VAE, self).__init__()
        with self.init_scope():
            # encoder 入力から隠れベクトルの作成
            self.le1 = L.Linear(n_in, n_h)
            self.le2 = L.Linear(n_h, n_h_2)
            #隠れベクトルから平均ベクトルの作成
            self.le3_mu = L.Linear(n_h_2, n_latent)  # 第１は入力信号数　
            #隠れベクトルから分散ベクトルの作成
            self.le3_ln_var = L.Linear(n_h_2, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h_2)
            self.ld2 = L.Linear(n_h_2, n_h)
            self.ld3 = L.Linear(n_h, n_in)

    def forward(self, x, sigmoid= False):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        h2 = F.tanh(self.le2(h1))
        mu = self.le3_mu(h2)
        ln_var = self.le3_ln_var(h2)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=False):
        h1 = F.tanh(self.ld1(z))
        h2 = F.tanh(self.ld2(h1))
        h3 = self.ld3(h2)
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3

    def get_loss_func(self, beta=1.0, k=1):
        """
        VAEの損失の計算
        Args:
            C (int): 正則化項をどれだけ効かせるかの変数、通常1.0が使用される
            k (int): サンプルを何回行うか
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # 復元誤差の計算
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                # rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)
                rec_loss += F.bernoulli_nll(
                    x, self.decode(z, sigmoid=False)) / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                beta * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf

