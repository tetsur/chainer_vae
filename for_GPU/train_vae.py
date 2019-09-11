#!/usr/bin/env python

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


def main():
    parser = argparse.ArgumentParser(description='Chainer: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='path/to/output',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # net内VAEオブジェクトの生成
    model = net.VAE(784, args.dimz,500,250)
    if 0 <= args.gpu:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # GPUを使うための処理
    # optimizer(パラメータ更新用)
    optimizer = chainer.optimizers.Adam()
    # optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)

    # モデルの読み込み npzはnumpy用
    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, model)

    train, test = chainer.datasets.get_mnist(withlabel=False)

   
#------------------イテレーターによるデータセットの設定-----------------------------------
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
#---------------------------------------------------------------
    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=model.get_loss_func())

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        eval_func=model.get_loss_func(k=1)))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='my_log_data'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/rec_loss', 'validation/main/rec_loss', 'elapsed_time']))
    # trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # トレーナーの実行
    trainer.run()
    #結合加重の表示させて、（上位何個かでもいい。）　んで、ロードの時の上位と比べてちゃんと保存復元されてるかみてみる。
    #https://qiita.com/mitmul/items/1e35fba085eb07a92560

    serializers.save_npz("sample.npz", model)



if __name__ == '__main__':
    main()
    
