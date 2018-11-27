import tensorflow as tf
import os
import numpy as np
import glob
from skimage import io, transform


# 读取图片
def read_img(path,image_W,image_H):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    images = []
    imagelable = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (image_W, image_H, 3))
            images.append(img)
            imagelable.append(idx)
            print('reading the idx:%s' % (idx))
    return np.asarray(images, np.float32), np.asarray(imagelable, np.int32)

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)#assert断言机制，如果后面的表达式为真，则直接抛出异常。在这里的意思,大概就是:样本和标签数量要对上
    if shuffle:
        # 生成一个np.arange可迭代长度是len(训练数据),也就是训练数据第一维数据的数量(就是训练数据的数量，训练图片的数量)
        indices = np.arange(len(inputs))
        # np.random.shuffle打乱arange中的顺序，使其随机循序化，如果是数组，只打乱第一维
        np.random.shuffle(indices)
    # 这个range(初始值为0，终止值为[训练图片数-每批训练图片数+1]，步长是[每批训练图片数])：例(0[起始值],80[训练图片数]-20[每批训练图片数],20[每批训练图片数]),也就是(0,60,20)当循环到60时,会加20到达80的训练样本
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            # 如果shuffle为真,将indices列表,切片(一批)赋值给excerpt
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
        # yield常见用法：该关键字用于函数中会把函数包装为generator。然后可以对该generator进行迭代: for x in fun(param).
        # 按照我的理解，可以把yield的功效理解为暂停和播放。
        # 在一个函数中，程序执行到yield语句的时候，程序暂停，返回yield后面表达式的值，在下一次调用的时候，从yield语句暂停的地方继续执行，如此循环，直到函数执行完。
        # 此处,就是返回每次循环中 从inputs和targets列表中,截取的 经过上面slice()切片函数定义过的 数据.
        # (最后的shuffle变量，决定了样本是否随机化)

