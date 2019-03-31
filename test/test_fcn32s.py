
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import skimage.data

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

def test_get_upsampling_weight():
    src = skimage.data.coffee()
    x = src.transpose(2, 0, 1)
    x = x[np.newaxis, :, :, :]
    x = torch.from_numpy(x).float()
    x = torch.autograd.Variable(x)

    in_channels = 3
    out_channels = 3
    kernel_size = 4

    m = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=2, bias=False)
    m.weight.data = get_upsampling_weight(
        in_channels, out_channels, kernel_size)

    y = m(x)

    y = y.data.numpy()
    y = y[0]
    y = y.transpose(1, 2, 0)
    dst = y.astype(np.uint8)

    assert abs(src.shape[0] * 2 - dst.shape[0]) <= 2
    assert abs(src.shape[1] * 2 - dst.shape[1]) <= 2

    return src, dst

def test():
    # With square kernels and equal stride
    m = nn.ConvTranspose2d(16, 33, 3, stride=2)
    # non-square kernels and unequal stride and with padding
    m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    input = torch.randn(20, 16, 50, 100)
    output = m(input)
    # exact output size can be also specified as an argument
    input = torch.randn(1, 16, 12, 12)
    downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
    upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
    h = downsample(input)
    h.size()
    torch.Size([1, 16, 6, 6])
    output = upsample(h, output_size=input.size())
    output.size()
    torch.Size([1, 16, 12, 12])

if __name__ == '__main__':
    test()
    # src, dst = test_get_upsampling_weight()
    # plt.subplot(121)
    # plt.imshow(src)
    # plt.title('x1: {}'.format(src.shape))
    # plt.subplot(122)
    # plt.imshow(dst)
    # plt.title('x2: {}'.format(dst.shape))
    # plt.show()
