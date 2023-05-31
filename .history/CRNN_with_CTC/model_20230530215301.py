import torch
import torch.nn as nn
from torch.nn import functional as F


class BidirectionalLSTM(nn.Module):
    # 构造一个双向LSTM需要提供给模型两个参数：input_size，hidden_size
    # 这里的embedding是一个线性层，把LSTM的输出hidden_size投射到真正需要的输出维度output_size上
    # 应该可以通过提供proj_size来直接让LSTM完成从hidden_size-->proj_size的映射，只不过输出的output_size = proj_size * (2 biLSTM? else 1)
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        # self.rnn 的输入形状：(seq_len, batch_size, input_size)
        # output，(h_n, c_n) = LSTM(input, (h_0, c_0)) 如果h_0, c_0留空，则默认均为全0
        # output 输出形状：(seq_len, batch_size, hidden_size * biLSTM)
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # 线性层，把rnn的输出从hidden_size*biLSTM映射到output_size
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()  # T-->seq_len, b-->batch_size, h-->hidden_size * biLSTM
        t_rec = recurrent.view(T * b, h) # 有点多余？直接接embedding然后return不好吗？

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    # 输入：
    # imgH: height of input image(想让img经过cnn后高度被压缩成1，其实这里的imgH只能输入高32的图片)
    # num_channel: 输入模型的图片的通道数
    # nclass: 待分类字符个数（等于所有字符数+1（CTC block））
    # 
    def __init__(self, imgH, num_channel, nclass, hidden_size, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = num_channel if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)  # kernel: 3，1，1 形状不变，channel从 3-->64
        # maxpool 2, 2 shape下采样一半（input_size/2）
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)  # kernel: 3，1，1 形状不变，channel从 64-->128
        # maxpool 2, 2 shape下采样一半（input_size/2）
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)  # kernel: 3，1，1 形状不变，channel从 128-->256
        convRelu(3)  # kernel: 3，1，1 形状不变，channel从 256-->256
        # maxpool 2, 2 shape下采样一半（input_size/2）
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2)))  # 256x4x16
        convRelu(4, True)  # kernel: 3，1，1 形状不变，channel从 256-->512
        convRelu(5)  # kernel: 3，1，1 形状不变，channel从 512-->512
        # maxpool (2, 2)的池化核，(2, 1)的步长，(0, 1)的padding，使特征图高度减半，但宽度+1(后面接一个2*2卷积，且不做padding，步长（1，1），刚好会让宽度-1，使最终输出保持在1*16)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        batch_size, channel, height, weight = conv.size()
        assert height == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [weight(seq_len), batch_size, channel(input_size)]
        # rnn features
        output = self.rnn(conv)
        # output.shape --> weight(seq_len), batch_size, num_class
        return output
