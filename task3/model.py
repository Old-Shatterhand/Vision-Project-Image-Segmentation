import torch
import torch.nn as nn


class RecCNN(nn.Module):
    def __init__(self, num_filter, t=2, kernel_size=3, stride=1, padding=1, bias=False):
        super(RecCNN, self).__init__()

        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv.forward(x)
        for i in range(self.t):
            x1 = self.conv.forward(x + x1)
        return x1


class RRCNN(nn.Module):
    def __init__(self, num_filter_in, num_filter_out, t, kernel_size=3, stride=1, padding=1, bias=False):
        super(RRCNN, self).__init__()

        self.pre_conv = nn.Conv2d(num_filter_in, num_filter_out, 1, padding=0, bias=bias)
        self.rec_cnn = nn.Sequential(
            RecCNN(num_filter_out, t, kernel_size, stride, padding, bias),
            # RecCNN(num_filter_out, t, kernel_size, stride, padding, bias)
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x1 = self.rec_cnn(x)
        return x + x1


class UpConv(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, kernel_size=3, stride=1, padding=1, bias=False):
        super(UpConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_filters_in, num_filters_out, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(num_filters_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv.forward(x)


class R2UNet(nn.Module):
    def __init__(self, num_classes, weights=None, t=2):
        super(R2UNet, self).__init__()

        self.d_rrcnn1 = RRCNN(3, 16, t)
        self.d_rrcnn2 = RRCNN(16, 32, t)
        self.d_rrcnn3 = RRCNN(32, 64, t)
        self.d_rrcnn4 = RRCNN(64, 128, t)
        self.d_rrcnn5 = RRCNN(128, 256, t)

        self.up_conv4 = UpConv(256, 128)
        self.u_rrcnn4 = RRCNN(256, 128, t)
        self.up_conv3 = UpConv(128, 64)
        self.u_rrcnn3 = RRCNN(128, 64, t)
        self.up_conv2 = UpConv(64, 32)
        self.u_rrcnn2 = RRCNN(64, 32, t)
        self.up_conv1 = UpConv(32, 16)
        self.u_rrcnn1 = RRCNN(32, 16, t)

        self.o_conv = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

        if weights is not None:
            self.load(weights)

    def forward(self, x):
        # dimensionality comments are given as width x height x channels,
        # nevertheless, the input should be channels first!!!

        # x = 128x256x3
        x1 = self.d_rrcnn1(x)
        x = nn.MaxPool2d(2)(x1)

        # x = 64x128x16
        x2 = self.d_rrcnn2(x)
        x = nn.MaxPool2d(2)(x2)

        # x = 32x64x32
        x3 = self.d_rrcnn3(x)
        x = nn.MaxPool2d(2)(x3)

        # x = 16x32x64
        x4 = self.d_rrcnn4(x)
        x = nn.MaxPool2d(2)(x4)

        # x = 8x16x128
        x = self.d_rrcnn5(x)

        # x = 8x16x256
        x = self.up_conv4(x)
        x = torch.cat((x, x4), dim=1)
        x = self.u_rrcnn4(x)

        # x = 16x32x128
        x = self.up_conv3(x)
        x = torch.cat((x, x3), dim=1)
        x = self.u_rrcnn3(x)

        # x = 32x64x64
        x = self.up_conv2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.u_rrcnn2(x)

        # x = 64x128x32
        x = self.up_conv1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.u_rrcnn1(x)

        # x = 128x256x15
        x = self.o_conv(x)

        return x

    def classify(self, x):
        return self.forward(x).argmax(dim=1)

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))
