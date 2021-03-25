import torch
import torch.nn as nn


"""
DISCLAIMER:
The model architectures in this file are inspired by https://github.com/LeeJunHyun/Image_Segmentation. You will find 
some similarities between the model architectures. This is majorly, because we both implemented the same papers and use
PyTorch. Since the paper are quite explicit with their networks, there are not that much possibilities for changes.

But other then Mr. Lee, we only use a AttentionR2UNet architecture with 4 layers and 3 attention blocks, similar to the 
provided R2UNet-paper.
Furthermore, you can see, that we changed the style of the code a bit as we wrote the code entirely on our own using our
own coding style and did not copy from the repo.
"""


class RecCNN(nn.Module):
    def __init__(self, num_filter, t=2, kernel_size=3, stride=1, padding=1, bias=False):
        """
        Recurrent convolutional neural network. The recurrence is done by passing the network input t-times through the
        same convolutional layer.
        :param num_filter: number of filters to use in the layer. The number of filters is the same for input and output.
        :param t: number of times to pass the input through the network
        :param kernel_size: size of the kernel to use
        :param stride: stride of the convolutional layer to use
        :param padding: padding of the convolutional layer
        :param bias: flag indicating the use of an additional bias for the layers
        """
        super(RecCNN, self).__init__()

        self.t = t

        # define the structure of the layer
        self.conv = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward an input through the network
        :param x: input tensor, should be 4 dimensional
        :return: 4-dimensional output-tensor
        """
        x1 = self.conv.forward(x)
        for i in range(self.t):
            x1 = self.conv.forward(x + x1)
        return x1


class RRCNN(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, t, kernel_size=3, stride=1, padding=1, bias=False):
        """
        Residual recurrent convolutional neural network. This puts together residual networks and
        :param num_filters_in: number of input filters
        :param num_filters_out: number of output filters
        :param t: number of times to pass the input through the recurrent parts of the network
        :param kernel_size: size of the kernel to use
        :param stride: stride of the convolutional layer to use
        :param padding: padding of the convolutional layer
        :param bias: flag indicating the use of an additional bias for the layers
        """
        super(RRCNN, self).__init__()

        # define the structure of the layer
        self.pre_conv = nn.Conv2d(num_filters_in, num_filters_out, 1, padding=0, bias=bias)
        self.rec_cnn = RecCNN(num_filters_out, t, kernel_size, stride, padding, bias)

    def forward(self, x):
        """
        Forward an input trough the layer. The residuality is created by adding up the outputs of two convolutional
        layers into one output
        :param x: input tensor, should be 4-dimensional
        :return: 4-dimensional output-tensor
        """
        x = self.pre_conv(x)
        x1 = self.rec_cnn(x)
        return x + x1


class UpConv(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, kernel_size=3, stride=1, padding=1, bias=False):
        """
        Inverse convolutional neural network (de- or up-convolutional). This reduces the number of filters and doubles
        the number of pixels in each direction by the factor 2 using upsampling, in a way, inverse max-pooling
        :param num_filters_in: number of input filters
        :param num_filters_out: number of output filters
        :param kernel_size: size of the kernel to use
        :param stride: stride of the convolutional layer to use
        :param padding: padding of the convolutional layer
        :param bias: flag indicating the use of an additional bias for the layers
        """
        super(UpConv, self).__init__()

        # define the structure of the layer
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_filters_in, num_filters_out, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(num_filters_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward an input through the layer.
        :param x: input tensor, should be 4-dimensional
        :return: 4-dimensional output-tensor
        """
        return self.conv.forward(x)


class AttentionCNN(nn.Module):
    def __init__(self, num_filters_in, num_filters_intern):
        """
        Attention convolution neural network. This uses the idea of attention in neural networks to increase the
        performance of the network.
        :param num_filters_in: number of filters of the input tensors
        :param num_filters_intern: number of filters of the output tensor
        """
        super(AttentionCNN, self).__init__()

        # define the structure of the network
        # define the adapter for the input from the encoding part of the final network
        self.from_encoding = nn.Sequential(
            nn.Conv2d(num_filters_in, num_filters_intern, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_filters_intern)
        )

        # define the adapter for the input from the decoding part of the final network
        self.from_decoding = nn.Sequential(
            nn.Conv2d(num_filters_in, num_filters_intern, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_filters_intern)
        )

        # define the structure to merge the two embeddings from encoding and decoding
        self.combine = nn.Sequential(
            nn.Conv2d(num_filters_intern, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, encoded, decoded):
        """
        Forward two inputs through the network
        :param encoded: Input from the encoding part of the network, should be 4-dimensional
        :param decoded: Input from the decoding part of the network, should be 4-dimensional
        :return: 4-dimensional output-tensor
        """
        tmp_encoded = self.from_encoding.forward(encoded)
        tmp_decoded = self.from_decoding.forward(decoded)
        combination = nn.ReLU(inplace=True)(tmp_encoded + tmp_decoded)
        combination = self.combine(combination)
        return decoded * combination


class R2UNet(nn.Module):
    def __init__(self, num_classes, weights=None, t=2):
        """
        Recurrent residual convolutional neural network of depth 4.
        :param num_classes: number of output classes to distinguish
        :param weights: file with precomputed weights
        :param t: number of loops in the recurrent layers
        """
        super(R2UNet, self).__init__()

        # feature extraction
        self.d_rrcnn1 = RRCNN(3, 16, t)
        self.d_rrcnn2 = RRCNN(16, 32, t)
        self.d_rrcnn3 = RRCNN(32, 64, t)
        self.d_rrcnn4 = RRCNN(64, 128, t)
        self.d_rrcnn5 = RRCNN(128, 256, t)

        # feature mapping onto the original image size
        self.up_conv4 = UpConv(256, 128)
        self.u_rrcnn4 = RRCNN(256, 128, t)
        self.up_conv3 = UpConv(128, 64)
        self.u_rrcnn3 = RRCNN(128, 64, t)
        self.up_conv2 = UpConv(64, 32)
        self.u_rrcnn2 = RRCNN(64, 32, t)
        self.up_conv1 = UpConv(32, 16)
        self.u_rrcnn1 = RRCNN(32, 16, t)

        self.o_conv = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

        # if provided, load precomputed weights
        if weights is not None:
            self.load(weights)

    def forward(self, x):
        """
        Forward a tensor through the network. The tensor should have the format:
        batch_size x num_channels x width x height
        :param x: input tensor to the network, should be 4-dimensional
        :return: prediction of the network as a 4-dimensional tensor
        """
        # The dimensionality comments are given as width x height x channels
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

        # x = 128x256x16
        x = self.o_conv(x)

        return x

    def classify(self, x):
        """
        Use the network for classification by applying pixelwise argmax to the output of forward
        :param x: input-tensor to classify, should be 4-dimensional
        :return: classifications for each pixel, given by their class index
        """
        return self.forward(x).argmax(dim=1)

    def save(self, file_name):
        """
        Save the state of the network into the given file
        :param file_name: path to the file to store the network weights in
        """
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        """
        Load the state of the network from the given file
        :param file_name: path to the file that stored the weights
        """
        self.load_state_dict(torch.load(file_name))


class AttR2UNet(nn.Module):
    def __init__(self, num_classes, weights=None, t=2):
        """
        Attention recurrent residual convolutional neural network of depth 4.
        :param num_classes: number of output classes to distinguish
        :param weights: file with precomputed weights
        :param t: number of loops in the recurrent layers
        """
        super(AttR2UNet, self).__init__()

        # define the feature extraction
        self.d_rrcnn1 = RRCNN(3, 16, t)
        self.d_rrcnn2 = RRCNN(16, 32, t)
        self.d_rrcnn3 = RRCNN(32, 64, t)
        self.d_rrcnn4 = RRCNN(64, 128, t)
        self.d_rrcnn5 = RRCNN(128, 256, t)

        # define the feature mapping
        self.up_conv4 = UpConv(256, 128)
        self.att_cnn4 = AttentionCNN(128, 64)
        self.u_rrcnn4 = RRCNN(256, 128, t)
        self.up_conv3 = UpConv(128, 64)
        self.att_cnn3 = AttentionCNN(64, 32)
        self.u_rrcnn3 = RRCNN(128, 64, t)
        self.up_conv2 = UpConv(64, 32)
        self.att_cnn2 = AttentionCNN(32, 16)
        self.u_rrcnn2 = RRCNN(64, 32, t)
        self.up_conv1 = UpConv(32, 16)
        self.u_rrcnn1 = RRCNN(32, 16, t)

        self.o_conv = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

        if weights is not None:
            self.load(weights)

    def forward(self, x):
        """
        Forward a tensor through the network. The tensor should have the format:
        batch_size x num_channels x width x height
        :param x: input tensor to the network, should be 4-dimensional
        :return: prediction of the network as a 4-dimensional tensor
        """
        # The dimensionality comments are given as width x height x channels
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
        x5 = self.d_rrcnn5(x)

        # x = 8x16x256
        x = self.up_conv4(x5)
        x4 = self.att_cnn4(x, x4)
        x = torch.cat((x, x4), dim=1)
        x = self.u_rrcnn4(x)

        # x = 16x32x128
        x = self.up_conv3(x)
        x3 = self.att_cnn3(x, x3)
        x = torch.cat((x, x3), dim=1)
        x = self.u_rrcnn3(x)

        # x = 32x64x64
        x = self.up_conv2(x)
        x2 = self.att_cnn2(x, x2)
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
        """
        Use the network for classification by applying pixelwise argmax to the output of forward
        :param x: input-tensor to classify, should be 4-dimensional
        :return: classifications for each pixel, given by their class index
        """
        return self.forward(x).argmax(dim=1)

    def save(self, file_name):
        """
        Save the state of the network into the given file
        :param file_name: path to the file to store the network weights in
        """
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        """
        Load the state of the network from the given file
        :param file_name: path to the file that stored the weights
        """
        self.load_state_dict(torch.load(file_name))
