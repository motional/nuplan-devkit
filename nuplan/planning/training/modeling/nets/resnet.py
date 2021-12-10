import torch
import torch.nn as nn


class ResNet1D(nn.Module):
    """
    1D ResNet unit which performs convolution, batch norm and applies relu with skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        depth: int = 2,
        b_norm: bool = False,
        activation: bool = True,
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for 1D convolution
        :param depth: Number of conv layers
        :param b_norm: If batchnorm should be used
        :param activation: If activation should be applied to final output
        """
        super(ResNet1D, self).__init__()
        self.act = activation

        # We will keep the same size so padding with half the kernel size
        padding = (kernel_size - 1) // 2

        # Main stream
        stream = []
        in_ch = in_channels
        for idx in range(depth):
            stream += [
                nn.Conv1d(
                    in_ch,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            ]
            if b_norm:
                stream += [nn.BatchNorm1d(out_channels)]
            if idx is not depth - 1:
                stream += [nn.ReLU(inplace=True)]
            in_ch = out_channels

        # remove last relu
        self.main_net = nn.Sequential(*stream)

        # Downsample net
        downsample = []
        if stride != 1 or out_channels != in_channels:
            downsample += [
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                )
            ]
            if b_norm:
                downsample += [nn.BatchNorm1d(out_channels)]
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = None

        # Layer for final output
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass
        :param x: Input tensor
        :return: Output from model
        """
        output = self.main_net(x)

        # If we need to pass input through some net to downsample spatially
        if self.downsample:
            x = self.downsample(x)

        output += x

        if self.act:
            output = self.relu(output)

        return output
