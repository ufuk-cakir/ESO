# import torch
import torch.nn as nn
import numpy as np


def calc_back_conv(input_size, conv_layer, dim):
    """Calculate the input size of a Conv2d layer

    Reverse calculation of the output size of a Conv2d layer. This is used to calculate the minimum input size of a CNN.

    Parameters
    ----------
    input_size : int
        The output size of the Conv2d layer
    conv_layer : torch.nn.Conv2d
        The Conv2d layer to calculate the input size of
    dim : int
        The dimension to calculate the input size of. 0 for height, 1 for width

    Returns
    -------
    int
        The input size of the Conv2d layer
    """
    kernel_size = conv_layer.kernel_size[dim]
    stride = conv_layer.stride[dim]
    padding = conv_layer.padding[dim]
    dilation = conv_layer.dilation[dim]

    #return ((input_size - 1) * stride) - 2 * padding + dilation * (kernel_size - 1) + 1
    #correction ? 
    return ((input_size - 1) * stride) - 2 * padding + kernel_size

def calc_back_pool(input_size, pool_layer, dim):
    """Calculate the input size of a MaxPool2d layer

    Reverse calculation of the output size of a MaxPool2d layer. This is used to calculate the minimum input size of a CNN.

    Parameters
    ----------
    input_size : int
        The output size of the MaxPool2d layer
    pool_layer : torch.nn.MaxPool2d
        The MaxPool2d layer to calculate the input size of
    dim : int
        The dimension to calculate the input size of. 0 for height, 1 for width

    Returns
    -------
    int
        The input size of the MaxPool2d layer
    """
    kernel_size = (
        pool_layer.kernel_size
        if isinstance(pool_layer.kernel_size, int)
        else pool_layer.kernel_size[dim]
    )
    stride=(pool_layer.stride
            if isinstance(pool_layer.kernel_size, int)
            else pool_layer.kernel_size[dim]
    )
    #return input_size * kernel_size
    #correction ?
    return ((input_size - 1) * stride) + kernel_size

def get_conv_output_dim(layer: nn.Module, input_dim: tuple) -> tuple:
    """Calculate output dimension of a CNN layer

    Parameters
    ----------
    layer : torch.nn.Module
        The CNN layer to calculate the output dimension of
    input_dim : tuple
        The input dimension of the CNN layer in the form of (n_channels, height, width)

    Returns
    -------
    tuple
        The output dimension of the CNN layer in the form of (n_channels, height, width)
    """
    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation

    input_channels, input_height, input_width = input_dim

    output_channels = layer.out_channels
    output_height = (
        input_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) / stride[0] + 1
    output_width = (
        input_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) / stride[1] + 1

    return (output_channels, int(output_height), int(output_width))


class BaseCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        conv_layers,
        conv_filters,
        dropout_rate,
        conv_kernel,
        max_pooling_size,
        fc_units,
        fc_layers,
        conv_padding = None, 
        stride_maxpool = None, 
    ):
        """Base CNN model for the classification of the images

        Parameters
        ----------
        input_shape : tuple
            The input shape of the images in the form of (n_channels, height, width)
        conv_layers : int
            The number of convolutional layers
        conv_filters : int
            The number of filters in the convolutional layers
        dropout_rate : float
            The dropout rate of the dropout layers
        conv_kernel : int
            The kernel size of the convolutional layers
        max_pooling_size : int
            The kernel size of the max pooling layers
        fc_units : int
            The number of units in the fully connected layers
        fc_layers : int
            The number of fully connected layers
        """
        super(BaseCNN, self).__init__()
        self.input_shape = input_shape
        n_channels = input_shape[0]
        self.n_conv_layers = conv_layers
        self.conv_filters = conv_filters
        self.dropout_rate = dropout_rate
        self.conv_kernel = conv_kernel
        self.max_pooling_size = max_pooling_size
        self.fc_units = fc_units
        self.n_fc_layers = fc_layers
        self.conv_padding=conv_padding
        if stride_maxpool is None : 
            self.stride_maxpool= max_pooling_size
        else : self.stride_maxpool= stride_maxpool

        if conv_padding is None : 
            self.conv_padding=0
        else : self.conv_padding = conv_padding

        # Convolutional layers
        self.conv_layers = nn.Sequential()
        self.conv_layers.add_module(
            "conv0",
            nn.Conv2d(n_channels, self.conv_filters, kernel_size=self.conv_kernel, padding=self.conv_padding),
        )
        self.conv_layers.add_module("relu0", nn.ReLU())
        self.conv_layers.add_module("dropout0", nn.Dropout(self.dropout_rate))
        self.conv_layers.add_module("maxpool0", nn.MaxPool2d(self.max_pooling_size, stride= self.stride_maxpool))

        for i in range(1, self.n_conv_layers):
            self.conv_layers.add_module(
                f"conv{i}",
                nn.Conv2d(
                    self.conv_filters, self.conv_filters, kernel_size=self.conv_kernel
                ),
            )
            self.conv_layers.add_module(f"relu{i}", nn.ReLU())
            self.conv_layers.add_module(f"dropout{i}", nn.Dropout(self.dropout_rate))
            self.conv_layers.add_module(
                f"maxpool{i}", nn.MaxPool2d(self.max_pooling_size)
            )

        # Fully connected layers
        self.fc_layers = nn.Sequential()
        # input_units = self.conv_filters * (128 // (self.max_pooling_size ** self.n_conv_layers)) * (76 // (self.max_pooling_size ** self.n_conv_layers))
        input_units = np.prod(self._calc_cnn_output_dim())
        for i in range(self.n_fc_layers):
            self.fc_layers.add_module(f"fc{i}", nn.Linear(input_units, self.fc_units))
            self.fc_layers.add_module(f"relu{i}", nn.ReLU())
            self.fc_layers.add_module(f"dropout{i}", nn.Dropout(self.dropout_rate))
            input_units = self.fc_units

        # Output layer
        self.output_layer = nn.Linear(self.fc_units, 2)
        self.softmax = nn.Softmax(dim=1)

    def _calc_cnn_output_dim(self) -> tuple:
        """Calculate output dimension of the CNN part of the network

        Parameters
        ----------
        None

        Returns
        -------
        tuple
        The output dimension of the CNN part of the network in the form of (n_channels, height, width)
        """
        output_dim = get_conv_output_dim(self.conv_layers[0], self.input_shape)
        for layer in self.conv_layers[1:]:
            # Check if layer is a convolutional layer
            if isinstance(layer, nn.Conv2d):
                output_dim = get_conv_output_dim(layer, output_dim)
            elif isinstance(layer, nn.MaxPool2d):
                output_dim = (
                    output_dim[0],
                    output_dim[1] // layer.kernel_size,
                    output_dim[2] // layer.kernel_size,
                )

        return output_dim

    def calculate_min_input_size(self):
        # Start with a size of 1 (minimum meaningful size)
        min_height, min_width = 1, 1
        # Convert the generator to a list and reverse iterate through the layers of the CNN
        for layer in reversed(list(self.modules())):
            if isinstance(layer, nn.Conv2d):
                min_height = calc_back_conv(min_height, layer, 0)  # for height
                min_width = calc_back_conv(min_width, layer, 1)  # for width
            elif isinstance(layer, nn.MaxPool2d):
                min_height = calc_back_pool(min_height, layer, 0)  # for height
                min_width = calc_back_pool(min_width, layer, 1)  # for width
        return int(min_height), int(min_width)

    def forward(self, x):
        """Forward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Shape should be (batch_size, n_channels, height, width)

        Returns
        -------
        torch.Tensor
            The output tensor. Shape should be (batch_size, n_classes). Outputs a probability for each class.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print("x size: ", x.size())
        x = self.fc_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x