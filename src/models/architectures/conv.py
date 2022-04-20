from __future__ import annotations
import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    """
    Currently hard-coded to two convolutional layers followed by two linear layers.
    Arch and default parameters are taken from [1].

    [1] https://github.com/pytorch/examples/blob/master/mnist/main.py.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_size: int,
        channels_1: int,
        channels_2: int,
        kernel: int,
        maxpool_kernel: int,
        linear_hidden: int,
        dropout_1: float,
        dropout_2: float,
    ):
        super().__init__()

        # Conversion from channels/kernels to linear dimension; See "Shape" section of [2] and [3]
        # [2] https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        # [3] https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        kernel_diminishment = 2 * (kernel - 1)
        dim_after_flatten = int(
            channels_2
            * (input_shape[1] - kernel_diminishment)
            * (input_shape[2] - kernel_diminishment)
            / maxpool_kernel**2
        )

        self.layers = nn.Sequential(
            # Conv 1
            nn.Conv2d(input_shape[0], channels_1, kernel),
            nn.ReLU(),
            # Conv 2
            nn.Conv2d(channels_1, channels_2, kernel),
            nn.ReLU(),
            # Ready for Linear
            nn.MaxPool2d(maxpool_kernel),
            nn.Dropout(dropout_1),
            nn.Flatten(1),
            # Hidden layer
            nn.Linear(dim_after_flatten, linear_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            # Output layer
            nn.Linear(linear_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
