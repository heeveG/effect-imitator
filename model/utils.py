import torch
import torch.nn as nn


def convolution_stack(dil, channels_in, channels_out, kernel_size):
    return nn.ModuleList(
        [nn.Conv1d(
            in_channels=(channels_in if i == 0 else channels_out),
            out_channels=channels_out,
            dilation=d,
            kernel_size=kernel_size,
        ) for i, d in enumerate(dil)
        ]
    )


def error_to_signal(y, y_pred):
    y = torch.cat((y[:, :, 0:1], y[:, :, 1:] - 0.95 * y[:, :, :-1]), dim=2)
    y_pred = torch.cat((y_pred[:, :, 0:1], y_pred[:, :, 1:] - 0.95 * y_pred[:, :, :-1]), dim=2)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)
