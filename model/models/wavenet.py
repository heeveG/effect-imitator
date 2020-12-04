import torch
import torch.nn as nn

from ..utils import convolution_stack


class WNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super(WNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        self.convs_sigm = convolution_stack(dilations, 1, num_channels, kernel_size)
        self.convs_tanh = convolution_stack(dilations, 1, num_channels, kernel_size)
        self.residuals = convolution_stack(dilations, num_channels, num_channels, 1)

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        out = x
        skips = []

        for conv_sigm, conv_tanh, residual in zip(
                self.convs_sigm, self.convs_tanh, self.residuals
        ):
            x = out
            out_sigm, out_tanh = conv_sigm(x), conv_tanh(x)
            out = torch.tanh(out_tanh) * torch.sigmoid(out_sigm)
            skips.append(out)
            out = residual(out)
            out = out + x[:, :, -out.size(2):]

        out = torch.cat([s[:, :, -out.size(2):] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out
