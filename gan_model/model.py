from torch import nn
from torch.nn.utils import spectral_norm

import utils


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transparent = args.transparent

        self.n_z = args.n_z
        self.n_gf = args.n_gf
        self.g_conv_type = args.g_conv_type
        if args.g_upscale_type == 0:
            self.g_upscale_type = 'nearest'
        else:  # 1
            self.g_upscale_type = 'bilinear'

        self.aspect_ratio = utils.check_aspect_ratio(*args.im_size)
        self.n_blocks = utils.determine_network_depth(*args.im_size)

        self.net = nn.Sequential()
        for idx in range(self.n_blocks, 0, -1):
            self.net.append(
                self._create_block(idx)
            )

    def _create_block(self, idx) -> nn.Sequential:
        # Setup torch.nn param
        in_channels = self.n_gf * 2 ** (idx - 1)
        out_channels = self.n_gf * 2 ** (idx - 2)
        kernel_size = (4, 4)
        if self.g_conv_type == 0:
            stride = 2
        else:  # 1
            stride = 1
        padding = 1

        if self.aspect_ratio == '4:3':
            upscale_size = (
                15 * 2 ** (self.n_blocks - idx - 2) + 1,
                20 * 2 ** (self.n_blocks - idx - 2) + 1,
            )
        else:
            upscale_size = 2 ** (self.n_blocks - idx + 2) + 1

        # Setup torch.nn param based on block depth
        if idx == self.n_blocks:  # block 0
            in_channels = self.n_z
            stride = 1
            if self.g_conv_type == 0:
                padding = 0
            if self.aspect_ratio == '4:3':
                kernel_size = (4, 5)
                if self.g_conv_type == 1:
                    kernel_size = (4, 4)
                    upscale_size = (6, 5)
        elif idx == self.n_blocks - 1 and self.aspect_ratio == '4:3' and self.g_conv_type == 1:  # block 1
            kernel_size = (4, 4)
            upscale_size = (9, 11)
        elif idx == self.n_blocks - 2 and self.aspect_ratio == '4:3':  # block 2
            kernel_size = (3, 4)  # ensure output shape is 20, 15 not 20, 16
            if self.g_conv_type == 1:
                kernel_size = (4, 4)
                upscale_size = (16, 21)
        elif idx == 1:  # last block
            in_channels = self.n_gf
            if self.transparent:
                out_channels = 4
            else:
                out_channels = 3

        # Create block
        block = nn.Sequential()

        # Conv
        if self.g_conv_type == 0:
            block.append(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size, stride, padding, bias=False)
            )
        else:
            block.append(
                nn.Upsample(size=upscale_size, mode=self.g_upscale_type)
            )
            block.append(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride, padding, bias=False)
            )

        # Norm/act
        if idx > 1:
            block.append(nn.InstanceNorm2d(out_channels))
            block.append(nn.LeakyReLU(0.2))
        else:
            block.append(nn.Tanh())

        return block

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transparent = args.transparent
        self.n_df = args.n_df
        self.d_norm_type = args.d_norm_type

        self.aspect_ratio = utils.check_aspect_ratio(*args.im_size)
        self.n_blocks = utils.determine_network_depth(*args.im_size)

        self.net = nn.Sequential()
        for idx in range(0, self.n_blocks):
            self.net.append(
                self._create_block(idx)
            )

    def _create_block(self, idx):
        # Setup torch.nn param
        in_channels = self.n_df * 2 ** (idx - 1)
        out_channels = self.n_df * 2 ** idx
        kernel_size = 4
        stride = 2
        padding = 1

        # Setup torch.nn param based on block depth
        if idx == 0:  # last block
            if self.transparent:
                in_channels = 4
            else:
                in_channels = 3
            out_channels = self.n_df
        elif (idx == self.n_blocks - 3 or idx == self.n_blocks - 2) and self.aspect_ratio == '4:3':  # 3rd/4th last block
            kernel_size = (3, 4)
        elif idx == self.n_blocks - 1:  # 2nd last block
            out_channels = 1
            stride = 1
            padding = 0
            if self.aspect_ratio == '4:3':
                kernel_size = (4, 5)

        # Create block
        block = nn.Sequential()

        # Conv
        Conv2d = nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding, bias=False)
        if self.d_norm_type == 1:
            Conv2d = spectral_norm(Conv2d)
        block.append(Conv2d)

        # Norm/act.
        if idx == 0:
            block.append(nn.LeakyReLU(0.2))
        elif idx < self.n_blocks - 1:
            if self.d_norm_type == 0:
                block.append(nn.InstanceNorm2d(out_channels))
            block.append(nn.LeakyReLU(0.2))
        else:
            block.append(nn.Sigmoid())

        return block

    def forward(self, x):
        y = self.net(x)
        # return y.view(-1, 1).squeeze(0)
        return y.view(-1)
