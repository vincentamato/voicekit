import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodDiscriminator(nn.Module):
    """
    Discriminator that splits audio into time-series with a given period.
    """
    def __init__(self, period: int, kernel_size: int = 5, conv_channels: list = [32, 128, 512, 1024, 1024],
                 stride: list = [1, 2, 2, 2, 2], padding: int = 2):
        super().__init__()
        self.period = period
        layers = []
        in_ch = 1
        for ch, st in zip(conv_channels, stride):
            layers.append(nn.Conv2d(in_ch, ch, (kernel_size, 1), (st, 1), (padding, 0)))
            layers.append(nn.LeakyReLU(0.1))
            in_ch = ch
        layers.append(nn.Conv2d(in_ch, 1, (3, 1), (1, 1), (1, 0)))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, T)
        Returns:
            scores: list of feature maps for each conv layer
        """
        B, C, T = x.size()
        # pad to multiple of period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            T = T + pad_len
        # reshape: (B, 1, T/period, period)
        x = x.view(B, C, T // self.period, self.period)
        features = []
        for layer in self.model:
            x = layer(x)
            features.append(x)
        x = torch.flatten(x, 1, -1)
        features.append(x)
        return features

class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator comprising several PeriodDiscriminators.
    """
    def __init__(self, periods=[2,3,5,7,11]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
        )

    def forward(self, x: torch.Tensor):
        """
        Returns list of feature lists from each PeriodDiscriminator
        """
        return [d(x) for d in self.discriminators]

class ScaleDiscriminator(nn.Module):
    """
    Discriminator that operates on input at a given scale (optionally downsampled).
    """
    def __init__(self, scales: int = 1, kernel_size: int = 15, channels: list = [128, 128, 256, 512, 1024, 1024, 1024],
                 stride: list = [1, 2, 2, 4, 4, 1, 1], padding: list = [7, 6, 6, 18, 18, 0, 0]):
        super().__init__()
        layers = []
        in_ch = 1
        for ch, k, s, p in zip(channels, [kernel_size]+[3]*6, stride, padding):
            layers.append(nn.Conv1d(in_ch, ch, k, s, p))
            layers.append(nn.LeakyReLU(0.1))
            in_ch = ch
        layers.append(nn.Conv1d(in_ch, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)
        self.pooling = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x: torch.Tensor) -> list:
        """
        x: (B, 1, T)
        Returns feature maps from each layer
        """
        features = []
        for layer in self.model:
            x = layer(x)
            features.append(x)
        return features

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator: runs ScaleDiscriminator at multiple downsampling rates.
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator()
        ])
        self.pooling = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x: torch.Tensor):
        """
        Returns list of feature lists from each ScaleDiscriminator.
        """
        results = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling(x)
            results.append(d(x))
        return results

class HiFiGANDiscriminator(nn.Module):
    """
    Wrapper that returns both Multi-Period and Multi-Scale discriminator outputs.
    """
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, x: torch.Tensor):
        """
        x: (B, 1, T)
        Returns:
            mpd_outs: list of lists of feature maps
            msd_outs: list of lists of feature maps
        """
        return self.mpd(x), self.msd(x)