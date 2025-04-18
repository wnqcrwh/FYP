import torch.nn as nn

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(num_parameters=out_planes, init=0.25)
        ]
        super(ConvBNPReLU, self).__init__(*layers)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.prelu1 = nn.PReLU(num_parameters=in_planes, init=0.25)
        self.prelu2 = nn.PReLU(num_parameters=out_planes, init=0.25)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        return x

class GDConv(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size, padding, bias=False):
        super(GDConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=in_planes,
                                   bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x
    
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNPReLU(in_planes, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNPReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# This is the main class for MobileFaceNet
# It initializes the weights of the model
# It uses the ConvBNPReLU, DepthwiseSeparableConv, GDConv and InvertedResidual classes to build the model
# It also uses the _make_divisible function to ensure that the number of channelsin each layer is divisible by 8
class MobileFaceNet(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, input_size=128):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileFaceNet, self).__init__()
        block = InvertedResidual
        input_channel = 64
        last_channel = 512

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        # input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv1 = ConvBNPReLU(3, input_channel, stride=2)
        self.dw_conv = DepthwiseSeparableConv(in_planes=64, out_planes=64, kernel_size=3, padding=1)
        features = list()
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel   
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
         # building last several layers
        self.conv2 = ConvBNPReLU(input_channel, self.last_channel, kernel_size=1)
        kernel_size = input_size // 16
        self.gdconv = GDConv(in_planes=512, out_planes=512, kernel_size=kernel_size, padding=0)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.conv1(x)
        x = self.dw_conv(x)

        x = self.features(x)

        x = self.conv2(x)
        x = self.gdconv(x)
        x = self.conv3(x)
        x = self.bn(x)

        x = x.view(B, T, -1)
        return x
