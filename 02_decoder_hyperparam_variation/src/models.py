import torch
import torch.nn.functional as F
from torch import nn


def conv_norm(in_channels, out_channels, kernel_size=3, act=True):

    """
    Create a convolutional layer followed by batch normalization and optional ReLU activation.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel.
        act (bool, optional): Whether to include ReLU activation.
    Returns:
        nn.Sequential: Sequential module with Conv2d, BatchNorm2d, and optional ReLU.
    """
    layer = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                padding=kernel_size//2, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if act:
        layer += [nn.ReLU()]
    return nn.Sequential(*layer)

class DecoderBlock(nn.Module):

    """
    Decoder block that combines encoder and decoder activations, adjusting resolution and channels.
    Args:
        enc_channels (int): Number of channels in encoder activation.
        dec_channels (int): Number of channels in decoder activation.
        extra_convs (bool, optional): Whether to add an extra conv-batchnorm-relu layer.
    """

    def __init__(self, enc_channels, dec_channels, extra_convs=False):
        super().__init__()
        self.channel_adjust = conv_norm(enc_channels, dec_channels, kernel_size=1, act=False)
        mix_layers = [conv_norm(dec_channels, dec_channels)]
        if extra_convs:
            mix_layers.append(conv_norm(dec_channels, dec_channels))
        self.mix = nn.Sequential(*mix_layers)

    def forward(self, x_enc, x_dec):
        """
        Forward pass for DecoderBlock.
        Args:
            x_enc (torch.Tensor): Encoder activation tensor.
            x_dec (torch.Tensor): Decoder activation tensor.
        Returns:
            torch.Tensor: Output tensor after combining and mixing.
        """
        x_dec_int = F.interpolate(x_dec, size=x_enc.shape[-2:], mode="nearest")
        x_enc_ad = self.channel_adjust(x_enc)
        y = x_dec_int + x_enc_ad
        return self.mix(y)

class Decoder(nn.Module):

    """
    Decoder module that reconstructs segmentation mask from encoder features.
    Args:
        encoder_channels_list (list): List of encoder feature channels.
        decoder_channels (int): Number of channels in decoder blocks.
        extra_convs (bool, optional): Whether to add extra conv layers in blocks.
    """

    def __init__(self, encoder_channels_list, decoder_channels, extra_convs=False):
        super().__init__()
        encoder_channels_list = encoder_channels_list[::-1]
        self.middle = conv_norm(encoder_channels_list[0], decoder_channels)
        blocks = []
        for channels in encoder_channels_list[1:]:
            blocks.append(DecoderBlock(channels, decoder_channels, extra_convs=extra_convs))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
        """
        Forward pass for Decoder.
        Args:
            features (list of torch.Tensor): List of encoder features.
        Returns:
            torch.Tensor: Output tensor after decoding.
        """
        features = features[::-1]
        x = self.middle(features[0])
        for idx in range(1, len(features)):
            x = self.blocks[idx-1](features[idx], x)
        return x

class EncoderDecoder(nn.Module):

    """
    EncoderDecoder architecture that samples activations from a ResNet encoder and reconstructs segmentation masks.
    Args:
        resnet_encoder (nn.Module): Pretrained ResNet encoder.
        decoder_channels (int): Number of channels in decoder blocks.
        num_classes (int): Number of output classes.
        use_strides (list, optional): List of strides to use for features.
        extra_convs (bool, optional): Whether to add extra conv layers in decoder blocks.
    """

    def __init__(self, resnet_encoder, decoder_channels, num_classes, 
                 use_strides=[2, 4, 8, 16, 32], extra_convs=False):
        super().__init__()
        self.resnet_encoder = resnet_encoder
        self.use_strides = use_strides
        all_strides = [2, 4, 8, 16, 32]
        all_channels = self.get_channels()
        selected_channels = [ch for ch, s in zip(all_channels, all_strides) if s in use_strides]
        self.decoder = Decoder(selected_channels, decoder_channels, extra_convs=extra_convs)
        self.classification = nn.Conv2d(decoder_channels, num_classes, 3, padding=1)

    def get_features(self, x):
        """
        Extract intermediate activations from the ResNet encoder.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            list: List of feature tensors at different resolutions (strides).
        """
        features = []
        re = self.resnet_encoder
        x = re.conv1(x)
        x = re.bn1(x)
        x = re.relu(x)
        features.append(x)
        x = re.maxpool(x)
        x = re.layer1(x)
        features.append(x)
        x = re.layer2(x)
        features.append(x)
        x = re.layer3(x)
        features.append(x)
        x = re.layer4(x)
        features.append(x)
        return features

    def get_channels(self):
        """
        Get the number of channels for each encoder feature.
        Returns:
            list: List of channel counts for encoder features.
        """
        re = self.resnet_encoder
        training = re.training
        re.eval()
        device = next(re.parameters()).device
        x = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            features = self.get_features(x)
        encoder_channels_list = [f.shape[1] for f in features]
        if training:
            re.train()
        return encoder_channels_list

    def forward(self, x):
        """
        Forward pass for EncoderDecoder.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Segmentation mask output tensor.
        """
        in_shape = x.shape[-2:]
        all_features = self.get_features(x)
        all_strides = [2, 4, 8, 16, 32]
        features = [
            feat for feat, stride in zip(all_features, all_strides) 
            if stride in self.use_strides
        ]
        x = self.decoder(features)
        if x.shape[-2:]!=in_shape:
            x = F.interpolate(x, size=in_shape, mode="nearest")
        x = self.classification(x)
        return x
