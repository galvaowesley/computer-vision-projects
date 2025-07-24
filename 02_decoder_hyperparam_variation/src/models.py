import torch
import torch.nn.functional as F
from torch import nn


def conv_norm(in_channels, out_channels, kernel_size=3, act=True):

    layer = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                padding=kernel_size//2, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if act:
        layer += [nn.ReLU()]
    
    return nn.Sequential(*layer)

class DecoderBlock(nn.Module):
    """Recebe a ativação do nível anterior do decoder `x_dec` e a ativação do
    encoder `x_enc`. É assumido que `x_dec` possui uma resolução espacial
    menor que que `x_enc` e que `x_enc` possui número de canais diferente
    de `x_dec`.
    
    O módulo ajusta a resolução de `x_dec` para ser igual a `x_enc` e o número
    de canais de `x_enc` para ser igual a `x_dec`.
    """

    def __init__(self, enc_channels, dec_channels, extra_convs=False):
        super().__init__()
        self.channel_adjust = conv_norm(enc_channels, dec_channels, kernel_size=1,
                                        act=False)
        
        mix_layers = [conv_norm(dec_channels, dec_channels)]
        if extra_convs:
            mix_layers.append(conv_norm(dec_channels, dec_channels))
        
        self.mix = nn.Sequential(*mix_layers)

    def forward(self, x_enc, x_dec):
        x_dec_int = F.interpolate(x_dec, size=x_enc.shape[-2:], mode="nearest")
        x_enc_ad = self.channel_adjust(x_enc)
        y = x_dec_int + x_enc_ad
        return self.mix(y)

class Decoder(nn.Module):

    def __init__(self, encoder_channels_list, decoder_channels, extra_convs=False):
        super().__init__()

        # Inverte lista para facilitar interpretação
        encoder_channels_list = encoder_channels_list[::-1]

        self.middle = conv_norm(encoder_channels_list[0], decoder_channels)
        blocks = []
        for channels in encoder_channels_list[1:]:
            blocks.append(DecoderBlock(channels, decoder_channels, extra_convs=extra_convs))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):

        # Inverte lista para facilitar interpretação
        features = features[::-1]

        x = self.middle(features[0])
        for idx in range(1, len(features)):
            # Temos um bloco a menos do que nro de features, por isso
            # o idx-1
            x = self.blocks[idx-1](features[idx], x)

        return x

class EncoderDecoder(nn.Module):
    """Amostra ativações de um modelo ResNet do Pytorch e cria um decodificador."""

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

        re = self.resnet_encoder
        # Armazena se o modelo estava em modo treinamento
        training = re.training
        re.eval()

        # Corrigido: cria o tensor no mesmo device do encoder
        device = next(re.parameters()).device
        x = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            features = self.get_features(x)
        encoder_channels_list = [f.shape[1] for f in features]

        # Volta para treinamento
        if training:
            re.train()

        return encoder_channels_list
        
    def forward(self, x):
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

        # A camada de classificação poderia estar antes da interpolação, o que
        # reduziria o custo computacional mas possivelmente levaria a segmentações
        # menos detalhadas
        x = self.classification(x)

        return x
