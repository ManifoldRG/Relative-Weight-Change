import torch
import timm


def load_efficientnet(model_name, num_classes=1000, input_channels=3, pretrained=False, ):

    model = timm.create_model(
        model_name,
        num_classes = num_classes,
        in_chans = input_channels,
        pretrained = pretrained)


    return model