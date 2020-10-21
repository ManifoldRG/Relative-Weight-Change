import numpy as np


def mse(prev, curr):

    # subtract
    sub = np.subtract(prev, curr)
    # square
    squared = np.square(sub)

    # mean squared error
    return squared.mean()


def mae(prev, curr):

    # subtract
    sub = np.subtract(prev, curr)
    # absolute
    absolute = np.abs(sub)

    # mean absolute error
    return absolute.mean()


def rmae(prev, curr):

    # subtract
    sub = np.subtract(prev, curr)
    # absolute
    absolute = np.abs(sub)
    # normalization term
    normalizer = np.abs(prev)
    # relative error
    relative = absolute/normalizer

    # mean absolute error
    return relative.mean()


def setup_delta_tracking(model, model_name, training_type):

    if model_name == "Resnet18":
        if training_type == "pretrained":
            prev_list = load_layers_resnet(model, pretrained=True)
        elif training_type == "no_pretrain":
            prev_list = load_layers_resnet(model, pretrained=False)
    elif model_name == "VGG19":
        prev_list = load_layers_vgg(model)

    mse_delta_dict, mae_delta_dict, rmae_delta_dict = {}, {}, {}
    layer_names = []

    for name, param in model.named_parameters():
        if len(param.size()) > 1:
            layer_names.append(name)

    for i, layer in zip(range(len(prev_list)), layer_names):
        mse_delta_dict[layer] = []
        mae_delta_dict[layer] = []
        rmae_delta_dict[layer] = []

    return prev_list, mse_delta_dict, mae_delta_dict, rmae_delta_dict, layer_names


def compute_delta(model, model_name, layer_names, prev_list, mse_delta_dict, mae_delta_dict, rmae_delta_dict, training_type):

    if model_name == "Resnet18":
        if training_type == "pretrained":
            curr_list = load_layers_resnet(model, pretrained=True)
        elif training_type == "no_pretrain":
            curr_list = load_layers_resnet(model, pretrained=False)
    elif model_name == "VGG19":
        curr_list = load_layers_vgg(model)

    for i, layer in zip(range(len(prev_list)), layer_names):

        # compute L2
        layer_mse_delta = mse(prev_list[i], curr_list[i])
        # compute L1
        layer_mae_delta = mae(prev_list[i], curr_list[i])
        # compute RL1
        layer_rmae_delta = rmae(prev_list[i], curr_list[i])

        # update dictionaries
        mse_delta_dict[layer].append(layer_mse_delta)
        mae_delta_dict[layer].append(layer_mae_delta)
        rmae_delta_dict[layer].append(layer_rmae_delta)

    prev_list = curr_list.copy()

    return mse_delta_dict, mae_delta_dict, rmae_delta_dict, prev_list


def load_layers_resnet(model, pretrained=False):

    layer_list = []

    layer_list.append(model.conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer1[0].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer1[0].conv2.weight.data.cpu().numpy())
    layer_list.append(model.layer1[1].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer1[1].conv2.weight.data.cpu().numpy())
    layer_list.append(model.layer2[0].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer2[0].conv2.weight.data.cpu().numpy())

    if pretrained:
        layer_list.append(
            model.layer2[0].downsample[0].weight.data.cpu().numpy())
    else:
        layer_list.append(
            model.layer2[0].shortcut[0].weight.data.cpu().numpy())

    layer_list.append(model.layer2[1].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer2[1].conv2.weight.data.cpu().numpy())
    layer_list.append(model.layer3[0].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer3[0].conv2.weight.data.cpu().numpy())

    if pretrained:
        layer_list.append(
            model.layer3[0].downsample[0].weight.data.cpu().numpy())
    else:
        layer_list.append(
            model.layer3[0].shortcut[0].weight.data.cpu().numpy())

    layer_list.append(model.layer3[1].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer3[1].conv2.weight.data.cpu().numpy())
    layer_list.append(model.layer4[0].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer4[0].conv2.weight.data.cpu().numpy())

    if pretrained:
        layer_list.append(
            model.layer4[0].downsample[0].weight.data.cpu().numpy())
    else:
        layer_list.append(
            model.layer4[0].shortcut[0].weight.data.cpu().numpy())

    layer_list.append(model.layer4[1].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer4[1].conv2.weight.data.cpu().numpy())
    if pretrained:
        layer_list.append(model.fc.weight.data.cpu().numpy())
    else:
        layer_list.append(model.linear.weight.data.cpu().numpy())

    return layer_list


def load_layers_vgg(model):

    layer_list = []

    layer_list.append(model.features[0].weight.data.cpu().numpy())
    layer_list.append(model.features[2].weight.data.cpu().numpy())
    layer_list.append(model.features[5].weight.data.cpu().numpy())
    layer_list.append(model.features[7].weight.data.cpu().numpy())
    layer_list.append(model.features[10].weight.data.cpu().numpy())
    layer_list.append(model.features[12].weight.data.cpu().numpy())
    layer_list.append(model.features[14].weight.data.cpu().numpy())
    layer_list.append(model.features[16].weight.data.cpu().numpy())
    layer_list.append(model.features[19].weight.data.cpu().numpy())
    layer_list.append(model.features[21].weight.data.cpu().numpy())
    layer_list.append(model.features[23].weight.data.cpu().numpy())
    layer_list.append(model.features[25].weight.data.cpu().numpy())
    layer_list.append(model.features[28].weight.data.cpu().numpy())
    layer_list.append(model.features[30].weight.data.cpu().numpy())
    layer_list.append(model.features[32].weight.data.cpu().numpy())
    layer_list.append(model.features[34].weight.data.cpu().numpy())
    layer_list.append(model.classifier[0].weight.data.cpu().numpy())
    layer_list.append(model.classifier[3].weight.data.cpu().numpy())
    layer_list.append(model.classifier[6].weight.data.cpu().numpy())

    return layer_list
