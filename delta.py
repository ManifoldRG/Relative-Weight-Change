from comet_ml import experiment
import numpy as np



def rmae(prev, curr):
    """
    Computes Reltaive L1 norm.
    """

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


def setup_delta_tracking(model):
    """
    This sets up tracking for deltas.
    """

    # load model layers.
    prev_list, layer_names = load_layers(model)

    # setup tracking dictionaries
    rmae_delta_dict = {}

    for layer in layer_names:
        rmae_delta_dict[layer] = []

    return prev_list, rmae_delta_dict


def compute_delta(model, prev_list, rmae_delta_dict, experiment, epoch):
    """
    This computes the deltas for the experiment.
    """

    # load model layers.
    curr_list, layer_names = load_layers(model)

    # update dictionaries with deltas.\
    print(
        f"At epoch: {epoch}... Logging RMAE in COMET...")
    for i, layer in zip(range(len(prev_list)), layer_names):

        # compute RL1.
        layer_rmae_delta = rmae(prev_list[i], curr_list[i])

        # update dictionaries.
        rmae_delta_dict[layer].append(layer_rmae_delta)

        # log the weight change
        experiment.log_metric(str(layer), layer_rmae_delta, epoch=epoch)

    # update previous.
    prev_list = curr_list.copy()

    return rmae_delta_dict, prev_list


def load_layers(model):
    layer_names, param_list = [], []

    for name, param in model.named_parameters():
        if len(param.size()) > 1:
            layer_names.append(name)
            param_list.append(param.clone().data.cpu().numpy())

    return param_list, layer_names