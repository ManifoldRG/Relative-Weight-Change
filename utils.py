import itertools
import matplotlib.pyplot as plt


def plot_deltas(delta_dict, model_name, save_name):
    marker = itertools.cycle(('.', '$...$', "v", "1", "8", "s", "p",
                              "P", "*", "+", "x", "d", "_", 4, 5, "D", "H", "3", "<", ">", "^"))
    plt.figure(figsize=(30, 20), dpi=200)
    # style
    plt.style.use('seaborn-darkgrid')

    for layer in delta_dict:

        if model_name == "Resnet18":
            if layer != 'conv1.weight' and layer != 'linear.weight' and layer != 'layer3.0.shortcut.0.weight' and layer != 'layer2.0.shortcut.0.weight':
                plt.plot(delta_dict[layer], label=layer, marker=next(
                    marker), animated=True, linewidth=0.5)

    plt.legend(loc=1, prop={'size': 8})
    plt.savefig(save_name)
