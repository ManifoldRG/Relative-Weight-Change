import json
import os
from os.path import join
from types import SimpleNamespace

import numpy as np
from comet_ml import Experiment

from main import run_experiment


def multi_train(configs_path="./configs.json"):

    # configs_path = "/home/shared/Layer-Wise-Learning-Trends-PyTorch/configs.json"
    print(f"Configs Path: {configs_path}")
    configs = SimpleNamespace(**json.load(open(configs_path)))
    print(configs)

    experiment = Experiment(
        api_key="ZgD8zJEiZErhwIzPMfZpitMjq",
        project_name="imagenet-rwc",
        workspace="ayushm-agrawal",
    )
    experiment.log_parameters(configs)

    average_rmae_dict, average_train_acc, average_test_acc = {
    }, np.zeros(configs.epochs), np.zeros(configs.epochs)

    og_config_exp_name = configs.exp_name

    for _, seed in enumerate(configs.seed_list):

        configs.exp_name = f"{seed}_{og_config_exp_name}"
        print(f"Training: {configs.exp_name}")

        configs.seed = seed
        configs.experiment = experiment
        rmae_dict, train_acc_arr, test_acc_arr = run_experiment(
            configs.epochs, configs.model_name, "untrained", configs)

        for layer in rmae_dict:
            if layer not in average_rmae_dict:
                average_rmae_dict[layer] = np.array(rmae_dict[layer])
            else:
                average_rmae_dict[layer] += np.array(rmae_dict[layer])

        average_train_acc += train_acc_arr
        average_test_acc += test_acc_arr

    for layer in average_rmae_dict:
        average_rmae_dict[layer] /= len(configs.seed_list)

    average_train_acc /= len(configs.seed_list)
    average_test_acc /= len(configs.seed_list)

    save_path = join(configs.arr_save_path, og_config_exp_name)
    print(f"Saving Arrays at {save_path}")

    np.save(save_path + "avg_train_acc.npy", average_train_acc)
    np.save(save_path + "avg_test_acc.npy", average_test_acc)
    np.save(save_path + "avg_rmae_dict.npy", average_rmae_dict)

    print("Done!")

    # PLOT THE GRAPHS USING THE AVERAGE ARRAYS


if __name__ == "__main__":
    multi_train()
