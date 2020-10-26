import numpy as np
import json
from comet_ml import Experiment
from types import SimpleNamespace
from main import run_experiment



def multi_train(configs_path="./configs.json"):

    configs = SimpleNamespace(**json.load(open(configs_path)))
    print(configs)

    average_train_acc, average_test_acc = np.zeros(configs.epochs), np.zeros(configs.epochs)

    average_exp_name = f"avg_{configs.exp_name}"

    for seed in configs.seed_list:

        configs.exp_name = f"{seed}_{configs.exp_name}"
        print(f"Training: {configs.exp_name}")
        
        configs.seed = seed
        _, train_acc_arr, test_acc_arr = run_experiment(configs.epochs, configs.model_name, "no_pretrain", configs)

        np.save(f"{configs.save_directory}{configs.exp_name}_train_arr", train_acc_arr)
        np.save(f"{configs.save_directory}{configs.exp_name}_test_arr", test_acc_arr)

        average_train_acc += train_acc_arr
        average_test_acc += test_acc_arr
    
    average_train_acc /= len(configs.seed_list)
    average_test_acc /= len(configs.seed_list)
    
    print("Updating Average Results")

    experiment = Experiment(api_key="y8YtCd3TVO7TurC3t1D0LP7Ju",
                            project_name="avg-exp-rwc", workspace="ayushm-agrawal")
    experiment.set_name(average_exp_name)

    for i range(len(average_train_acc)):

        experiment.log_metric("Train Acc", average_train_acc[i], epoch=i+1)
        experiment.log_metric("Test Acc", average_test_acc[i], epoch=i+1)
    
    experiment.end()

    print("Done!")
    
if __name__ == "__main__":
    multi_train()