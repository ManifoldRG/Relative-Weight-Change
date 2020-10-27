import numpy as np
import json
from comet_ml import Experiment
from types import SimpleNamespace
from main import run_experiment



def multi_train(configs_path="./configs.json"):

    configs = SimpleNamespace(**json.load(open(configs_path)))
    print(configs)

    average_rmae_dict, average_train_acc, average_test_acc =  {}, np.zeros(configs.epochs), np.zeros(configs.epochs)

    og_config_exp_name = configs.exp_name

    for idx, seed in enumerate(configs.seed_list):

        configs.exp_name = f"{seed}_{og_config_exp_name}"
        print(f"Training: {configs.exp_name}")
        
        configs.seed = seed
        rmae_dict, train_acc_arr, test_acc_arr = run_experiment(configs.epochs, configs.model_name, "untrained", configs)

        # np.save(f"{configs.save_directory}{configs.exp_name}_train_arr", train_acc_arr)
        # np.save(f"{configs.save_directory}{configs.exp_name}_test_arr", test_acc_arr)
        # save rmae dict
        
        with open(f"{configs.save_directory}{configs.exp_name}_rmae", 'w') as fp:
            json.dump(rmae_dict, fp)

        for layer in rmae_dict:
            if layer not in average_rmae_dict:
                average_rmae_dict[layer]  = np.array(rmae_dict[layer])
            else:
                average_rmae_dict[layer] += np.array(rmae_dict[layer])

        # average_train_acc += train_acc_arr
        # average_test_acc += test_acc_arr
        
    for layer in average_rmae_dict:
        average_rmae_dict[layer] /= 5

    # average_train_acc /= len(configs.seed_list)
    # average_test_acc /= len(configs.seed_list)
    
    print("Updating Average Results")

    experiment = Experiment(api_key="y8YtCd3TVO7TurC3t1D0LP7Ju",
                            project_name="avg-exp-rwc", workspace="ayushm-agrawal")
    experiment.set_name(f"avg_{og_config_exp_name}")

    for i in range(len(average_train_acc)):

        experiment.log_metric("Train Acc", average_train_acc[i], epoch=i+1)
        experiment.log_metric("Test Acc", average_test_acc[i], epoch=i+1)
    
    experiment.end()

    print("Done!")
    
if __name__ == "__main__":
    multi_train()