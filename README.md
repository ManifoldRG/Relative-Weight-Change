# Relative Weight Change: Understanding how Neural Networks Learn
[![Made With python 3.7](https://img.shields.io/badge/Made%20with-Python%203.7-brightgreen)]() [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()

This repository contains the **original PyTorch** implementation of the paper [Investigating Learning in Deep Neural Networks using Layer-Wise Weight Change](https://github.com/Manifold-Computing/Layer-Wise-Learning-Trends/blob/main/Investigating%20Learning%20in%20Deep%20Neural%20Networks%20using%20Layer-Wise%20Weight%20Change.pdf) by [Ayush Manish Agrawal](https://github.com/ayushm-agrawal), [Atharva Tendle](https://github.com/atharva-tendle), [Harsh Sikka](https://github.com/harshsikka123), [Sahib Singh](https://github.com/sahibsin) and [Amr Kayid](https://github.com/AmrMKayid).

## Requirements
```
pip install -r requirements.txt
```

## How to run the code ?
### Using dataset/architectures provided in the repository

```
python3 multi_train.py
```
#### Update the configs.json file to adjust the hyperparameters:

- `epochs` : Number of epochs to train the model for
  - Default : `150` 
- `seed_list` : This initializes different weights for each experiment. We average over the results to determine a general learning trend
  - Options : `Length of list defines the number of different experiments for a given arch and dataset`
  - Default : `[0, 42, 123, 1000, 1234]`
- `model_name` : The architecture to use. Architectures included in the code are 
  - Options : `AlexNet, VGG19, ResNet18, Xception`
  - Default : `AlexNet`
- `dataset` : Dataset to use for training. Datasets included in the code are 
  - Options : `CIFAR-10, CIFAR-100, FMNIST, MNIST`
  - Default : `CIFAR-10`
- `lr` : The learning rate to use for the experiments. We have not used Adaptive learning rate for the simplicity in interpreting the trends
  - Default : `0.001`
- `momentum` : Used for the optimizer
  - Default : `0.9`
- `weight_decay` : Used for the optimizer
  - Default : `1e-4`
- `batch_size` : The number of images per batch in the training dataset
  - Default : `128`
- `target_val_acc` : Used for early stopping\
  - Default : `94%`
        
### Using datasets/architectures that are not included with this repository:
- Adding a new architecture :
        - Add a new file `new_model.py` for your `new_model` by going to `/models/`
        - **IMPORTANT** Make sure that your model class has *input_channels* and *num_classes* are added as parameters.
        - Now, locate to `/utils/helpers.py` and go to `line 26` where it says `# pick model`. Add you model in a conditional as 
            ```
            elif model_name == "NewModel":
                 model = new_model(input_channels = configs.input_channels, num_classes = configs.num_classes)
            ```
         - Remember to pass the `NewModel` in the `configs.json` file.
- Adding a new dataset :
        - Locate to `/utils/data.py` and create a new function `def load_my_new_model(configs)` to load your own dataset.
        - Now, in the same file, go to `line 8` and find the function `load_dataset`.
        - Add a conditional to load your model
          ```
          elif configs.dataset == "NewDataset":
            return load_my_new_model(configs)
          ```
        - Remember to add `NewDataset` in the `configs.json` before running
        
### Utils/delta.py
- This file contains our new proposed **Relative Weight Change** metric to determine the layer wise learning trends in deep networks.
## Repository Structure
```
Layer-Wise-Learning-Trends-PyTorch
├── models
│   ├── alexnet.py
│   │   
│   ├── resnet.py
│   │  
│   └── vgg.py
│       
├── utils
|   ├── data.py
│   │   
│   ├── delta.py
│   │  
│   └── helpers.py
├── main.py
├── README.md
├── requirements.txt
├── train.py
├── mult_train.py
└── configs.json
```

#### Citation
```bibtex
@misc{agrawal2020investigating,
      title={Investigating Learning in Deep Neural Networks using Layer-Wise Weight Change}, 
      author={Ayush Manish Agrawal and Atharva Tendle and Harshvardhan Sikka and Sahib Singh and Amr Kayid},
      year={2020},
      eprint={2011.06735},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement 
- This README.md is inspired from [rahulvigneswaran/Lottery_Ticket_Hypothsis](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch)
- VGG model was borrowed from [chengyangfu/pytorch-vgg-cifar10](https://github.com/chengyangfu/pytorch-vgg-cifar10)
- ResNet model was borrowed from [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)


## Issue / Want to Contribute ? :
Open a new issue or do a pull request incase you are facing any difficulty with the code base or if you want to contribute to it.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/Manifold-Computing/Layer-Wise-Learning-Trends-PyTorch/issues)

Want to be a part of our organization? Checkout [Manifold Computing](https://manifoldcomputing.com/)
