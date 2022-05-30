# User authentication based on Mouse Movements  
This project presents a solution to authentify users based on their mouse movements with the design of a 
Gated Recurrent Unit (GRU) classifier that is able to differentiate imposters from the actual user. 
In the report of the BSP, we defined and reported the results obtained in the different empirical experiments. In order to make our research reproducable and verifiable, we published the code to train and evaluate the models for every experiment. 

This repository is licensed under the MIT license.
## Requirements
In order to correctly execute the experiments, and to obtain the same reported results, following dependencies shall be installed,

- Python Packages, all packages present in the requirements.txt file shall be installed in the given version.
- Tensorflow in the version 2.7.0 shall be installed

## Training the model
For every experiment, in order to obtain the results, the models need to be trained with the following command
```python3 ./experiment<ExperimentID>/training.py```
Once executed, the script created the three following folders,
- **./models**, every model for every user is saved in this folder. Every model is saved as follows, **user<#User>-fold-<#ofthefold>**.
- **./models-testingsets**: for every trained model, the corresponding testing dataset and labels are saved in this folder under a numpy file. Every testing set is saved as follows, **user<#User>-fold-<#ofthefold>-test-X.npy**. Every testing label, is saved as follows,**user<#User>-fold-<#ofthefold>-test-Y.npy**.
- **./models-Tensorboard**: in this folder, the training history of the trained model are saved, in order to be used by Tensorboard, which allows you visualise the training history and metrics. Each trained model has its own subfolder, saved as follows, **user<#User>-fold-<#ofthefold>**.
## Evaluating the model
- Computing the evaluation metrics: 

- Using Tensorboard:
## Author

Paul Houssel - Student of the University of Luxembourg

## Acknowledgements and References

The work was supported by the University of Luxembourg and Luis. Leiva

The dataset used in this work is available [here](https://github.com/balabit/Mouse-Dynamics-Challenge).
