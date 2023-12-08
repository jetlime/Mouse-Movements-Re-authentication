# User authentication based on Mouse Movements  
This project presents a solution to authentify users based on their mouse movements with the design of a 
Gated Recurrent Unit (GRU) classifier that is able to differentiate imposters from the actual user.
Our solution is based on the [Balabit](https://github.com/balabit/Mouse-Dynamics-Challenge) dataset.

We publish the code required to reproduce reported experimental results to ensure our research is reproducable and verifiable.

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

Once the training script has been executed, we can now evaluate the model as follows, 

```python3 ./experiment<ExperimentID>/evaluation.py```

This scipt will report the average EER, Accuracy and AUC score of the given experiment. The results are given for each user, an average for every user is also computed. 

- Using Tensorboard:

By using Tensorboard we are able to visualise the training history and metrics in function of the training iterations and epochs. 
Tensorboard is reporting the results on a localhost webpage, once the following command is executed,

```tensorboard --logdir ./experiment<ExperimentID>/models-Tensorboard```

## Author

Paul Houssel - Student of the University of Luxembourg

Luis A. Leiva - Assistant professor in computer science, University of Luxembourg

## Acknowledgements 

This work was supported by the Horizon 2020 FET program of the European Union (grant CHIST-ERA-20-BCI-001) and the European Innovation Council Pathfinder program (SYMBIOTIK project).

