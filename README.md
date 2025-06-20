# DMCnet demo _ONLINE Time Series Forecasting Models


## Introduction for DMCnet


## Introduction for PCpower



## Requirements

- python == 3.9.13
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- tqdm == 4.62.3
- einops == 0.4.0

## Benchmarking

### 1. Data preparation

We follow the same data formatting as the Informer repo (https://github.com/zhouhaoyi/Informer2020), which also hosts the raw data.
Please put all raw data (csv) files in the ```./data``` folder.

You can directly access and test our **PCpower dataset** in the ```./data``` folder.  

### 2. Run experiments

To replicate our results on the ETT, weather, WTH, and PCpower datasets, run
```
sh run.sh
```

### 3.  Arguments

You can specify one of the above method via the ```--method``` argument.

**Dataset:** supports the following datasets: Electricity Transformer - ETT (including ETTh1, ETTh2, ETTm1, and ETTm2), ECL,  weather, WTH,and PCpower. You can specify the dataset via the ```--data``` argument，also modify this file  ```./main```for addtional data sheet test.

**Other arguments:** Other useful arguments for experiments are:

- ```--test_bsz```: batch size used for testing: must be set to **1** for online learning,
- ```--seq_len```: look-back windows' length, set to **60** by default,
- ```--pred_len```: forecast windows' length, set to **1** for online learning.




## Acknowledgement

This library is mainly constructed based on the following repos, following the training-evaluation pipelines and the implementation of baseline models:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library.
- OneNet： https://github.com/yfzhang114/OneNet

All the experiment datasets are public, and we obtain them from the following links:
- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.

