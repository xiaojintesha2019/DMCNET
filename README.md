# DMCnet demo _ONLINE Time Series Forecasting Models


## Introduction for DMCnet
![LOADING](/DMCNET.jpg "DMCNET OVERVIEW")

## Introduction for PCpower

![LOADING](/pcpower.png "pcpower demo")
The annotation of the PCpower dataset follows the principle of accurately depicting the association between user behavior and component power consumption, and constructs a clear, layered, and logically rigorous annotation system for different usage scenarios. For example, in the entertainment web page scenario, key information such as the URL of each newly opened web page, the time of access, and the page-staying duration is recorded in detail. This aims to outline the user's behavior trajectory in the web browsing process and the corresponding power consumption changes of each component through these fine-grained annotations.

In the office work scenario, for the key operations of commonly used office software (such as Word, Excel, PPT, etc.), including inserting pictures, editing text, creating and saving files, the specific timestamps, operation sequences, and related file attributes are all meticulously annotated. In this way, the mapping relationship between office-operation behavior and the power consumption of each component is established.

In the professional creation scenario, for the core links involved in video editing, such as importing videos, editing videos, previewing videos, and exporting and rendering, not only the start and end times and operation parameters of each link are recorded, but also the power consumption status changes of each component of the notebook computer corresponding to them are annotated. Through this comprehensive and detailed annotation system, the intrinsic connection between complex-operation behavior and power consumption in professional creation tasks is deeply revealed.

Through this annotation process, not only can it provide annotated data with clear semantics and structure for subsequent behavior analysis, scenario classification, and power consumption prediction research based on data mining and machine learning, but it also helps to deepen the theoretical understanding of the coupling relationship between user behavior and computer system energy consumption, and promotes the academic research progress in user behavior perception, energy efficiency optimization, and intelligent system control in related fields.

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

