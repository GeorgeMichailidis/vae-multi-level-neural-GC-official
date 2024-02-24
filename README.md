# A VAE-based Framework for Learning Multi-Level Neural Granger-Causal Connectivity

(Copyright 2024) by **Jiahe Lin**, **Huitian Lei** and **George Michailidis**

## Environment Setup

Assume anaconda/miniconda/miniforge has already been installed. To set up the environment, proceed with the following commands:
```console
conda create -n vae-gc python=3.9
conda activate vae-gc
conda install pyyaml numpy pandas scipy scikit-learn
conda install matplotlib seaborn 
pip install pytorch-lightning pytorch
```
See also `requirements.txt`.

To verify that your GPU is up and running:
```console
python -c "import torch; print(torch.cuda.is_available())"
```
Some useful links in case GPUs are not configured correctly:
* https://pytorch.org
* https://lightning.ai/docs/pytorch/stable//starter/installation.html


## Repo Layout

We outline the major components in this repository for ease of navigation.

* `bin/`: shell scripts for execution; see also section [Experiments in the paper](#Experiments-In-the-Paper)
* `src/`:
    - `multiSubVAE.py` and `oneSubVAE.py`: pl.lightning-based modules that encapsulate the pipeline for running VAE-based multi-entity/single-entity methods on a given dataset
    - `simMultiSubVAE.py` and `simOneSubVAE.py`: pl.lightning-based modules for running VAE-based multi-entity/single-entity methods on synthetic data, where the underlying true GC graphs are _known_. Specifically, dataloader (`torch.utils.data.dataloader.DataLoader` object, see `_simdl.py`) and graph evaluation (through `torchmetrics`) is integrated in every step of training, to facilitate model development and tracking.
        * Of note, the printed metrics during training do not correspond to the final metrics presented in the paper (e.g., AUROC and AUPRC). In particular, for the case where graph type is numeric, it calculates a Pearson Correlation-type metric between the truth and the estimates at the individual sample level.
    - `datasets/`: objects with `torch.utils.data.dataset.Dataset` being the base class, to read a specific (type) of dataset from disk so that it can be loaded properly through `DataLoader` later on. 
        * See also the demo in [Run Your Own Datasets](#Run-Your-Own-Datasets) for a concrete example
* `generator/`: scripts used for generating synthetic data
    - `simulator/`:  various simulator objects for synthetic data generation of the corresponding setting
* `utils/`: utilities
    - `simrunner.py`, `realrunner.py`: wrapper functions for training models for synthetic and real data experiments
    - `utils_data.py`: utility functions for data processing and trajectory parsing
    - `utils_eval.py`: for results evaluation
* `configs/`: data parameters (e.g., # of entities, # of nodes, trajectory length, etc) and hyperparameters for all VAE-based methods. Some naming convention:
    - no suffix: this is the base config and the synthetic data setting parameters are specified here in the `data_params` section. The remaining sections correspond to multi-entity learning using a node-centric decoder
    - pattern `*_edge`: edge-centric decoder
    - pattern `*_oneSub`: parameters corresponding to single-entity learning

    One can alternatively deviates from these naming conventions, use any customized config file name and pass it with `--config` in the run command to override the default ones. 
* `root/`:
    - `run_sim.py`: script for running synthetic data experiments using multi-entity VAE-based method
    - `run_simOne.py`: script for running synthetic data experiments using single-entity VAE-based method
    - `train.py`: script for running real data experiments using multi-entity VAE-based method
    - `train_one.py`: script for running real data experiments using single-entity VAE-based method

## Experiments In the Paper
### Synthetic data
* Run synthetic data experiments based on the VAE-based methods, including the proposed one (multi-entity learning) and its single-entity counterpart:
    ```console
    cd bin
    mkdir -p logs
    ## data generation is included by default; toggle (in the shell script) to false if not needed (say, the data has already been generated)
    ## argvs: SETTING_NAME, GPU_ID, CONFIG_VERSION (default to none, indicating no suffix)
    ## choose SETTING_NAME amongst {Lorenz96, LinearVAR, NonLinearVAR, Lotka, Springs5}
    bash run-simVAE.sh [SETTING_NAME] 0 &>logs/[SETTING_NAME]_run.log
    ```
* Evaluate a single run (a specific experiment setting and a single data seed):
    ```console
    ## the following command should be executed in the root dir
    python -u eval_sim.py --ds_str=[SETTING_NAME] --seed=0
    ```
* Evaluate all data replicates for a specific experiment setting:
    ```console
    cd bin
    ## argv: ds_str
    bash eval_sim.sh [SETTING_NAME]
    ```

### Real data - multi subject EEG

Data is available from [rsed2017-dataverse](https://dataverse.tdl.org/dataverse/rsed2017). Once the data are downloaded, they should be put under `data_real/EEG_CSV_files/`, with the filenames being `Subject[ID]_EO.csv` or `Subject[ID]_EC.csv`, depending on the underlying neurophysiological experiment setting.

* Prepare the raw datasets so that the long trajectories are parsed for the VAE-based method to consume
    ```console
    python -u process_EEG.py --ds_str='EEG_EC,EEG_EO'
    ```
* Run the experiments
    ```console
    cd bin
    ## argv: ds_str; choose between EEG_EO and EEG_EC
    bash run-real.sh EEG_EO &>logs/EO_log.log
    ```
## Run Your Own Datasets

See `./demo.ipynb`. In the notebook, we generate a demo dataset and outline the steps/files required to utilize our end-to-end training pipeline. 

## Citation and Contact
To cite this work:
```
@article{Lin2024VAE,
    title = {A VAE-based Framework for Learning Multi-Level Neural Granger-Causal Connectivity},
    author = {Jiahe Lin, Huitian Lei and George Michailidis},
    year = {2024},
    journal = {Transactions on Machine Learning Research},
    url = {https://openreview.net/pdf?id=kNCZ95mw7N}
}
```

* For questions on the paper and/or collaborations based on the methods (extensions or applications), contact [George Michailidis](mailto:gmichail@ucla.edu) 
* For questions on the code implementation, contact [Jiahe Lin](mailto:jiahelin@umich.edu) and/or [Huitian Lei](mailto:ehlei@umich.edu)


## Miscellaneous

Below lists the competitor models considered in the paper and their corresponding repositories

- VAE-based neural relational learning that identifies edge types for a single entity

    - NRI (Kipf et al., 2018) [[link to NRI paper](https://proceedings.mlr.press/v80/kipf18a.html)] [[link to NRI repo](https://github.com/ethanfetaya/NRI)]

- Prediction-model based Granger-causal estimation for a single-entity learning; code therein has been referenced 
    - GVAR (Marcinkevics and Vogt, 2021) [[link to GVAR paper](https://openreview.net/forum?id=DEa4JdMWRHp)] [[link to GVAR repo](https://github.com/i6092467/GVAR)]
    - NGC (Tank et al., 2021) [[link to NGC paper](https://arxiv.org/pdf/1802.05842.pdf)] [[link to NGC repo](https://github.com/iancovert/Neural-GC)]
    - TCDF (Nauta et al., 2019) [[link to TCDF paper](https://www.mdpi.com/2504-4990/1/1/19)] [[link to TCDF repo](https://github.com/M-Nauta/TCDF)]
