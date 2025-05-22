![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/refs/heads/master/timefuse/method.png)

<h2 align="center">
    [ICML'25] Sample-level Adaptive Model Fusion for Time Series Forecasting <br>
</h2>

**TimeFuse is an ensemble time series forecasting framework that adaptively fuses multiple models at the sample level to exploit their complementary strengths.**
We extract the meta-features of the input time series and use them to learn a fusion model that dynamically combines the predictions of a diverse set of base models.
The framework is model-aganostic, allowing for the integration of various forecasting models, including deep learning and traditional statistical methods.

Our implementation is based on the [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) to take advantage of its extensive collection of forecasting models and searched training configs.

<!-- ## Table of Contents -->
<!-- - [🧳 Usage](#-usage)
- [🗂️ Project Structure](#️-project-structure) -->


## 🧳 Usage

1. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
2. **Prepare the datasets:**
   
   Simply download https://drive.google.com/file/d/1H40tc6cO1lWVcU6WBJ0sg_jwOGh_XBDE/view?usp=sharing and put all files in the root directory of the project. The dataset should be in the `dataset/` directory.
   We also provide some pretrained model checkpoints in the `checkpoints/` directory, and extracted meta-training data in the `meta_data/` directory. You can use them to quickly run the example notebook `run_timefuse.ipynb` with the default config without training the base models from scratch.

3. **Run an TimeFuse experiment:**

    **We provide a step-by-step guide to run TimeFuse experiment in the `run_timefuse.ipynb` notebook, please check the notebook for more details.**

    The general steps are:
    1. **Load experiment configs**: Load the default config from `run_config.json` or create your own config file.
    2. **Train base models**: Train the base models with loaded configs, if the checkpoints does not exist.
    3. **Extract meta-features**: Extract meta-training data and store in the `meta_data/` folder.
        Given a training time series pair (`X_in`, `X_out`) and k base models, we extract the following meta-training data:
        - `x_meta`: the meta-features of the input time series `X_in`
        - `y_model_preds`: the predictions of the k base models (i.e., their preditions of `X_out`)
        - `y_true`: the ground truth `X_out`
    4. **Train the fusion model**: Train the fusion model with the meta-training data.
    5. **Evaluate the fusion model**: Compare TimeFuse and all base models on the test set.


## 🗂️ Project Structure

- `args.py`, `load_configs.py`, `meta_feature.py`, `timefuse.py`: Core scripts for configuration, meta-feature extraction, and fusion logic.
- `run.py`: Main entry point for running experiments.
- `run_base_model_train.ipynb`: Example notebook for base model training.
- `run_config.json`: Example configuration file.
- `data_provider/`: Data loading and preprocessing utilities.
- `dataset/`: Datasets for experiments.
- `exp/`: Experiment modules.
- `layers/`: Model layers and components.
- `meta_data/`: Meta-data for fusion.
- `models/`: Model definitions.
- `scripts/`: Helper scripts.
- `utils/`: Utility functions.
- `checkpoints/`: Saved model checkpoints.