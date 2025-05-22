import re
import os
from copy import copy
import numpy as np
from args import get_parser
from pathlib import Path
import json

data_configs = {
    "ETTh1": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "n_dim": 7,
    },
    "ETTh2": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "ETTh2.csv",
        "data": "ETTh2",
        "n_dim": 7,
    },
    "ETTm1": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "ETTm1.csv",
        "data": "ETTm1",
        "n_dim": 7,
    },
    "ETTm2": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "ETTm2.csv",
        "data": "ETTm2",
        "n_dim": 7,
    },
    "weather": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "weather.csv",
        "data": "custom",
        "n_dim": 21,
    },
    "electricity": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "electricity.csv",
        "data": "custom",
        "n_dim": 321,
    },
    "traffic": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "traffic.csv",
        "data": "custom",
        "n_dim": 862,
    },
    "PEMS03": {
        "root_path": "./dataset/short_term_forecast/PEMS/",
        "data_path": "PEMS03.npz",
        "data": "PEMS",
        "n_dim": 358,
    },
    "PEMS04": {
        "root_path": "./dataset/short_term_forecast/PEMS/",
        "data_path": "PEMS04.npz",
        "data": "PEMS",
        "n_dim": 307,
    },
    "PEMS07": {
        "root_path": "./dataset/short_term_forecast/PEMS/",
        "data_path": "PEMS07.npz",
        "data": "PEMS",
        "n_dim": 883,
    },
    "PEMS08": {
        "root_path": "./dataset/short_term_forecast/PEMS/",
        "data_path": "PEMS08.npz",
        "data": "PEMS",
        "n_dim": 170,
    },
    "NP": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "NP.csv",
        "n_dim": 3,
        "e_layers": 3,  # from TimeXer
        "batch_size": 4,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "PJM": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "PJM.csv",
        "n_dim": 3,
        "e_layers": 3,  # from TimeXer
        "batch_size": 16,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "BE": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "BE.csv",
        "n_dim": 3,
        "e_layers": 2,  # from TimeXer
        "batch_size": 16,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "FR": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "FR.csv",
        "n_dim": 3,
        "e_layers": 2,  # from TimeXer
        "batch_size": 16,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "DE": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "DE.csv",
        "n_dim": 3,
        "e_layers": 1,  # from TimeXer
        "batch_size": 4,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
}

support_datasets = list(data_configs.keys())

default_models = [
    "DLinear",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "PAttn",
    "TimeMixer",
    "TimeXer",
]

default_args = get_parser().parse_args(  # TSLib default args
    """
    --task_name long_term_forecast
    --is_training 1
    --root_path ./dataset/ETT-small/
    --data_path ETTh1.csv
    --model_id ETTh1_96_96
    --model TimesNet
    --data ETTh1
    --features M
    --seq_len 96
    --label_len 48
    --pred_len 96
    --e_layers 2
    --d_layers 1
    --factor 3
    --enc_in 7
    --dec_in 7
    --c_out 7
    --d_model 16
    --d_ff 32
    --des Exp
    --itr 1
    --top_k 5
    """.split()
)

model_dim_lim = {
    "long_term_forecast": (32, 512),
    "short_term_forecast": (16, 64),
    "imputation": (64, 128),
    "classification": (32, 64),
    "anomaly_detection": (32, 128),
}


def get_model_dim(args):
    """
    If no existing config found, set the model dimension based on the input dimensions following TimesNet.
    """
    d_min, d_max = model_dim_lim[args.task_name]
    d_model = int(min(max(np.exp2(np.ceil(np.log(args.n_dim))), d_min), d_max))
    return d_model


def load_config_from_shell(shell_path):
    """
    Parse the shell script and return a dictionary with configurations as strings.

    Parameters:
        shell_path (str): Path to the shell script file.

    Returns:
        dict: A dictionary where keys are (seq_len, label_len, pred_len) combinations,
              and values are the corresponding configurations as argument strings.
    """
    # Read the shell script
    shell_content = Path(shell_path).read_text()

    # Merge lines with trailing backslashes (\\)
    merged_content = ""
    for line in shell_content.splitlines():
        if line.strip().endswith("\\"):
            merged_content += line.strip()[:-1] + " "
        else:
            merged_content += line.strip() + "\n"

    # Extract environment variables
    env_vars = {}
    env_pattern = re.compile(r"^(\w+)=([^\n]+)", re.MULTILINE)
    for match in env_pattern.finditer(merged_content):
        var, value = match.groups()
        env_vars[var] = value.strip().strip("'\"")

    # Extract Python commands and arguments
    config = {}
    command_pattern = re.compile(r"python -u run\.py(.*)", re.MULTILINE)
    for command_match in command_pattern.finditer(merged_content):
        command = command_match.group(1).strip()

        # Replace variables with their values
        for var, value in env_vars.items():
            command = re.sub(rf"\${{{var}}}|\${var}", value, command)
            command = re.sub(rf"\${var}([\"'].*?[\"'])", rf"{value}\1", command)

        # Extract arguments for (seq_len, label_len, pred_len) as key
        args_pattern = re.compile(
            r"--seq_len\s+(\d+).*--label_len\s+(\d+).*--pred_len\s+(\d+)"
        )
        match = args_pattern.search(command)
        if match:
            seq_len, label_len, pred_len = map(int, match.groups())
            key = f"{seq_len}_{pred_len}"

            # Remove the "python -u run.py" part and trim whitespace
            cleaned_command = re.sub(r"^\s*", "", command).strip()
            config[key] = cleaned_command

    return config


def get_config_path(
    dataname, model, base_path="./scripts/long_term_forecast/", verbose=False
):

    if dataname.startswith("ETT"):
        datapath = "ETT_script"
        model_path = f"{model}_{dataname}.sh"
    elif dataname.startswith("electricity"):
        datapath = "ECL_script"
        model_path = f"{model}.sh"
    else:
        datapath = f"{dataname.title()}_script"
        model_path = f"{model}.sh"
    config_path = base_path + datapath + "/" + model_path
    if os.path.exists(config_path):
        if verbose:
            print(f"Loading TSLib config from {config_path}")
        return config_path
    else:
        # if verbose:
        #     print(f"Config path not found: {config_path}")
        return None


def get_forecast_exp_args(
    dataname,
    modelname,
    seq_len,
    label_len,
    pred_len,
    default_args=default_args,
    data_configs=data_configs,
    base_config_path="./scripts/long_term_forecast/",
    verbose=False,
):
    has_config = False
    try:
        # check if the config file exists
        config_path = get_config_path(
            dataname, modelname, base_config_path, verbose=verbose
        )
        config = load_config_from_shell(config_path)
        # load args from the training script
        key = f"{seq_len}_{pred_len}"
        command = config[key]
        has_config = True
    except:
        pass

    if has_config:
        # load args from the training script
        args = get_parser().parse_args(command.split())
        args.data_name = dataname
        args.n_dim = data_configs[dataname]["n_dim"]
        args.root_path = data_configs[dataname]["root_path"]
        args.model_id = f"{dataname}_{seq_len}_{pred_len}"
        args.train_epochs = 10
        return args
    else:
        # args are not in the config file
        args = copy(default_args)
        args.seq_len = seq_len
        args.label_len = label_len
        args.pred_len = pred_len

        args.data_name = dataname
        args.model = modelname
        for config in data_configs[dataname]:
            setattr(args, config, data_configs[dataname][config])
        args.enc_in = data_configs[dataname]["n_dim"]
        args.dec_in = data_configs[dataname]["n_dim"]

        if "c_out" in data_configs[dataname].keys() and not modelname in [
            "TimeMixer",
        ]:
            args.c_out = data_configs[dataname]["c_out"]
        else:
            args.c_out = data_configs[dataname]["n_dim"]

        inferred_d_model = get_model_dim(args)
        if "d_model" in data_configs[dataname].keys():
            args.d_model = data_configs[dataname]["d_model"]
        else:
            args.d_model = inferred_d_model
        if "d_ff" in data_configs[dataname].keys():
            args.d_ff = data_configs[dataname]["d_ff"]
        else:
            args.d_ff = inferred_d_model

        args.model_id = f"{dataname}_{seq_len}_{pred_len}"
        args.train_epochs = 10
        return args


def get_dataset_forecast_settings(dataset):
    """
    Get the default forecast settings for a given dataset.
    """
    if dataset in [
        "ETTh1",
        "ETTh2",
        "ETTm1",
        "ETTm2",
        "weather",
        "electricity",
        "traffic",
    ]:
        return [
            [96, 48, 96],
            [96, 48, 192],
            [96, 48, 336],
            [96, 48, 720],
        ]
    elif dataset in ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]:
        return [
            [96, 6, 6],
            [96, 12, 12],
            [96, 24, 24],
        ]
    elif dataset in ["NP", "PJM", "BE", "FR", "DE"]:
        return [
            [168, 48, 24],  # from TimeXer
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_all_exp_args(
    datasets,
    models=default_models,
    forecast_settings="auto",
    override_args={},
    base_config_path="./scripts/long_term_forecast/",
    default_args=default_args,
    verbose=False,
):
    assert set(datasets).issubset(
        set(support_datasets)
    ), f"Datasets {datasets} not in {support_datasets}"

    all_args = {}

    if forecast_settings != "auto":
        # forecast_settings is a list of tuples, e.g. [(96, 48, 96), (192, 96, 192)]
        assert isinstance(forecast_settings, list), "forecast_settings should be a list"
        assert all(
            len(setting) == 3 for setting in forecast_settings
        ), "forecast_settings should be a list of (seq_len, label_len, pred_len)"

    for dataset in datasets:
        if forecast_settings == "auto":
            # get the default forecast settings for the dataset
            forecast_settings_parsed = get_dataset_forecast_settings(dataset)
        else:
            # use the provided forecast settings
            forecast_settings_parsed = forecast_settings

        for forecast_setting in forecast_settings_parsed:
            for model in models:
                key = f"{dataset}_{model}_{forecast_setting[0]}_{forecast_setting[1]}_{forecast_setting[2]}"
                args = get_forecast_exp_args(
                    dataset,
                    model,
                    forecast_setting[0],
                    forecast_setting[1],
                    forecast_setting[2],
                    base_config_path=base_config_path,
                    default_args=default_args,
                    verbose=verbose,
                )
                for k, v in override_args.items():
                    setattr(args, k, v)

                # special rules
                if args.model == "TimeMixer":
                    args.label_len = 0  # TimeMixer does not use label_len
                if args.model == "Nonstationary_Transformer":
                    # for numrical stability
                    args.learning_rate = max(args.learning_rate, 0.001)

                all_args[key] = args

    return all_args


def load_run_config(json_path, verbose=True):
    """
    Load a JSON config file and optionally print its contents in formatted one-line style.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    verbose : bool, optional
        Whether to print the loaded configuration, by default True

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(json_path, "r") as file:
        config = json.load(file)

    if verbose:
        label_width = 26  # fixed width for column names
        print(f"Run config file: {json_path}".center(label_width * 2, "="))
        for key, value in config.items():
            if isinstance(value, list):
                k = f"[{len(value)}] {key}"
            else:
                k = key
            print(f"{k:<{label_width}}: {value}")

        # print(
        #     f"{'Datasets':<{label_width}} ({len(config['datasets'])}): {config['datasets']}"
        # )
        # print(
        #     f"{'Models':<{label_width}} ({len(config['models'])}): {config['models']}"
        # )
        # print(
        #     f"{'Forecast Settings':<{label_width}} ({len(config['forecast_settings'])}): {config['forecast_settings']}"
        # )

    return config
