import functools

search_space = {
    "logreg":
        {
            "batch_size": {"type": "cat", "args": [1024, 2048, 4096]},
            "learning_rate": {"type": "float", "args": [1e-4, 1e-2]},
            "weight_decay": {"type": "float", "args": [1e-3, 1e-1]},
        },
    "mlp":
        {
            "batch_size": {"type": "cat", "args": [1024, 2048, 4096]},
            "first_hidden_coef": {"type": "float", "args": [0.5, 2]},
            "layers_num": {"type": "int", "args": [1, 3]},
            "learning_rate": {"type": "float", "args": [1e-4, 1e-2]},
            "weight_decay": {"type": "float", "args": [1e-3, 1e-1]},
            "dropout": {"type": "float", "args": [0.05, 0.2]},
        },
    "resnet":
        {
            "batch_size": {"type": "cat", "args": [1024, 2048, 4096]},
            "hidden_factor": {"type": "float", "args": [0.5, 2]},
            "resnet_block_num": {"type": "int", "args": [1, 3]},
            "learning_rate": {"type": "float", "args": [1e-4, 1e-2]},
            "weight_decay": {"type": "float", "args": [1e-3, 1e-1]},
            "dropout": {"type": "float", "args": [0.05, 0.2]},
        },
}


ds_features_count = {
    "mnist_2": 28*28/2,
    "mnist_3": 28*28/3,
    "mnist_4": 28*28/4,
    "mnist_5": 28*28/5,
    'sbol_1': 1345,
    "sbol_smm_2": 1354/2,
    "sbol_zvuk_2": 1354/2,
    "sbol_smm_zvuk_3": 1365/3, #1364
    'home_credit_1': 90 / 2,
    'home_credit_bureau_2': (90 + 15 + 1) / 2,
    'home_credit_pos_2': (90 + 231 + 1) / 2,
    'home_credit_bureau_pos_3': (90 + 15 + 231) / 3,
    'avito_1': 21,
    'avito_texts_2': (21+772+1)/2,
    'avito_images_2': (21+4+1)/2,
    'avito_texts_images_3': (21+772+4+1)/3


}

metrics_to_opt_dict = {
    "mnist": "metrics.test_roc_auc_macro",
    "sbol": "metrics.test_roc_auc_macro",
    "sbol_smm": "metrics.test_roc_auc_macro",
    "sbol_zvuk": "metrics.test_roc_auc_macro",
    "sbol_smm_zvuk": "metrics.test_roc_auc_macro",
    "home_credit": "metrics.test_roc_auc_macro",
    "home_credit_bureau": "metrics.test_roc_auc_macro",
    "home_credit_pos": "metrics.test_roc_auc_macro",
    "home_credit_bureau_pos": "metrics.test_roc_auc_macro",
    "avito": "metrics.test_rmse",
    "avito_texts": "metrics.test_rmse",
    "avito_images": "metrics.test_rmse",
    "avito_texts_images": "metrics.test_rmse",

}

def suggest_params(trial, config):
    suggested_params = {}
    models_search_space = search_space[config.vfl_model.vfl_model_name]
    for param_name, param_values in models_search_space.items():
        border_min, border_max = param_values["args"][0], param_values["args"][1]
        if param_values["type"] == "float":
            suggested_params[param_name] = trial.suggest_float(param_name, border_min, border_max)
        elif param_values["type"] == "int":
            suggested_params[param_name] = trial.suggest_int(param_name, border_min, border_max)
        elif param_values["type"] == "cat":
            suggested_params[param_name] = trial.suggest_categorical(param_name, param_values["args"])
        else:
            ValueError(f"Unsupported type {param_values['type']}")
    return suggested_params


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def compute_hidden_layers(config, suggested_params, param_val):
    hidden_layers = []
    avg_features_count = ds_features_count[f"{config.data.dataset}_{config.common.world_size}"]
    hidden_layers.append(int(param_val * avg_features_count))
    if suggested_params["layers_num"] > 1:
        if config.common.world_size > 1:
            output_dim = config.master.master_model_params["output_dim"]
        else:
            output_dim = config.member.member_model_params["output_dim"]
        delta = hidden_layers[0] - output_dim
        diff = int(delta / suggested_params["layers_num"])
        for _ in range(suggested_params["layers_num"] - 1):
            hidden_layers.append(hidden_layers[-1] - diff)
    return hidden_layers


def change_member_model_param(config, model_param_name, new_value):
    member_model_params = config.member.member_model_params
    member_model_params[model_param_name] = new_value
    rsetattr(config, f"member.member_model_params", member_model_params)


def change_master_model_param(config, model_param_name, new_value):
    master_model_params = config.master.master_model_params
    master_model_params[model_param_name] = new_value
    rsetattr(config, f"master.master_model_params", master_model_params)