import functools

search_space = {
    "logreg":
        {
            "batch_size": {"type": "cat", "args": [512, 1024, 2048]}, #32, 64, 128, 256, 512, , 4096
            "learning_rate": {"type": "float", "args": [1e-4, 1e-2]},
            "weight_decay": {"type": "float", "args": [1e-3, 1e-1]},
        }
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