import toml
CONFIG_FILE_NAME = "model_config.toml"


def load(path):
    config = toml.load(path)

    g = globals()
    for k, v in config.items():
        g[k] = v


def save(model_dir):
    import os

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config = {}
    for k, v in globals().items():
        if not k.startswith("_"):
            config[k] = v

    with open(os.path.join(model_dir, CONFIG_FILE_NAME), "w+") as f:
        toml.dump(config, f)
