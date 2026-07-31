import yaml
from pathlib import Path
from config.config_types import Params


# define the paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PARAMS_PATH = ROOT_DIR / "params.yaml"

# load the config file
def load_config(params_path: Path | str):
    with open(params_path, "r") as file:
        yaml_config = yaml.safe_load(file)
        
        # validate the params
        params_config = Params.model_validate(obj=yaml_config,
                              strict=True)
        return params_config
    
params_config = load_config(PARAMS_PATH)