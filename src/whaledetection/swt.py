from config_loader import load_config
from root_path import find_project_root
root = find_project_root()
config_path=root / "configs"/"config.yaml"
cfg = load_config(config_path)
print(cfg.swt.swt_frame_length)