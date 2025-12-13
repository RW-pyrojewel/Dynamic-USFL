# src/config/parser.py
"""
Simple config loader for YAML experiment configs.

Usage
-----
from src.config.parser import load_config

cfg = load_config("configs/ham18_static_usfl.yaml")
print(cfg.experiment.name)
print(cfg.data.batch_size)
"""

import os
from typing import Any, Dict, Optional
from types import SimpleNamespace

import yaml


def _dict_to_namespace(obj: Any) -> Any:
    """
    Recursively convert a dict (and nested dicts/lists) to SimpleNamespace,
    so we can use dot access: cfg.data.batch_size.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_dict_to_namespace(v) for v in obj]
    else:
        return obj


def _apply_overrides(cfg_dict: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """
    Apply simple dotted-key overrides on top of a dict config.

    Example:
        overrides = {
            "training.epochs": 100,
            "optimizer.lr": 0.001,
        }
    """
    for key, value in overrides.items():
        parts = key.split(".")
        cur = cfg_dict
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        # 简单的类型转换：把 "123" 转成 int，把 "0.1" 转成 float
        if isinstance(value, str):
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        cur[parts[-1]] = value


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    """
    Load a YAML config file and return a SimpleNamespace object.

    Parameters
    ----------
    path : str
        Path to the YAML config file.
    overrides : dict, optional
        Dotted-key overrides, e.g. {"training.epochs": 100}.

    Returns
    -------
    cfg : SimpleNamespace
        The loaded config. Access with cfg.data.batch_size, etc.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    if cfg_dict is None:
        cfg_dict = {}

    # Apply command-line / external overrides if provided
    if overrides:
        _apply_overrides(cfg_dict, overrides)

    # Attach config path for reference
    cfg_dict.setdefault("experiment", {})
    cfg_dict["experiment"]["config_path"] = path

    # Auto-create output_dir if present
    out_dir = cfg_dict.get("experiment", {}).get("output_dir", None)
    if out_dir is not None:
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        cfg_dict["experiment"]["output_dir"] = out_dir

    return _dict_to_namespace(cfg_dict)


def parse_overrides(override_list) -> Dict[str, str]:
    """
    Parse ["training.epochs=50", "optimizer.lr=0.001"] into dict.
    类型转换交给 load_config 里 _apply_overrides 去做。
    """
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value.")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    return overrides
