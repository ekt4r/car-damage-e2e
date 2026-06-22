from pathlib import Path
import yaml


def test_configs_are_valid_yaml():
    config_paths = list(Path("configs").rglob("*.yaml"))

    assert len(config_paths) > 0

    for path in config_paths:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "data" in cfg
        assert "model" in cfg
        assert "training" in cfg
        assert "optimizer" in cfg
        assert "scheduler" in cfg