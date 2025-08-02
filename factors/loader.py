import os
import yaml
from qlib.data.dataset.loader import QlibDataLoader


class CQilbDL(QlibDataLoader):
    """Dataloader to get CQilb"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config(yaml_path=None):
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "factors.yaml"
            )
        with open(yaml_path, "r", encoding="utf-8") as f:
            factors = yaml.safe_load(f)

        # 只保留source为"custom_factor"且包含"qlib_expression"字段的因子
        factors = [
            {
                "name": factor["name"],
                "field": factor["qlib_expression"],
            }
            for factor in factors
            if factor.get("source") == "custom_factor" and "qlib_expression" in factor
        ]

        names = [factor["name"] for factor in factors]
        fields = [factor["field"] for factor in factors]
        return fields, names
