from qlib.contrib.data.handler import DataHandlerLP, check_transform_proc, _DEFAULT_LEARN_PROCESSORS, _DEFAULT_INFER_PROCESSORS
from qlib.contrib.data.loader import Alpha158DL

from factors.loader import CPyDL, CQilbDL

class CombineHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )

        data_loader = {
            "class": "qlib.data.dataset.loader.NestedDataLoader",
            "kwargs": {
                "dataloader_l": [
                    {
                        "class": "qlib.contrib.data.loader.Alpha158DL",
                        "kwargs": {
                            "config": {
                                "label": kwargs.pop("label", self.get_label_config()),
                            }
                        },
                    },
                    {
                        "class": "factors.loader.CQilbDL",
                    },
                    {
                        "class": "factors.loader.CPyDL",
                    },
                ],
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        alpha158_fields, alpha158_names = Alpha158DL.get_feature_config(conf)
        cpydl_fields, cpydl_names = CPyDL.get_feature_config()
        cqilbdl_fields, cqilbdl_names = CQilbDL.get_feature_config()
        fields = alpha158_fields + cpydl_fields + cqilbdl_fields
        names = alpha158_names + cpydl_names + cqilbdl_names

        return fields, names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
