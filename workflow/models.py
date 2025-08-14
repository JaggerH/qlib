from qlib.contrib.model.pytorch_nn import DNNModelPytorch
from qlib.contrib.meta.data_selection.utils import ICLoss


class TimeDNN(DNNModelPytorch):
    """
    修改get_metric的原因
    针对单个股票的择时交易，原始的ICLoss是增对股票集合，股票数量小于50直接报错
    """
    def get_metric(self, pred, target, index):
        return -ICLoss(skip_size=1)(pred, target, index)
