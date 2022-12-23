from collections import OrderedDict
from data import build_test_loader_for_m_resnet

class Trainer(object):
    def __init__(self) -> None:
        pass

    @classmethod
    def build_test_loader(dataset_name):
        pass

    @classmethod
    def test(cls, dataset_name, model, evaluators=None):
        results = OrderedDict()
        data_loader, num_query = build_test_loader_for_m_resnet()

