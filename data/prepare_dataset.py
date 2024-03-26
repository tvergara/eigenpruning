from data.cb import prepare_cb
from data.copa import prepare_copa
from data.rte import prepare_rte
from data.int_sum import prepare_sum_dataset

def prepare_datasets(name, model):
    if name == 'cb':
        return prepare_cb(model)
    if name == 'copa':
        return prepare_copa(model)
    if name == 'rte':
        return prepare_rte(model)
    if name == 'int_sum':
        return prepare_sum_dataset(model)
