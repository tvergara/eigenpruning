from data.cb import prepare_cb
from data.copa import prepare_copa
from data.rte import prepare_rte

def prepare_datasets(name, tokenizer):
    if name == 'cb':
        return prepare_cb(tokenizer)
    if name == 'copa':
        return prepare_copa(tokenizer)
    if name == 'rte':
        return prepare_rte(tokenizer)
