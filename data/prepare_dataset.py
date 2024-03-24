from data.cb import prepare_cb
from data.copa import prepare_copa

def prepare_datasets(name, tokenizer):
    if name == 'cb':
        return prepare_cb(tokenizer)
    if name == 'copa':
        return prepare_copa(tokenizer)
