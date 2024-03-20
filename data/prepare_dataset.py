from data.cb import prepare_cb

def prepare_datasets(name, tokenizer):
    if name == 'cb':
        return prepare_cb(tokenizer)
