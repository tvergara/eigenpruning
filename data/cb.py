from datasets import load_dataset
from data.constants import MAX_LENGTH


PROMPT = """Premise: {premise}

Hypothesis: {hypothesis}

Does the premise entail, contradict or is neutral to the hypothesis?
Answer (entail, contradict, neutral):"""


def prepare_cb(tokenizer):
    tokenizer.padding_side = 'left'
    train_dataset = load_dataset('super_glue', 'cb', split='train')
    test_dataset = load_dataset('super_glue', 'cb', split='validation')

    token_map = {
        0: tokenizer(' entail')['input_ids'][0],
        1: tokenizer(' contradict')['input_ids'][0],
        2: tokenizer(' neutral')['input_ids'][0],
    }


    return {
        'train': tokenize_dataset(train_dataset, tokenizer, token_map),
        'test': tokenize_dataset(test_dataset, tokenizer, token_map)
    }

def tokenize_dataset(dataset, tokenizer, token_map):
    dataset = dataset.map(
        lambda x: format_examples(x, token_map),
        batched=True
    )
    dataset = dataset.map(
        lambda examples: tokenizer(
            examples['prompt'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        ),
        batched=True
    )

    return dataset

def format_examples(examples, token_map):
        prompts = [
            PROMPT.format(premise=premise, hypothesis=hypothesis)
            for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
        ]
        correct_tokens = [token_map[label] for label in examples['label']]
        return {'prompt': prompts, 'correct_token': correct_tokens}
