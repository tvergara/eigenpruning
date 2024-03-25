from datasets import load_dataset
from data.constants import MAX_LENGTH


PROMPT = """Premise: {premise}

Hypothesis: {hypothesis}

Does the premise entail or contradict to the hypothesis?
Answer (entails or contradicts): The premise"""

FIRST_TOKEN_INDEX = 1

def prepare_rte(model):
    tokenizer = model.tokenizer
    train_dataset = load_dataset('super_glue', 'rte', split='train')
    test_dataset = load_dataset('super_glue', 'rte', split='validation')

    token_map = {
        0: tokenizer(' entails')['input_ids'][FIRST_TOKEN_INDEX],
        1: tokenizer(' contradicts')['input_ids'][FIRST_TOKEN_INDEX],
    }

    tokenizer.padding_side = 'left'
    train_dataset = tokenize_dataset(train_dataset, tokenizer, token_map)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'correct_token'])
    test_dataset = tokenize_dataset(test_dataset, tokenizer, token_map)
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'correct_token'])

    return {
        'train': train_dataset,
        'test': test_dataset
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
