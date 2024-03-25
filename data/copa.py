from datasets import load_dataset
from data.constants import MAX_LENGTH


PROMPT = """Premise: {premise}
{question}

Alternative 1: {choice_1}
Alternative 2: {choice_2}

What is the correct alternative?
The correct alternative is alternative"""

QUESTION_MAP = {
    'cause': 'What was the cause of this?',
    'effect': 'What happened as a result?',
}

FIRST_TOKEN_INDEX = 0

def prepare_copa(model):
    tokenizer = model.tokenizer
    train_dataset = load_dataset('super_glue', 'copa', split='train')
    test_dataset = load_dataset('super_glue', 'copa', split='validation')

    token_map = {
        0: tokenizer(' 1')['input_ids'][FIRST_TOKEN_INDEX],
        1: tokenizer(' 2')['input_ids'][FIRST_TOKEN_INDEX],
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
            PROMPT.format(premise=premise, choice_1=choice_1, choice_2=choice_2, question=QUESTION_MAP[question])
            for premise, choice_1, choice_2, question in zip(examples['premise'], examples['choice1'], examples['choice2'], examples['question'])
        ]
        correct_tokens = [token_map[label] for label in examples['label']]
        return {'prompt': prompts, 'correct_token': correct_tokens}
