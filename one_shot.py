from model import dataset
from model import model
from model import tokenizer

dash_line = '-'.join('' for x in range(100))


# ONE SHOT Prompt Engineering "What was going on?"
def make_prompt(example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}


"""

    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    prompt += f"""
Dialogue:

{dialogue}

What was going on?
"""

    return prompt

# One Shot Example
example_indices_full = [40]
example_index_to_summarize = 200
one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)
print(one_shot_prompt)

summary = dataset['test'][example_index_to_summarize]['summary']
inputs = tokenizer(one_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0],
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ONE SHOT:\n{output}')
