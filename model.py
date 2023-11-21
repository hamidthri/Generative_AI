from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

# Loading Dataset
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

# Loading the FLAN-T5 Model
model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Showing two examples
dash_line = '-'.join('' for x in range(100))
example_indices = [40, 200]
for i, index in enumerate(example_indices):
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print('INPUT DIALOGUE:')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('BASELINE HUMAN SUMMARY:')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()

# Tokenize Example
sentence = "What time is it, Tom?"
sentence_encoded = tokenizer(sentence, return_tensors='pt')
sentence_decoded = tokenizer.decode(
        sentence_encoded["input_ids"][0],
        skip_special_tokens=True
    )
print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])
print('\nDECODED SENTENCE:')
print(sentence_decoded)





# generation_config = GenerationConfig(max_new_tokens=50)
# # generation_config = GenerationConfig(max_new_tokens=10)
# # generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)
# # generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)
# # generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=1.0)
# inputs = tokenizer(few_shot_prompt, return_tensors='pt')
# output = tokenizer.decode(
#     model.generate(
#         inputs["input_ids"],
#         generation_config=generation_config,
#     )[0],
#     skip_special_tokens=True
# )
# print(dash_line)
# print(f'MODEL GENERATION - FEW SHOT:\n{output}')
# print(dash_line)
# print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')