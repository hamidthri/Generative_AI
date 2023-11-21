# Generative_AI
## DialogSum:  Prompt Engineering for Summarization

This repository explores the impact of prompt engineering on summarization using the DialogSum dataset. The dataset is loaded using the Hugging Face `datasets` library.

## Loading the FLAN-T5 Model
We utilize the FLAN-T5 model for sequence-to-sequence summarization. The model is loaded and tokenized using the Hugging Face `transformers` library.

## Prompt Engineering Experiments
This repository includes three Python files to demonstrate the impact of prompt engineering on summarization results:

zero_shot.py: Examines summarization results with a default prompt.
one_shot.py: Explores the influence of a single additional prompt on summarization.
few_shot.py: Investigates the impact of multiple prompts on summarization outcomes.
Feel free to explore and run these files to observe how different prompt engineering strategies affect the quality of the summarization results.

Note: Ensure you have the necessary dependencies installed, and consider creating a virtual environment to manage your project's dependencies.
