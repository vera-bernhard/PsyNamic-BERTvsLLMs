# Can LLMs Beat BERT in Biomedical Information Extraction? Evaluating Prompting and Fine-Tuning Strategies for NER and Classification

Author: Vera Bernhard
Date: December 2025

This repository contains the code and data for the Master's thesis "Can LLMs Beat BERT in Biomedical Information Extraction? Evaluating Prompting and Fine-Tuning Strategies for NER and Classification" by Vera Bernhard
================

The repository is structured as follows:
- `bert_baseline/`: contains the prediction files and evaluation files for the BERT baseline models.
- `data/`: Contains the PsyNamic dataset
- `evaluation/`: contains the evaluation, post-processing scripts and plot scripts.
- `few_shot`: contains all predictions and plots for the few-shot experiments.
- `finetuning/`: contains all files related to fine-tuning the LLMs.
    - `ift/`: contains the instruction fine-tuning dataset and training scripts.
    - `lst/`: contains the label-supervised fine-tuning scripts and predictions.
- `prompts/`: contains the prompt templates, scripts to generate prompts and annotation guidelines of the PsyNamic dataset.
- `test`: contains unit test for the evaluation and post-processing scripts.
- `zero_shot/`: contains all predictions and plots for the zero-shot experiments, and also the predictions of the instruction fine-tuned model.
  