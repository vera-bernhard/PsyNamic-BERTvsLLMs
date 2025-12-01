# Can LLMs Beat BERT in Biomedical Information Extraction?  
## Evaluating Prompting and Fine-Tuning Strategies for NER and Classification

**Author:** Vera Bernhard  
**Date:** December 2025  
**Institution:** University of Zurich, Switzerland
---

## Repository Overview

This repository contains the code and data for the Master’s thesis  
**“Can LLMs Beat BERT in Biomedical Information Extraction? Evaluating Prompting and Fine-Tuning Strategies for NER and Classification”** by Vera Bernhard.

### Structure

- `bert_baseline/`: Prediction files and evaluation outputs for the BERT baseline models  
- `data/`: The PsyNamic dataset  
- `evaluation/`: Evaluation, post-processing, and plotting scripts  
- `few_shot/`: Predictions and plots for the few-shot experiments  
- `finetuning/`: All files related to fine-tuning LLMs  
  - `ift/`: Instruction fine-tuning dataset and training scripts  
  - `lst/`: Label-supervised fine-tuning scripts and predictions  
- `prompts/`: Prompt templates, prompt generation scripts, and annotation guidelines for the PsyNamic dataset  
- `test/`: Unit tests for evaluation and post-processing scripts  
- `zero_shot/`: Predictions and plots for zero-shot experiments, including predictions from the instruction fine-tuned model  

### Technologies Used

- **Python 3.12**
- **Hugging Face Transformers** – model loading, inference, and training
- **PEFT** – parameter-efficient fine-tuning methods
- **TRL** – training large language models with instruction tuning
- **BiLLM** – converting LLMs from uni-directional to bidirectional for classification tasks  
  https://github.com/WhereIsAI/BiLLM