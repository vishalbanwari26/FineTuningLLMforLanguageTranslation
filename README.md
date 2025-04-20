# Fine-Tune LLM for Language Translation 🌐

This project focuses on fine-tuning a multilingual Large Language Model (LLM) for German-French language translation using efficient training techniques and evaluating it using standard metrics.

## 📌 Project Overview

This repository contains the implementation and interface for fine-tuning the `bigscience/bloomz-3b` model on various German-French datasets (real, synthetic, combined), evaluating performance using BLEU and BERTScore, and providing a Gradio-powered translation interface.

## 🧠 Objectives

- Fine-tune a multilingual general-purpose LLM for German-French translation.
- Evaluate performance across multiple datasets and metrics.
- Explore parameter-efficient fine-tuning with LoRA.
- Deploy the best model on Hugging Face and build a user-friendly Gradio interface.

## 📊 Dataset

- **Primary dataset:** [Tatoeba](https://huggingface.co/datasets/tatoeba) — 1000 German-French pairs (80% train / 20% test).
- **Synthetic dataset:** 1600 pairs generated using [LLaMA3-70B](https://huggingface.co/meta-llama).
- **Combined dataset:** Merged Tatoeba and synthetic pairs.

## 🧩 Model and Fine-Tuning Strategy

- **Base Model:** `bigscience/bloomz-3b`
- **PEFT Method:** LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit with BitsAndBytesConfig
- **LoRA Config:**
  - `r = 16`, `alpha = 64`, `dropout = 0.1`
  - `target_modules = ["query_key_value"]`
  - `modules_to_save = ["lm_head"]`

### 🔧 Training Parameters

- `batch_size = 2`
- `gradient_accumulation_steps = 4`
- `learning_rate = 2e-5`
- `num_train_epochs = 2`
- `optimizer = paged_adamw_8bit`
- `fp16 = True`, `gradient_checkpointing = True`

## 📈 Evaluation Metrics

- **SacreBLEU** — measures n-gram overlap.
- **BERTScore** — semantic similarity using BERT embeddings.

| Model      | Dataset            | SacreBLEU | BERTScore-F1 |
|------------|--------------------|-----------|--------------|
| Model A    | Pretrained         | 25.63     | 0.92         |
| Model B    | Tatoeba (real)     | 39.91     | 0.94         |
| Model C    | Synthetic          | 25.59     | 0.91         |
| Model D    | Combined           | 27.39     | 0.92         |

## 🎛️ Gradio Interface

A web-based UI to test translations using the fine-tuned model:
- Input: German text
- Output: French translation
- Hosted model: [`vbanwari/Fine-tuned-for-language-translation-bloomz3b-tatoeba`](https://huggingface.co/vbanwari/Fine-tuned-for-language-translation-bloomz3b-tatoeba)

## 🌱 Key Learnings & Reflections

- **BLEU vs BERTScore:** Valuable insight into evaluating translation quality with surface vs semantic measures.
- **Model Hosting:** Hosted a model on Hugging Face for the first time.
- **Data Quality > Quantity:** High-quality real translations outperformed large synthetic datasets.

## ⚖️ Ethical Considerations

- Use of open-source datasets like Tatoeba with informed consent.
- Carbon impact minimized via efficient training strategies.
- Ensured synthetic data generation aligns with LLaMA3’s acceptable use policy.

## 📁 Files

- `Portfolio_Task1_VB.ipynb`: Implementation steps and evaluation.
- `Portfolio_Interface_Gradio_VB.ipynb`: Gradio UI for translation.

## 📚 References

- [PEFT with Hugging Face](https://github.com/huggingface/peft)
- [BLOOMZ model](https://huggingface.co/bigscience/bloomz-3b)
- [SacreBLEU](https://aclanthology.org/W18-6319/)
- [BERTScore](https://arxiv.org/abs/1904.09675)

---

> **Author**: Vishal Banwari

