# StepCo-for-HF

**Hugging Face–Compatible Implementation of StepCo**

This repository provides a **Hugging Face–adapted implementation of StepCo**, a framework proposed in

> **Enhancing Mathematical Reasoning in LLMs by Stepwise Correction** (ACL 2025)

The original StepCo pipeline is reimplemented to be fully compatible with the Hugging Face ecosystem, including:

* 🤗 `transformers`
* 🤗 `datasets`
* 🤗 `accelerate`
* standardized model loading & caching
* reproducible training and inference workflows

This repo is intended for **researchers who want to reproduce, extend, or integrate StepCo into modern HF-based LLM pipelines**.

---

## 🔄 Differences from the Original Implementation

| Component       | Original      | This Repo                        |
| --------------- | ------------- | -------------------------------- |
| Model loading   | Custom        | 🤗 `AutoModel` / `AutoTokenizer` |

---

## 📖 Citation

If you use this repository, please cite the original paper:

```bibtex
@inproceedings{wu-etal-2025-stepco,
    title = "Enhancing Mathematical Reasoning in {LLM}s by Stepwise Correction",
    author = "Wu, Zhenyu  and
      Zeng, Qingkai  and
      Zhang, Zhihan  and
      Tan, Zhaoxuan  and
      Shen, Chao  and
      Jiang, Meng",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics",
    year = "2025"
}
```
