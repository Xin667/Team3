# Submission Files Overview

Salman

All my work is organized in a folder named Salman, which contains the files I worked on for the project. Here’s a quick description of each file:

submissions/my_team/cefr_t5_model.jsonl – The main submission file with the simplified test instances produced by my fine-tuned T5 model.

cefr_t5_model_results.xlsx – Excel file showing the evaluation metrics for the submission.

hackathon_and_final_exam.html – HTML version of hackathon and final exam presentation. You can also open it directly here https://molab.marimo.io/notebooks/nb_nExdKGHyZ8vU7qnxciqBsq

tsar2025_test.jsonl – The test dataset that was used to generate the simplified outputs in the submission file.

# Advanced Natural Language Processing - The Road to LLMs - Winter Term 2025/26

This is the GitHub Repository for the ANLP course in the winter term 2025/26. Here, you can find all lecture slides as well as the notebooks and code examples.

## Requirements & Grading

The exam and grade consists of

- A short (max 2 pages) report (pdf format) on the implementation, results and the work done by the team members, along with well-documented notebooks and code (50%, per team)
- A 10 minute oral presentation per team member on their individual work on the 15th of January 2026 (50%, individual)

The final code and report has to be ready in your team's GitHub Repo by the 16th of January 2026, 6pm.

## The challenge

The challenge is the [Shared Task](https://tsar-workshop.github.io/shared-task/) of the 2025 Fourth Workshop on Text Simplification, Accessibility and Readability that was hold alongside the EMNLP 2025 conference.

The task consists of simplifying a given sentence for a specific CEFR reading level (A2 or B1) while preserving the original meaning as much as possible. The data consists of a statement, its simplified version and the target reading level.

We will evaluate your approaches according to the Shared Task evaluation:

- CEFR Compliance: A CEFR-level classifier will verify whether the simplified paragraph meets the specified target level.

- Meaning Preservation: Semantic similarity between the source paragraph and the system output.

- Similarity to References: Semantic similarity between the system output and references.

The [evaluation script](./evaluation/tsar2025_evaluation_script.py) can be found in our repository in the `evaluation/` folder.

You can find the [trial data](./data/tsar2025_trialdata.jsonl) in our repository in the `data/` folder. The test data will be provided in the end of the term. You will run your approaches on the test set and submit your results as JSON line files in your repository.

**Important:** You have to submit and discuss at least one approach using Decoder-Encoder Models (e.g. RNNs, LSTMs, T5, BART) and one approach using Large Language Models.

## Schedule

| Lecture | Date       | Topic                     | Details                                                                    |
| :------ | :--------- | :------------------------ | :------------------------------------------------------------------------- |
| 1       | 03.12.2025 | Introduction & NLP Recap  | Organization, the challenge, word embeddings                               |
| 2       | 04.12.2025 | RNNs and LSTMs            | First sequence to sequence neural networks with memory                     |
| 3       | 10.12.2025 | Attentions & Transformers | Transformer architecture, parallalization in NLP, encoder & decoder blocks |
| 4       | 11.12.2025 | Transformer Based Models  | BERT, GPT, T5, BART                                                        |
| 5       | 17.12.2025 | Hackathon / Check-In      |                                                                            |
| 6       | 18.12.2025 | LLM Architecture          | From decoder models to LLMs, the LLM landscape (closed and open source)    |
| 7       | 07.01.2026 | LLM Engineering           | prompt engineering, retrieval augmented generation (RAG), evaluating LLMs  |
| 8       | 08.01.2026 | Hackathon / Check-In      |                                                                            |
| 9       | 14.01.2026 | LLM Shortcomings          | prompt injection, hallucinations, bias, designing LLM based applications   |
| 10      | 15.01.2026 | Presentations             | You present your work                                                      |

## Teams

- Team 2: Janina Sophie Janssen, Tim Vossmerbaeumer, Kristan Boettjer
- Team 3: Dongxin Wang, Tzu-Lun Yeh, Salman Tariq
- Team 4: Thuy Dieu Linh Nguyen, Samaneh Ilchi, Beyza Simsek
- Team 5: syed ghazanfar ali shah, Ksenia Blokhin, Ashwinkumar Ashok Kadam,

## Initial Setup

## Text Simplification Datasets

Overview: https://github.com/jantrienes/text-simplification-datasets

- ASSET (also by Fernando Alva-Manchego): https://github.com/facebookresearch/asset

## Further Reading & Ressources

- https://richardsieg.github.io/resources/
- [Stanford Class - Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
- https://e2eml.school/transformers.html
- https://jalammar.github.io/illustrated-transformer/
- Build a large language model (from Scratch) - Sebastian Raschka
- AI engineering: building applications with foundation models - Chip Huyen
- Hands-On Large Language Models - Jay Alammar, Maarten Grootendorst
