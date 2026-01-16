# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets==4.4.2",
#     "evaluate==0.4.6",
#     "huggingface-hub==0.36.0",
#     "marimo",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "scikit-learn==1.8.0",
#     "tabulate==0.9.0",
#     "torch==2.9.1",
#     "tqdm==4.67.1",
#     "transformers[torch]==4.57.3",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _(mo):
    mo.md(r"""
    ## Configuration
    """)
    return


@app.cell
def _():
    import os
    import random
    import torch
    import numpy as np
    import marimo as mo
    import json
    from tqdm import tqdm
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
        BartTokenizer,
        BartForConditionalGeneration,
        Seq2SeqTrainer, 
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq
    )
    from huggingface_hub import login

    # ================= Configuration =================
    HF_TOKEN = "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" 
    DATA_PATH = "data/asset/"

    # Output
    OUTPUT_DIR_T5 = "./results_t5_level"
    OUTPUT_DIR_BART = "./results_bart_level"

    # Hugging Face uoload
    HUB_MODEL_ID_T5 = "X1in/anlp_t5_level_aware"
    HUB_MODEL_ID_BART = "X1in/anlp_bart_level_aware"

    login(token=HF_TOKEN)
    return (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BartForConditionalGeneration,
        BartTokenizer,
        DATA_PATH,
        DataCollatorForSeq2Seq,
        Dataset,
        HUB_MODEL_ID_BART,
        HUB_MODEL_ID_T5,
        OUTPUT_DIR_BART,
        OUTPUT_DIR_T5,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        json,
        mo,
        os,
        random,
        torch,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Training T5 & Bart
    """)
    return


@app.cell
def _(Dataset, os, random):
    # ================= 1.1 load data =================

    def load_asset_data(asset_folder_path, split="valid"):
        """load row ASSET data"""
        src_sentences = []
        tgt_sentences = []

        print(f"Loading {split} data from: {asset_folder_path}")

        # iterate folders, parse ASSET filename structure
        if not os.path.exists(asset_folder_path):
            print(f"Error: Path {asset_folder_path} does not exist!")
            return Dataset.from_dict({"source": [], "target": []})

        for file_name in os.listdir(asset_folder_path):
            if file_name.endswith(f".{split}.orig"):
                base_name = file_name[: -(len(split) + 6)]
                orig_path = os.path.join(asset_folder_path, file_name)

                # read original (complex) sentences
                with open(orig_path, "r", encoding="utf-8") as f:
                    orig_sentences = [line.strip() for line in f if line.strip()]

                # find corresponding simplified files
                simp_files = [
                    os.path.join(asset_folder_path, f) for f in os.listdir(asset_folder_path)
                    if f.startswith(base_name) and f".{split}.simp." in f
                ]

                # create pairs: (Original Sentence) -> (Simplified Sentence)
                simp_sentences_list = []
                for simp_file in simp_files:
                    with open(simp_file, "r", encoding="utf-8") as f:
                        simp_sentences_list.append([line.strip() for line in f if line.strip()])

                for i, orig in enumerate(orig_sentences):
                    for simp_list in simp_sentences_list:
                        if i < len(simp_list):
                            src_sentences.append(orig)
                            tgt_sentences.append(simp_list[i])

        return Dataset.from_dict({"source": src_sentences, "target": tgt_sentences})

    # ================= 1.2 Prompt engineering in training with prefix =================
    def add_level_prompts(examples):
        """
        Injects control tokens (Prompts) into the input data.
        This teaches the model to recognize instructions like 'Simplify to A2'.
        """
        inputs = []
        targets = []

        prompts = [
            "Simplify to A2: ",
            "Simplify to B1: ",
            "Make this text A2 level: ",
            "Make this text B1 level: "
        ]

        for src, tgt in zip(examples["source"], examples["target"]):
            prefix = random.choice(prompts)
            inputs.append(prefix + src)
            targets.append(tgt)

        return {"input_text": inputs, "target_text": targets}
    return add_level_prompts, load_asset_data


@app.cell
def _(
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    DATA_PATH,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    add_level_prompts,
    load_asset_data,
    torch,
):
    # ================= 2. Training =================

    def run_training(model_type, output_dir, hub_id):
        print(f"\n{'='*20} Starting Fine-tuning: {model_type} {'='*20}")

        # a. load Pre-trained Base Model and Tokenizer
        # We start with weights from Google (T5) or Facebook (BART)
        if model_type == "t5":
            model_name = "google/flan-t5-base"
            TokenizerClass = AutoTokenizer
            ModelClass = AutoModelForSeq2SeqLM
        elif model_type == "bart":
            model_name = "facebook/bart-base"
            TokenizerClass = BartTokenizer
            ModelClass = BartForConditionalGeneration

        tokenizer = TokenizerClass.from_pretrained(model_name)
        model = ModelClass.from_pretrained(model_name)

        # b. prepare data
        dataset = load_asset_data(DATA_PATH, split="valid")
        # prompt
        dataset = dataset.map(add_level_prompts, batched=True, remove_columns=["source", "target"])
        # split test dataset
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

        # c. tokenization function
        def preprocess_function(examples):
            model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True, padding="max_length")
            labels = tokenizer(examples["target_text"], max_length=256, truncation=True, padding="max_length")

            # pad token -100
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] 
                for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_train = dataset["train"].map(preprocess_function, batched=True)
        tokenized_eval = dataset["test"].map(preprocess_function, batched=True)

        # d. training arguments
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=3e-5 if model_type == "bart" else 2e-5, # T5 usually requires lower LR
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10, # number of times the model sees the entire dataset
            weight_decay=0.01, # regularization to prevent overfitting
            save_total_limit=2,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(), # use mixed precision for faster training on GPU
            push_to_hub=True,
            hub_model_id=hub_id,
            report_to="none",
            eval_strategy="epoch",  
            save_strategy="epoch",

            load_best_model_at_end=True,
            logging_steps=100
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        )

        # d. start training & save
        trainer.train()

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        trainer.push_to_hub()

        print(f"✅ {model_type.upper()} Model Pushed to {hub_id}")
    return (run_training,)


@app.cell
def _(
    HUB_MODEL_ID_BART,
    HUB_MODEL_ID_T5,
    OUTPUT_DIR_BART,
    OUTPUT_DIR_T5,
    run_training,
):
    # ================= main programm =================

    if __name__ == "__main__":
        # 1. train t5
        run_training("t5", OUTPUT_DIR_T5, HUB_MODEL_ID_T5)

        # 2. train BART
        run_training("bart", OUTPUT_DIR_BART, HUB_MODEL_ID_BART)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Genaration
    """)
    return


@app.cell
def _(os):
    #MODEL_PATH = "X1in/anlp_t5_level_aware"  
    #MODEL_PATH = "./results_t5_level" 
    #MODEL_TYPE = "t5"                 
    MODEL_PATH = "./results_bart_level"  
    MODEL_TYPE = "bart"             

    TEST_FILE = "/__modal/volumes/vo-4yAAHadV8jqH6NeYKIpdsF/data/tsar2025_test.jsonl" 

    TEAM_NAME = "MyTeam"
    SUBMISSION_DIR = f"submissions/{TEAM_NAME}"
    OUTPUT_FILENAME = f"{MODEL_TYPE}_run.jsonl"
    OUTPUT_FILE = os.path.join(SUBMISSION_DIR, OUTPUT_FILENAME)
    return MODEL_PATH, MODEL_TYPE, OUTPUT_FILE, SUBMISSION_DIR, TEST_FILE


@app.cell
def _(
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    MODEL_PATH,
    MODEL_TYPE,
    OUTPUT_FILE,
    SUBMISSION_DIR,
    TEST_FILE,
    json,
    os,
    torch,
    tqdm,
):
    def run_generation():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading: {MODEL_PATH} ({device})...")

        if MODEL_TYPE == "t5":
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
        else:
            tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
            model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

        test_data = []
        print(f"reading test file: {TEST_FILE} ...")

        try:
            with open(TEST_FILE, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue 
                    try:
                        test_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"error in {i+1}. Content: {line[:50]}... Error: {e}")
        except Exception as e:
            print(f"other error: {e}")

        print(f"loaded {len(test_data)} test data")

        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        results = []

        for item in tqdm(test_data):
            # prompt
            target_cefr = item.get("target_cefr") 
            input_text = f"Simplify to {target_cefr}: {item['original']}"

            inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)

            # generation hyperparameters, how the model decodes probability into text.
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=128, # maximum length of the generated sentence
                    num_beams=5, # beam Search: explores 5 possible paths to find the best sentence
                    early_stopping=True,
                    no_repeat_ngram_size=0, # prevents repeating 3-word phrases
                    repetition_penalty=1.2, # [Crucial]: Penalizes repeating words to encourage rewriting/simplification, avoid the 'copy-paste' behavior common in seq2seq models."
                    length_penalty=0.8 # [Crucial]: Values < 1.0 encourage shorter (simpler) sentences
                )
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            results.append({
                "text_id": item["text_id"],
                "simplified": prediction
            })

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        print(f"\n prediction file: {OUTPUT_FILE}")

    # run
    run_generation()
    return


if __name__ == "__main__":
    app.run()
