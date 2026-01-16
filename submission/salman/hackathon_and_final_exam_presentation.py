# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate==1.12.0",
#     "datasets==4.5.0",
#     "huggingface-hub==0.36.0",
#     "ipython==9.8.0",
#     "ipywidgets==8.1.8",
#     "marimo",
#     "matplotlib==3.10.8",
#     "nltk==3.9.2",
#     "numpy==2.2.6",
#     "torch==2.9.1",
#     "tqdm==4.67.1",
#     "transformers[torch]==4.57.5",
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
def _():
    import marimo as mo
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainer, 
        Seq2SeqTrainingArguments,
        TrainerCallback,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset
    import matplotlib.pyplot as plt
    from pathlib import Path
    import warnings

    warnings.filterwarnings("ignore")
    return (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Dataset,
        Path,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        mo,
        plt,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Hackathon
    """)
    return


@app.cell
def _():
    from huggingface_hub import login

    # login(token="removed for pushing into gihtub")
    return


@app.cell
def _(mo):
    mo.md("""
    # T5 Fine-tuning for Text Simplification

    This notebook demonstrates fine-tuning T5-base on the ASSET text simplification dataset using HuggingFace Trainer.

    Reference: https://www.philschmid.de/fine-tune-flan-t5
    """)
    return


@app.cell
def load_asset_data(Dataset):
    def load_asset_data(asset_folder_path, split="valid"):
        """Load ASSET dataset from folder path and return HuggingFace Dataset"""
        import os

        src_sentences = []
        tgt_sentences = []

        print(f"Loading data from: {asset_folder_path}")

        for file_name in os.listdir(asset_folder_path):
            if file_name.endswith(f".{split}.orig"):
                base_name = file_name[: -(len(split) + 6)]  # Remove .split.orig extension
                orig_path = os.path.join(asset_folder_path, file_name)

                with open(orig_path, "r", encoding="utf-8") as f:
                    orig_sentences = [line.strip() for line in f if line.strip()]

                simp_files = [
                    os.path.join(asset_folder_path, simp_file_name)
                    for simp_file_name in os.listdir(asset_folder_path)
                    if simp_file_name.startswith(base_name) and f".{split}.simp." in simp_file_name
                ]

                simp_sentences_list = []
                for simp_file in simp_files:
                    with open(simp_file, "r", encoding="utf-8") as f:
                        simp_sentences = [line.strip() for line in f if line.strip()]
                        simp_sentences_list.append(simp_sentences)


                for i, orig_sentence in enumerate(orig_sentences):
                    for simp_sentences in simp_sentences_list:
                        if i < len(simp_sentences):
                            src_sentences.append("Simplify: " + orig_sentence)
                            tgt_sentences.append(simp_sentences[i])

        if not src_sentences or not tgt_sentences:
            print("Warning: No data loaded. Check the folder structure and file naming conventions.")

        # Create HuggingFace Dataset
        data_dict = {
            "source": src_sentences,
            "target": tgt_sentences,
        }

        return Dataset.from_dict(data_dict)
    return (load_asset_data,)


@app.cell
def _(AutoTokenizer, load_asset_data, mo):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)

    train_dataset_t5 = load_asset_data("data/asset/", split="valid")
    test_dataset_t5 = load_asset_data("data/asset/", split="test")

    mo.md(f"""
    ### Dataset Loaded!

    - **Training samples**: {len(train_dataset_t5):,}
    - **Test samples**: {len(test_dataset_t5):,}
    - **Model**: T5-base
    - **Tokenizer vocabulary size**: {len(tokenizer):,}

    **Example pair:**
    - Complex: {train_dataset_t5[0]['source']}
    - Simple: {train_dataset_t5[0]['target']}
    """)
    return test_dataset_t5, tokenizer, train_dataset_t5


@app.cell
def _(tokenizer):
    def preprocess_function(examples, max_length=256):
        """Tokenize the examples and prepare decoder inputs"""
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            examples["target"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    return (preprocess_function,)


@app.cell
def _(preprocess_function, test_dataset_t5, train_dataset_t5):
    tokenized_train = train_dataset_t5.map(
        preprocess_function, batched=True, remove_columns=["source", "target"]
    )
    tokenized_test = test_dataset_t5.map(
        preprocess_function, batched=True, remove_columns=["source", "target"]
    )

    print(f"Tokenized training samples: {len(tokenized_train):,}")
    print(f"Tokenized test samples: {len(tokenized_test):,}")
    return tokenized_test, tokenized_train


@app.cell
def _(AutoModelForSeq2SeqLM):
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return (model_t5,)


@app.cell
def _(DataCollatorForSeq2Seq, model_t5, tokenizer):
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model_t5,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    return (data_collator,)


@app.cell
def _(mo):
    mo.md("""
    ## Training Hyperparameters

    Configure the training parameters:
    """)
    return


@app.cell
def _(mo):
    learning_rate_t5 = mo.ui.slider(
        1e-5, 10e-5, value=5e-5, step=1e-6, label="Learning Rate"
    )
    n_epochs_t5 = mo.ui.slider(1, 10, value=5, step=1, label="Number of Epochs")
    batch_size_t5 = mo.ui.slider(8, 64, value=16, step=8, label="Batch Size")
    warmup_ratio_t5 = mo.ui.slider(0.0, 0.2, value=0.1, step=0.01, label="Warmup Ratio")

    mo.hstack([
        mo.vstack([learning_rate_t5, n_epochs_t5]),
        mo.vstack([batch_size_t5, warmup_ratio_t5])
    ])
    return batch_size_t5, learning_rate_t5, n_epochs_t5, warmup_ratio_t5


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start Training")
    train_button
    return (train_button,)


@app.cell
def _(
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    batch_size_t5,
    data_collator,
    learning_rate_t5,
    mo,
    model_t5,
    n_epochs_t5,
    plt,
    tokenized_test,
    tokenized_train,
    torch,
    train_button,
    warmup_ratio_t5,
):
    # Determine device
    if torch.cuda.is_available():
        device_t5 = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_t5 = torch.device("mps")
    else:
        device_t5 = torch.device("cpu")


    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=n_epochs_t5.value,
        per_device_train_batch_size=batch_size_t5.value,    
        per_device_eval_batch_size=batch_size_t5.value,
        warmup_ratio=warmup_ratio_t5.value,
        learning_rate=learning_rate_t5.value,
        fp16=False, # Overflows with fp16
        # weight_decay=0.01,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        # optim="adamw_torch_fused",
        # label_smoothing_factor=0.1,
        # max_grad_norm=0.5,
        use_mps_device=(device_t5.type == "mps"),

        # huggingface
        push_to_hub=True,                     
        hub_model_id="X1in/anlp_t5",

    )

    # Custom callback to track losses for plotting
    class LossCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.train_steps = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                    self.train_steps.append(state.global_step)
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    self.eval_steps.append(state.global_step)

    loss_callback = LossCallback()

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model_t5,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        callbacks=[loss_callback],
    )

    fig_t5, ax_t5 = plt.subplots(figsize=(10, 6))

    if train_button.value:
        print(f"Training on device: {device_t5}")
        print(f"Model parameters: {sum(p.numel() for p in model_t5.parameters()):,}\n")
        print("Training Configuration:")
        print("=" * 40)
        print(f"Learning Rate:    {learning_rate_t5.value}")
        print(f"Number of Epochs: {n_epochs_t5.value}")
        print(f"Batch Size:       {batch_size_t5.value}")
        print(f"Warmup Ratio:     {warmup_ratio_t5.value}")
        print("=" * 40 + "\n")

        # Train the model
        train_result = trainer.train()

        # Get final evaluation
        eval_result = trainer.evaluate()

        # Plot losses
        ax_t5.clear()
        if loss_callback.train_losses:
            ax_t5.plot(loss_callback.train_steps, loss_callback.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)

        if loss_callback.eval_losses:
            ax_t5.plot(loss_callback.eval_steps, loss_callback.eval_losses, 'r-', label='Eval Loss', linewidth=2, marker='o', markersize=6)

        ax_t5.set_xlabel("Training Steps", fontsize=12)
        ax_t5.set_ylabel("Loss", fontsize=12)
        ax_t5.set_title("T5 Training Progress", fontsize=14, fontweight="bold")
        ax_t5.legend(fontsize=11)
        ax_t5.grid(True, alpha=0.3)
        plt.tight_layout()

        mo.md(f"""
        ### Training Complete!

        **Final Training Loss**: {train_result.training_loss:.4f}

        **Final Evaluation Loss**: {eval_result['eval_loss']:.4f}

        **Training Time**: {train_result.metrics['train_runtime']:.2f} seconds ({train_result.metrics['train_runtime']/60:.1f} minutes)
        """)

        mo.output.append(ax_t5)
    return device_t5, trainer


@app.cell
def _(Path, tokenizer, trainer):


    save_path = Path("./fine_tuned_t5_simplification")
    save_path.mkdir(exist_ok=True)

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    return


@app.cell
def _(trainer):
    #push to huggingface
    trainer.push_to_hub()
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, tokenizer):
    #use the trained model in huggingface

    #from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "X1in/anlp_t5" 

    #load
    hub_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    hub_tokenizer = AutoTokenizer.from_pretrained(model_name)

    #start with example
    input_text = "April is the fourth month of the year in the Gregorian Calendar, and one of four months with a length of 30 days."
    input_ids = hub_tokenizer(input_text, return_tensors="pt").input_ids

    outputs = hub_model.generate(input_ids, max_length=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return


@app.cell
def _(mo):
    mo.md(rf"""
    ## Interactive Simplification

    Test the fine-tuned model and compare with pre-trained models:
    """)
    return


@app.cell
def _(mo):
    test_input = mo.ui.text_area(
        label="Enter a complex sentence to simplify:",
        value="April is the fourth month of the year in the Gregorian Calendar, and one of four months with a length of 30 days.",
        rows=3,
    )
    test_input
    return (test_input,)


@app.cell
def _(
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    device_t5,
    mo,
    model_t5,
    test_input,
    tokenizer,
    torch,
):
    def simplify_with_t5(text, model, tokenizer, device, max_length=128):
        """Simplify text using T5 model"""
        model.eval()

        input_text = "Simplify: " + text

        input_ids = tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_length=max_length, num_beams=4, early_stopping=True
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)


    input_text_test = test_input.value.strip()

    # Our fine-tuned model
    simplified_ours = simplify_with_t5(
        input_text_test, model_t5, tokenizer, device_t5
    )

    # Load pre-trained models
    tokenizer_eilamc = AutoTokenizer.from_pretrained(
        "eilamc14/t5-base-text-simplification", legacy=False
    )
    model_eilamc = AutoModelForSeq2SeqLM.from_pretrained(
        "eilamc14/t5-base-text-simplification"
    ).to(device_t5)

    # tokenizer_mrm = T5Tokenizer.from_pretrained(
    #     "mrm8488/t5-small-finetuned-text-simplification", legacy=False
    # )
    # model_mrm = T5ForConditionalGeneration.from_pretrained(
    #     "mrm8488/t5-small-finetuned-text-simplification"
    # ).to(device_t5)

    # Simplify with pre-trained models
    simplified_eilamc = simplify_with_t5(
        input_text_test, model_eilamc, tokenizer_eilamc, device_t5
    )
    # simplified_mrm = simplify_with_t5(
    #     input_text_test, model_mrm, tokenizer_mrm, device_t5
    # )

    mo.md(f"""
    ### Model Comparison

    **Original (Complex):**
    > {input_text_test}

    ---

    | Model | Simplified Output |
    |-------|-------------------|
    | **Our Fine-tuned T5-base** | {simplified_ours} |
    | **eilamc14/t5-base-text-simplification** | {simplified_eilamc} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Training Model V2 with with different hyperparameters
    """)
    return


@app.cell
def _():
    # login(token="removed for pushing into github")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Importing V1 model
    """)
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, device_t5, mo):
    # Load the pre-trained model from your colleague
    model_name_hubk = "X1in/anlp_t5" 

    mo.md(f"Loading pre-trained model: **{model_name_hubk}**...")

    hub_modelk = AutoModelForSeq2SeqLM.from_pretrained(model_name_hubk)
    hub_tokenizerk = AutoTokenizer.from_pretrained(model_name_hubk)

    # Move to device
    hub_modelk = hub_modelk.to(device_t5)

    mo.md("Model loaded successfully from Hugging Face Hub!")
    return hub_modelk, hub_tokenizerk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## V2 with improved hyperparameters
    """)
    return


@app.cell
def _(mo):
    mo.md("## Train Improved Model (Version 2)")

    # Improved defaults: Lower LR, More epochs
    lr_v2 = mo.ui.slider(1e-6, 1e-4, value=2e-5, step=1e-6, label="Learning Rate (v2)")
    epochs_v2 = mo.ui.slider(1, 20, value=10, step=1, label="Epochs (v2)")
    batch_v2 = mo.ui.slider(4, 32, value=16, step=4, label="Batch Size (v2)")

    train_button_v2 = mo.ui.run_button(label="Start Training V2")

    mo.vstack([
        mo.md("**Hyperparameters for Higher Accuracy**"),
        lr_v2, epochs_v2, batch_v2, train_button_v2
    ])
    return batch_v2, epochs_v2, lr_v2, train_button_v2


@app.cell
def _(
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    batch_v2,
    device_t5,
    epochs_v2,
    lr_v2,
    mo,
    plt,
    tokenized_test,
    tokenized_train,
    tokenizer,
    train_button_v2,
):
    # ... (Previous imports and model init code remains, paste this entire block)

    # 1. Initialize a FRESH V2 model
    model_v2_train = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model_v2_train = model_v2_train.to(device_t5)

    # 2. Training Arguments (Using V2 sliders)
    args_v2 = Seq2SeqTrainingArguments(
        output_dir="./results_v2",
        num_train_epochs=epochs_v2.value,
        per_device_train_batch_size=batch_v2.value,
        per_device_eval_batch_size=batch_v2.value,
        warmup_ratio=0.05, 
        learning_rate=lr_v2.value,
        fp16=False, 
        predict_with_generate=True,
        logging_dir="./logs_v2",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        push_to_hub=True,                     
        hub_model_id="X1in/anlp_t5_v2",
    )

    # 3. Custom Callback (v2 version)
    class LossCallbackV2(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.train_steps = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                    self.train_steps.append(state.global_step)
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    self.eval_steps.append(state.global_step)

    loss_callback_v2 = LossCallbackV2()

    # --- FIX: Define Collator Here ---
    collator_v2 = DataCollatorForSeq2Seq(
        tokenizer,
        model=model_v2_train,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # 4. Trainer Initialization (v2)
    trainer_v2 = Seq2SeqTrainer(
        model=model_v2_train,
        args=args_v2,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=collator_v2,  # Use the local collator
        callbacks=[loss_callback_v2],
    )

    # 5. Plotting Setup
    fig_v2, ax_v2 = plt.subplots(figsize=(10, 6))

    if train_button_v2.value:
        print(f"Starting V2 Training on {device_t5}...")
        print(f"Params: LR={lr_v2.value}, Epochs={epochs_v2.value}, Batch={batch_v2.value}")
        print("=" * 40)

        # Train
        train_result_v2 = trainer_v2.train()

        # Evaluate
        eval_result_v2 = trainer_v2.evaluate()

        # Plot Losses
        ax_v2.clear()
        if loss_callback_v2.train_losses:
            ax_v2.plot(loss_callback_v2.train_steps, loss_callback_v2.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
        if loss_callback_v2.eval_losses:
            ax_v2.plot(loss_callback_v2.eval_steps, loss_callback_v2.eval_losses, 'r-', label='Eval Loss', linewidth=2, marker='o', markersize=6)

        ax_v2.set_xlabel("Training Steps")
        ax_v2.set_ylabel("Loss")
        ax_v2.set_title("T5 (V2) Training Progress")
        ax_v2.legend()
        ax_v2.grid(True, alpha=0.3)
        plt.tight_layout()

        # Push to Hub
        trainer_v2.push_to_hub()

        mo.output.replace(mo.md(f"""
        ### V2 Training Complete!

        **Final Training Loss**: {train_result_v2.training_loss:.4f}
        **Final Eval Loss**: {eval_result_v2['eval_loss']:.4f}

        **Model pushed to**: [X1in/anlp_t5_v2](https://huggingface.co/X1in/anlp_t5_v2)
        """))
    else:
        mo.md("Click **Start Training V2** to begin.")

    # Display Plot
    mo.vstack([fig_v2])
    return (model_v2_train,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Importing V2 model after training
    """)
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, device_t5, mo):
    # Reload your trained V2 model from Hugging Face
    my_model_id = "X1in/anlp_t5_v2" 

    mo.md(f"📥 Reloading your trained model: **{my_model_id}**...")

    # Load Model
    model_v2_train = AutoModelForSeq2SeqLM.from_pretrained(my_model_id)
    model_v2_train = model_v2_train.to(device_t5)

    # Load Tokenizer using a NEW variable name
    tokenizer_v2 = AutoTokenizer.from_pretrained(my_model_id)

    mo.md("✅ **Model Reloaded!** You can now run the testing cell below.")
    return (model_v2_train,)


@app.cell
def _(mo):
    mo.md("### Test Your New Model (V2)")

    test_input_v2 = mo.ui.text_area(
        label="Enter a sentence to simplify (V2):",
        value="It was their moral right, they felt, to exploit the weak and the poor. Few of them thought their lives should change, even fewer believed it could.",
        rows=3,
    )
    test_input_v2
    return (test_input_v2,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Simplification

    Test the fine-tuned model and compare with pre-trained models:
    """)
    return


@app.cell
def _(
    device_t5,
    hub_modelk,
    hub_tokenizerk,
    mo,
    model_baseline_3,
    model_v2_train,
    test_input_v2,
    tokenizer,
    tokenizer_baseline_3,
    torch,
):
    # Helper to simplify (using local tokenizer for v2)
    def simplify_v2_final(text, model, tokenizer):
        if model is None: return "N/A"
        model.eval()
        inputs = tokenizer("Simplify: " + text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device_t5)
        with torch.no_grad():
            out = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    result_v2_output = None

    if test_input_v2.value:
        text_in_v2 = test_input_v2.value.strip()

        # 1. Your New V2 Model (from training cell)
        out_my_v2 = simplify_v2_final(text_in_v2, model_v2_train, tokenizer)

        # 2. Colleague's V1 Model (if loaded previously)
        try:
            out_colleague = simplify_v2_final(text_in_v2, hub_modelk, hub_tokenizerk)
        except NameError:
            out_colleague = "N/A (hub_model not loaded)"

        # 3. Baseline (if loaded previously)
        try:
            out_baseline = simplify_v2_final(text_in_v2, model_baseline_3, tokenizer_baseline_3)
        except NameError:
            out_baseline = "N/A (baseline not loaded)"

        result_v2_output = mo.md(f"""
        ### 🏆 Model Comparison Results

        **Input:**
        > {text_in_v2}

        ---

        | Model | Output |
        | :--- | :--- |
        | **✨ Your New Model (V2)** | **{out_my_v2}** |
        | Colleague (V1) | {out_colleague} |
        | Baseline (eilamc14) | {out_baseline} |
        """)

    else:
        result_v2_output = mo.md("Waiting for input...")

    result_v2_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BART Model
    """)
    return


@app.cell
def _():
    # Load BART model with new variable names
    from transformers import BartForConditionalGeneration, BartTokenizer
    # Load BART model and tokenizer
    print("Loading BART model and tokenizer...")
    tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-base")
    model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    print(f"✓ BART model loaded with {sum(p.numel() for p in model_bart.parameters()):,} parameters")

    try:
        from evaluate import load
        sari_metric = load("sari")
        bleu_metric = load("bleu")
        METRICS_AVAILABLE = True
        print("✓ SARI and BLEU metrics loaded")
    except:
        print("Note: Metrics not available, using loss only")
        METRICS_AVAILABLE = False
    return (
        BartForConditionalGeneration,
        BartTokenizer,
        METRICS_AVAILABLE,
        bleu_metric,
        load,
        model_bart,
        sari_metric,
        tokenizer_bart,
    )


@app.cell
def _(load_asset_data, mo, tokenizer_bart):
    # Load ASSET validation set and split it for training
    full_valid_dataset = load_asset_data("data/asset/", split="valid")

    # Split validation set: 80% train, 20% validation
    split_dataset = full_valid_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset_bart = split_dataset['train']
    valid_dataset_bart = split_dataset['test']

    # Load test set
    test_dataset_bart = load_asset_data("data/asset/", split="test")

    mo.md(f"""
    ### ASSET Dataset Loaded! ✓

    | Split | Samples |
    |-------|---------|
    | **Training** | {len(train_dataset_bart):,} |
    | **Validation** | {len(valid_dataset_bart):,} |
    | **Test** | {len(test_dataset_bart):,} |

    **Model**: BART-base  
    **Tokenizer vocabulary**: {len(tokenizer_bart):,}

    **Example training pair:**
    - **Complex**: {train_dataset_bart[0]['source'][:150]}...
    - **Simple**: {train_dataset_bart[0]['target'][:150]}...
    """)
    return test_dataset_bart, train_dataset_bart, valid_dataset_bart


@app.cell
def _(tokenizer_bart):
    def preprocess_function_bart(examples, max_length=128):
        """Tokenize for BART - no prefix needed"""
        model_inputs = tokenizer_bart(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer_bart(
            examples["target"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels["input_ids"] = [
            [(l if l != tokenizer_bart.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return (preprocess_function_bart,)


@app.cell
def _(
    preprocess_function_bart,
    test_dataset_bart,
    train_dataset_bart,
    valid_dataset_bart,
):
    # Tokenize ASSET datasets
    print("Tokenizing ASSET datasets...")
    tokenized_train_bart = train_dataset_bart.map(
        preprocess_function_bart, batched=True, remove_columns=["source", "target"]
    )
    tokenized_valid_bart = valid_dataset_bart.map(
        preprocess_function_bart, batched=True, remove_columns=["source", "target"]
    )
    tokenized_test_bart = test_dataset_bart.map(
        preprocess_function_bart, batched=True, remove_columns=["source", "target"]
    )

    print(f"✓ Tokenized training: {len(tokenized_train_bart):,}")
    print(f"✓ Tokenized validation: {len(tokenized_valid_bart):,}")
    print(f"✓ Tokenized test: {len(tokenized_test_bart):,}")
    return tokenized_test_bart, tokenized_train_bart, tokenized_valid_bart


@app.cell
def _(
    METRICS_AVAILABLE,
    bleu_metric,
    np,
    sari_metric,
    test_dataset_bart,
    tokenizer_bart,
):
    def compute_metrics_bart(eval_preds):
        """Compute SARI and BLEU for BART"""
        if not METRICS_AVAILABLE:
            return {}

        predictions, labels = eval_preds
        decoded_preds = tokenizer_bart.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer_bart.pad_token_id)
        decoded_labels = tokenizer_bart.batch_decode(labels, skip_special_tokens=True)

        # Get source sentences for SARI (use test dataset as reference)
        sources = [ex['source'] for ex in test_dataset_bart][:len(decoded_preds)]

        sari_score = sari_metric.compute(
            sources=sources,
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        bleu_score = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )

        return {
            "sari": sari_score["sari"],
            "bleu": bleu_score["bleu"] * 100
        }
    return (compute_metrics_bart,)


@app.cell
def _(mo):
    # BART hyperparameters with improved defaults
    learning_rate_bart = mo.ui.slider(
        1e-6, 1e-4, value=5e-5, step=1e-6, label="Learning Rate"  # Changed from 3e-5 to 5e-5
    )
    n_epochs_bart = mo.ui.slider(1, 15, value=12, step=1, label="Number of Epochs")  # Changed from 8 to 12
    batch_size_bart = mo.ui.slider(4, 32, value=8, step=4, label="Batch Size")
    grad_accum_bart = mo.ui.slider(1, 8, value=4, step=1, label="Gradient Accumulation")

    # Add new sliders for generation parameters
    length_penalty_bart = mo.ui.slider(
        0.5, 1.5, value=0.8, step=0.1, label="Length Penalty (lower = shorter outputs)"
    )
    num_beams_bart = mo.ui.slider(
        2, 8, value=4, step=1, label="Number of Beams"
    )

    mo.hstack([
        mo.vstack([learning_rate_bart, n_epochs_bart, length_penalty_bart]),
        mo.vstack([batch_size_bart, grad_accum_bart, num_beams_bart])
    ])
    return (
        batch_size_bart,
        grad_accum_bart,
        learning_rate_bart,
        length_penalty_bart,
        n_epochs_bart,
        num_beams_bart,
    )


@app.cell
def _(mo):
    train_button_bart = mo.ui.run_button(label="🚀 Start BART Training")
    train_button_bart
    return (train_button_bart,)


@app.cell
def _(
    DataCollatorForSeq2Seq,
    METRICS_AVAILABLE,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    batch_size_bart,
    compute_metrics_bart,
    grad_accum_bart,
    learning_rate_bart,
    model_bart,
    n_epochs_bart,
    num_beams_bart,
    tokenized_train_bart,
    tokenized_valid_bart,
    tokenizer_bart,
    torch,
):
    # Determine device
    if torch.cuda.is_available():
        device_bart = torch.device("cuda")
        print("Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device_bart = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device_bart = torch.device("cpu")
        print("Using CPU")

    # Data collator
    data_collator_bart = DataCollatorForSeq2Seq(
        tokenizer_bart,
        model=model_bart,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )


    # Improved training arguments
    training_args_bart = Seq2SeqTrainingArguments(
        output_dir="./results_bart",
        num_train_epochs=n_epochs_bart.value,
        per_device_train_batch_size=batch_size_bart.value,
        per_device_eval_batch_size=batch_size_bart.value,
        gradient_accumulation_steps=grad_accum_bart.value,

        # IMPROVED: Learning rate schedule
        learning_rate=learning_rate_bart.value,
        warmup_ratio=0.1,  # Changed from warmup_steps=500 to ratio
        lr_scheduler_type="cosine",  # Cosine annealing

        # IMPROVED: Regularization
        weight_decay=0.01,
        max_grad_norm=1.0,  # Gradient clipping

        # Precision
        fp16=torch.cuda.is_available(),

        # Generation parameters (length_penalty moved to generate() call)
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=num_beams_bart.value,
        # NOTE: generation_length_penalty removed - will be set in generate() instead

        # IMPROVED: Logging
        logging_dir="./logs_bart",
        logging_steps=50,
        logging_first_step=True,

        # IMPROVED: Evaluation
        eval_strategy="steps",
        eval_steps=250,

        # IMPROVED: Saving
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=True,

        # Metrics
        metric_for_best_model="sari" if METRICS_AVAILABLE else "eval_loss",
        greater_is_better=True if METRICS_AVAILABLE else False,

        # Other
        report_to="none",
        use_mps_device=(device_bart.type == "mps"),
        dataloader_num_workers=2,

        # Hub settings
        push_to_hub=True,
        hub_model_id="X1in/anlp_bart_simplification",
        hub_strategy="every_save",
    )


    # Enhanced callback with more metrics
    class EnhancedLossCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.eval_sari = []
            self.eval_bleu = []
            self.learning_rates = []
            self.train_steps = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                    self.train_steps.append(state.global_step)
                if "learning_rate" in logs:
                    self.learning_rates.append(logs["learning_rate"])
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    self.eval_steps.append(state.global_step)
                if "eval_sari" in logs:
                    self.eval_sari.append(logs["eval_sari"])
                if "eval_bleu" in logs:
                    self.eval_bleu.append(logs["eval_bleu"])

    loss_callback_bart = EnhancedLossCallback()

    # Initialize trainer
    trainer_bart = Seq2SeqTrainer(
        model=model_bart,
        args=training_args_bart,
        train_dataset=tokenized_train_bart,
        eval_dataset=tokenized_valid_bart,
        data_collator=data_collator_bart,
        tokenizer=tokenizer_bart,
        callbacks=[loss_callback_bart],
        compute_metrics=compute_metrics_bart if METRICS_AVAILABLE else None,
    )

    print("✓ Trainer initialized and ready")
    return device_bart, loss_callback_bart, trainer_bart


@app.cell
def _(
    METRICS_AVAILABLE,
    batch_size_bart,
    device_bart,
    grad_accum_bart,
    learning_rate_bart,
    length_penalty_bart,
    loss_callback_bart,
    mo,
    model_bart,
    n_epochs_bart,
    num_beams_bart,
    plt,
    tokenized_test_bart,
    train_button_bart,
    trainer_bart,
):
    # Enhanced training execution with better visualization
    fig_bart, axes_bart = plt.subplots(1, 2, figsize=(16, 6))

    if train_button_bart.value:
        print(f"\n{'='*60}")
        print(f"BART Training Configuration")
        print(f"{'='*60}")
        print(f"Device: {device_bart}")
        print(f"Model: BART-base")
        print(f"Parameters: {sum(p.numel() for p in model_bart.parameters()):,}")
        print(f"Learning Rate: {learning_rate_bart.value}")
        print(f"Epochs: {n_epochs_bart.value}")
        print(f"Batch Size: {batch_size_bart.value}")
        print(f"Gradient Accumulation: {grad_accum_bart.value}")
        print(f"Effective Batch Size: {batch_size_bart.value * grad_accum_bart.value}")
        print(f"Length Penalty: {length_penalty_bart.value}")
        print(f"Num Beams: {num_beams_bart.value}")
        print(f"{'='*60}\n")

        # Train the model
        train_result_bart = trainer_bart.train()

        # Evaluate on ASSET test set
        print("\n" + "="*60)
        print("Evaluating on ASSET test set...")
        print("="*60)
        test_results_bart = trainer_bart.evaluate(tokenized_test_bart)

        # Plot 1: Loss curves
        axes_bart[0].clear()
        if loss_callback_bart.train_losses:
            axes_bart[0].plot(loss_callback_bart.train_steps, loss_callback_bart.train_losses, 
                       'b-', label='Train Loss', linewidth=2, alpha=0.7)
        if loss_callback_bart.eval_losses:
            axes_bart[0].plot(loss_callback_bart.eval_steps, loss_callback_bart.eval_losses, 
                       'r-', label='Val Loss', linewidth=2, marker='o', markersize=4)
        axes_bart[0].set_xlabel("Training Steps", fontsize=12)
        axes_bart[0].set_ylabel("Loss", fontsize=12)
        axes_bart[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
        axes_bart[0].legend(fontsize=11)
        axes_bart[0].grid(True, alpha=0.3)

        # Plot 2: Metrics (SARI & BLEU) and Learning Rate
        axes_bart[1].clear()
        ax2 = axes_bart[1].twinx()

        if loss_callback_bart.eval_sari:
            axes_bart[1].plot(loss_callback_bart.eval_steps, loss_callback_bart.eval_sari, 
                       'g-', label='SARI', linewidth=2, marker='o', markersize=4)
        if loss_callback_bart.eval_bleu:
            axes_bart[1].plot(loss_callback_bart.eval_steps, loss_callback_bart.eval_bleu, 
                       'orange', label='BLEU', linewidth=2, marker='s', markersize=4)

        if loss_callback_bart.learning_rates:
            ax2.plot(loss_callback_bart.train_steps[:len(loss_callback_bart.learning_rates)], 
                    loss_callback_bart.learning_rates, 
                    'purple', label='Learning Rate', linewidth=1, alpha=0.5, linestyle='--')

        axes_bart[1].set_xlabel("Training Steps", fontsize=12)
        axes_bart[1].set_ylabel("SARI/BLEU Score", fontsize=12)
        ax2.set_ylabel("Learning Rate", fontsize=12, color='purple')
        axes_bart[1].set_title("Evaluation Metrics & LR Schedule", fontsize=14, fontweight="bold")
        axes_bart[1].legend(loc='upper left', fontsize=11)
        ax2.legend(loc='upper right', fontsize=11)
        axes_bart[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Display results
        results_text = f"""
        ## Training Complete! 🎉

        ### Training Summary
        - **Final Training Loss**: {train_result_bart.training_loss:.4f}
        - **Training Time**: {train_result_bart.metrics['train_runtime']:.1f}s ({train_result_bart.metrics['train_runtime']/60:.1f} min)
        - **Samples/Second**: {train_result_bart.metrics['train_samples_per_second']:.2f}
        - **Steps/Second**: {train_result_bart.metrics['train_steps_per_second']:.2f}

        ### Test Set Results
        - **Test Loss**: {test_results_bart['eval_loss']:.4f}
        """

        if METRICS_AVAILABLE and 'eval_sari' in test_results_bart:
            results_text += f"""
        - **SARI Score**: {test_results_bart['eval_sari']:.2f}
        - **BLEU Score**: {test_results_bart['eval_bleu']:.2f}

        ### Best Checkpoint
        - **Best SARI**: {max(loss_callback_bart.eval_sari) if loss_callback_bart.eval_sari else 'N/A'}
        - **Best Step**: {loss_callback_bart.eval_steps[loss_callback_bart.eval_sari.index(max(loss_callback_bart.eval_sari))] if loss_callback_bart.eval_sari else 'N/A'}
        """

        mo.md(results_text)
        mo.output.append(fig_bart)
    return (train_result_bart,)


@app.cell
def _(
    Path,
    batch_size_bart,
    grad_accum_bart,
    learning_rate_bart,
    length_penalty_bart,
    loss_callback_bart,
    mo,
    n_epochs_bart,
    num_beams_bart,
    tokenizer_bart,
    train_result_bart,
    trainer_bart,
):
    # Save the best BART model (automatically loaded by trainer after training)
    save_path_bart = Path("./fine_tuned_bart_simplification")
    save_path_bart.mkdir(exist_ok=True)

    # Save model and tokenizer
    print(f"\n{'='*60}")
    print("Saving fine-tuned model...")
    print(f"{'='*60}")

    trainer_bart.save_model(save_path_bart)
    tokenizer_bart.save_pretrained(save_path_bart)

    # Also save generation config with your optimized parameters
    from transformers import GenerationConfig

    generation_config = GenerationConfig(
        max_length=128,
        num_beams=num_beams_bart.value,
        length_penalty=length_penalty_bart.value,
        no_repeat_ngram_size=3,
        early_stopping=True,
        do_sample=False,
    )
    generation_config.save_pretrained(save_path_bart)

    print(f"✓ Model saved to {save_path_bart}")
    print(f"✓ Tokenizer saved to {save_path_bart}")
    print(f"✓ Generation config saved to {save_path_bart}")

    # Push to HuggingFace Hub
    print(f"\n{'='*60}")
    print("Pushing to HuggingFace Hub...")
    print(f"{'='*60}")

    try:
        trainer_bart.push_to_hub(
            commit_message=f"Fine-tuned BART for text simplification - SARI: {max(loss_callback_bart.eval_sari) if loss_callback_bart.eval_sari else 'N/A'}"
        )
        print("✓ Model pushed to HuggingFace Hub: X1in/anlp_bart_simplification")
        print(f"✓ View at: https://huggingface.co/X1in/anlp_bart_simplification")
    except Exception as e:
        print(f"⚠ Could not push to hub: {e}")
        print("  Make sure you're logged in: huggingface-cli login")

    # Save training metrics for later analysis
    import json

    metrics_path = save_path_bart / "training_metrics.json"
    metrics_data = {
        "final_train_loss": train_result_bart.training_loss if 'train_result_bart' in locals() else None,
        "best_eval_loss": min(loss_callback_bart.eval_losses) if loss_callback_bart.eval_losses else None,
        "best_sari": max(loss_callback_bart.eval_sari) if loss_callback_bart.eval_sari else None,
        "best_bleu": max(loss_callback_bart.eval_bleu) if loss_callback_bart.eval_bleu else None,
        "hyperparameters": {
            "learning_rate": learning_rate_bart.value,
            "num_epochs": n_epochs_bart.value,
            "batch_size": batch_size_bart.value,
            "grad_accumulation": grad_accum_bart.value,
            "length_penalty": length_penalty_bart.value,
            "num_beams": num_beams_bart.value,
        }
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"✓ Training metrics saved to {metrics_path}")

    mo.md(f"""
    ## 💾 Model Saved Successfully!

    ### Saved Files
    - **Model weights**: `{save_path_bart}`
    - **Tokenizer**: `{save_path_bart}`
    - **Generation config**: `{save_path_bart}/generation_config.json`
    - **Training metrics**: `{save_path_bart}/training_metrics.json`

    ### Load Later With:
    ```python
    from transformers import BartForConditionalGeneration, BartTokenizer

    model = BartForConditionalGeneration.from_pretrained("{save_path_bart}")
    tokenizer = BartTokenizer.from_pretrained("{save_path_bart}")
    ```

    Or from HuggingFace Hub:
    ```python
    model = BartForConditionalGeneration.from_pretrained("X1in/anlp_bart_simplification")
    tokenizer = BartTokenizer.from_pretrained("X1in/anlp_bart_simplification")
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load BART Model
    """)
    return


@app.cell
def _(BartForConditionalGeneration, BartTokenizer):
    model = BartForConditionalGeneration.from_pretrained("X1in/anlp_bart_simplification")
    tokenizer_Bart = BartTokenizer.from_pretrained("X1in/anlp_bart_simplification")
    return


@app.cell
def _(length_penalty_bart, num_beams_bart, torch):

    def simplify_with_bart(text, model, tokenizer_Bart, device, max_length=128):
        """Enhanced simplify function with better generation parameters"""
        model.eval()
        model.to(device)

        input_ids = tokenizer_Bart(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=max_length, 
                num_beams=num_beams_bart.value,  # Use slider value
                early_stopping=True,
                length_penalty=length_penalty_bart.value,  # Use slider value
                no_repeat_ngram_size=3,  # Prevent repetition
                do_sample=False,  # Deterministic for consistency
                temperature=1.0,  # Default temperature
            )

        return tokenizer_Bart.decode(outputs[0], skip_special_tokens=True)
    return (simplify_with_bart,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Testing with BART
    """)
    return


@app.cell
def _(device_bart, mo, model_bart, simplify_with_bart, tokenizer_bart):
    # Test multiple examples to see improvement
    test_examples = [
        "The committee’s decision, though ostensibly unanimous, concealed a profound divergence of underlying priorities.",
        "The utilization of sophisticated methodologies enables researchers to ascertain the veracity of hypothetical assertions.",
        "Notwithstanding the considerable efforts expended by the administration, the amelioration of the situation remains elusive.",
    ]

    comparison_results = []
    for example in test_examples[:3]:  # Test first 3
        simplified = simplify_with_bart(example, model_bart, tokenizer_bart, device_bart)
        comparison_results.append({
            "original": example,
            "simplified": simplified,
            "compression": len(simplified.split()) / len(example.split())
        })

    comparison_md = "### 🔍 Batch Simplification Test\n\n"
    for i, result in enumerate(comparison_results, 1):
        comparison_md += f"""
    **Example {i}:**
    - **Original** ({len(result['original'].split())} words): {result['original']}
    - **Simplified** ({len(result['simplified'].split())} words): {result['simplified']}
    - **Compression Ratio**: {result['compression']:.2%}

    ---
    """

    mo.md(comparison_md)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Failed Approach: Exam Presentation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Imports and Environment Setup
    """)
    return


@app.cell
def _():
    # import marimo as mo
    # import torch
    # from transformers import (
    #     AutoModelForSeq2SeqLM,
    #     AutoTokenizer,
    #     Seq2SeqTrainer, 
    #     Seq2SeqTrainingArguments,
    #     TrainerCallback,
    #     DataCollatorForSeq2Seq,
    # )
    # from datasets import Dataset
    # import matplotlib.pyplot as plt
    # from pathlib import Path
    import numpy as np
    import random

    # import warnings

    # warnings.filterwarnings("ignore")
    return np, random


@app.cell
def _(mo):
    mo.md(r"""
    ### Logging into hugging face
    """)
    return


@app.cell
def _(mo):

    # login(token="removed for pushing into github")

    mo.md("""
    ✓ All imports loaded successfully
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Evaluation Metrics
    This block sets up automatic evaluation metrics for level-aware text generation.
    SARI and BLEU are used to measure generation quality, while a custom implementation
    of the Flesch–Kincaid Grade Level (FKGL) is used to assess readability and text
    complexity.
    """)
    return


@app.cell
def _(load, mo):
    mo.md("## 📊 Evaluation Metrics (Level-Aware)")

    # Load SARI and BLEU metrics
    # SARI: measures quality of text simplification (how well words are added, deleted, or kept)
    # BLEU: measures similarity to reference text (precision-based n-gram overlap)
    try:
        sari_metric_level = load("sari")
        bleu_metric_level = load("bleu")
        METRICS_AVAILABLE_LEVEL = True
        mo.md("✓ Metrics loaded for level-aware training")
    except Exception as e:
        mo.md(f"⚠️ Metrics not available: {e}")
        METRICS_AVAILABLE_LEVEL = False

    # FKGL calculator: readability score in US grade levels
    def calculate_fkgl_level(text):
        """Calculate Flesch-Kincaid Grade Level - Level-aware version"""
        import re
        if not text or len(text.strip()) == 0:
            return 0

        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        words_list = text.split()
        words = max(1, len(words_list))
        syllables = sum([max(1, len(re.findall(r'[aeiouAEIOU]+', word))) for word in words_list])

        fkgl = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        return max(0, fkgl)

    mo.md("✓ Readability metrics ready (level-aware)")
    return (
        METRICS_AVAILABLE_LEVEL,
        bleu_metric_level,
        calculate_fkgl_level,
        sari_metric_level,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Data Loading
    """)
    return


@app.cell
def _(Dataset, mo):
    mo.md("## 📁 Data Loading (Level-Aware)")

    def load_asset_data_level(asset_folder_path, split="valid"):
        """Load ASSET dataset - Level-aware version"""
        import os
        src_sentences = []
        tgt_sentences = []

        print(f"Loading {split} data from: {asset_folder_path}")

        for file_name in os.listdir(asset_folder_path):
            if file_name.endswith(f".{split}.orig"):
                base_name = file_name[: -(len(split) + 6)]
                orig_path = os.path.join(asset_folder_path, file_name)

                with open(orig_path, "r", encoding="utf-8") as f:
                    orig_sentences = [line.strip() for line in f if line.strip()]

                simp_files = [
                    os.path.join(asset_folder_path, simp_file_name)
                    for simp_file_name in os.listdir(asset_folder_path)
                    if simp_file_name.startswith(base_name) and f".{split}.simp." in simp_file_name
                ]

                simp_sentences_list = []
                for simp_file in simp_files:
                    with open(simp_file, "r", encoding="utf-8") as f:
                        simp_sentences = [line.strip() for line in f if line.strip()]
                        simp_sentences_list.append(simp_sentences)

                for i, orig_sentence in enumerate(orig_sentences):
                    for simp_sentences in simp_sentences_list:
                        if i < len(simp_sentences):
                            src_sentences.append(orig_sentence)
                            tgt_sentences.append(simp_sentences[i])

        if not src_sentences or not tgt_sentences:
            print("⚠️ Warning: No data loaded.")

        return Dataset.from_dict({
            "source": src_sentences,
            "target": tgt_sentences,
        })

    mo.md("✓ Data loading function ready (level-aware)")
    return (load_asset_data_level,)


@app.cell
def _(load_asset_data_level, mo):
    mo.md("## 📊 Loading ASSET Dataset (Level-Aware)")

    # Load ASSET data
    dataset_valid_level = load_asset_data_level("data/asset/", split="valid")
    dataset_test_level = load_asset_data_level("data/asset/", split="test")

    # Split valid into train/validation (80/20)
    split_data_level = dataset_valid_level.train_test_split(test_size=0.2, seed=42)
    dataset_train_original_level = split_data_level['train']
    dataset_validation_level = split_data_level['test']

    mo.md(f"""
    ### ASSET Dataset Loaded (Level-Aware) ✓

    | Split | Samples |
    |-------|---------|
    | **Training** | {len(dataset_train_original_level):,} |
    | **Validation** | {len(dataset_validation_level):,} |
    | **Test** | {len(dataset_test_level):,} |
    """)
    return (
        dataset_test_level,
        dataset_train_original_level,
        dataset_validation_level,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## ASSET Dataset Loaded (Level-Aware) ✓

    | Split       | Samples |
    |------------|---------|
    | Training   | 16,000  |
    | Validation | 4,000   |
    | Test       | 3,590   |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Dataset Analysis
    This block performs exploratory analysis of the dataset by measuring sentence
    length, readability (Flesch–Kincaid Grade Level), compression ratio, and FKGL
    reduction.
    """)
    return


@app.cell
def _(calculate_fkgl_level, dataset_train_original_level, mo, np):

    mo.md("## 🔍 Dataset Analysis Function")

    def analyze_dataset_level(dataset, name="Dataset"):
        """Analyze dataset - Level-aware version"""
        num_samples = len(dataset)

        source_lengths = [len(ex['source'].split()) for ex in dataset]
        target_lengths = [len(ex['target'].split()) for ex in dataset]

        source_fkgl = [calculate_fkgl_level(ex['source']) for ex in dataset]
        target_fkgl = [calculate_fkgl_level(ex['target']) for ex in dataset]

        compression_ratios = [t/s if s > 0 else 0 for s, t in zip(source_lengths, target_lengths)]
        fkgl_reductions = [s - t for s, t in zip(source_fkgl, target_fkgl)]

        return {
            'name': name,
            'samples': num_samples,
            'avg_source_words': np.mean(source_lengths),
            'avg_target_words': np.mean(target_lengths),
            'avg_source_fkgl': np.mean(source_fkgl),
            'avg_target_fkgl': np.mean(target_fkgl),
            'avg_compression': np.mean(compression_ratios),
            'avg_fkgl_reduction': np.mean(fkgl_reductions),
        }

    train_stats_level = analyze_dataset_level(dataset_train_original_level, "Training")

    mo.md(f"""
    ### Dataset Statistics (Level-Aware)

    | Metric | Value |
    |--------|-------|
    | **Samples** | {train_stats_level['samples']:,} |
    | **Avg Source Words** | {train_stats_level['avg_source_words']:.1f} |
    | **Avg Target Words** | {train_stats_level['avg_target_words']:.1f} |
    | **Source FKGL** | {train_stats_level['avg_source_fkgl']:.1f} |
    | **Target FKGL** | {train_stats_level['avg_target_fkgl']:.1f} |
    | **FKGL Reduction** | {train_stats_level['avg_fkgl_reduction']:.1f} |
    """)
    return analyze_dataset_level, train_stats_level


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset Statistics (Level-Aware)

    | Metric             | Value |
    |--------------------|-------|
    | Samples            | 16,000 |
    | Avg Source Words   | 19.2   |
    | Avg Target Words   | 16.8   |
    | Source FKGL        | 12.2   |
    | Target FKGL        | 9.3    |
    | FKGL Reduction     | 2.8    |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### NLP Concepts Used (NLTK & Augmentation)
    NLTK provides classical NLP tools such as tokenization, POS tagging, and lexical
    resources like WordNet. POS tagging assigns grammatical roles to words, enabling
    POS-aware synonym replacement using WordNet. This controlled data augmentation
    introduces lexical diversity while preserving semantic meaning and grammatical
    structure, improving model robustness in level-aware text simplification.
    """)
    return


@app.cell
def _(mo, random):
    mo.md("## 🔄 NLTK Setup for Augmentation")

    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    from nltk.corpus import wordnet
    from nltk import word_tokenize, pos_tag

    def get_wordnet_pos_level(treebank_tag):
        """Convert POS tags - Level version"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        return None

    def get_synonyms_level(word, pos=None):
        """Get synonyms - Level version"""
        synonyms = []
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and '-' not in synonym:
                    synonyms.append(synonym)
        return list(set(synonyms))

    def synonym_replacement_level(text, n=3):
        """Replace n words with synonyms - Level version"""
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words)

            new_words = words.copy()
            replaceable = []

            for i, (word, tag) in enumerate(pos_tags):
                if len(word) > 4:
                    wn_pos = get_wordnet_pos_level(tag)
                    if wn_pos:
                        syns = get_synonyms_level(word, wn_pos)
                        if syns:
                            replaceable.append((i, word, syns))

            if len(replaceable) > n:
                replaceable = random.sample(replaceable, n)

            for idx, original, syns in replaceable:
                new_words[idx] = random.choice(syns)

            return ' '.join(new_words)
        except:
            return text

    mo.md("✓ Augmentation functions ready (level-aware)")
    return (synonym_replacement_level,)


@app.cell
def _(mo):
    mo.md(r"""
    ### CEFR Level-Aware Data Augmentation
    This block implements instruction-based data augmentation using CEFR proficiency
    levels. Each source sentence is augmented with level-specific simplification
    prompts and synonym-based variations, enabling the model to learn controllable
    text simplification conditioned on explicit readability targets.
    """)
    return


@app.cell
def _(
    Dataset,
    analyze_dataset_level,
    dataset_train_original_level,
    mo,
    random,
    synonym_replacement_level,
    train_stats_level,
):
    mo.md("## 🚀 Level-Aware Data Augmentation")

    def augment_example_cefr(example, num_augmentations=4):
        """Augment with CEFR level-aware prompts"""
        source_text = example['source']
        target_text = example['target']

        augmented = []

        # CEFR level-aware instruction templates
        level_templates_cefr = [
            "Simplify this C2 level text to A2 level: {text}",
            "Simplify this C1 level text to A2 level: {text}",
            "Simplify this B2 level text to B1 level: {text}",
            "Simplify this C2 level text to B1 level: {text}",
            "Simplify this B2 level text to A2 level: {text}",
            "Simplify this C1 level text to B1 level: {text}",
            "Simplify: {text}",  # Keep some generic
        ]

        # Original with level-aware instruction
        template = random.choice(level_templates_cefr)
        augmented.append({
            'source': template.format(text=source_text),
            'target': target_text
        })

        # Synonym variations with level-aware prompts
        for _ in range(num_augmentations - 1):
            aug_source = synonym_replacement_level(source_text, n=random.randint(2, 4))
            template = random.choice(level_templates_cefr)
            augmented.append({
                'source': template.format(text=aug_source),
                'target': target_text
            })

        return augmented

    def augment_dataset_cefr(dataset, augmentation_factor=4):
        """Augment entire dataset with CEFR level prompts"""
        all_augmented = []

        total = len(dataset)
        for i, example in enumerate(dataset):
            augmented_examples = augment_example_cefr(example, num_augmentations=augmentation_factor)
            all_augmented.extend(augmented_examples)

            if (i + 1) % 500 == 0:
                print(f"Augmented {i+1}/{total} examples with CEFR prompts...")

        return Dataset.from_dict({
            'source': [d['source'] for d in all_augmented],
            'target': [d['target'] for d in all_augmented]
        })

    # Apply CEFR level-aware augmentation
    mo.md("Starting CEFR level-aware augmentation (this may take a few minutes)...")
    dataset_train_augmented_cefr = augment_dataset_cefr(
        dataset_train_original_level, 
        augmentation_factor=4
    )

    aug_stats_cefr = analyze_dataset_level(dataset_train_augmented_cefr, "CEFR Augmented")

    mo.md(f"""
    ### CEFR Level-Aware Augmentation Complete! ✓

    | Metric | Original | Augmented | Growth |
    |--------|----------|-----------|--------|
    | **Samples** | {train_stats_level['samples']:,} | {aug_stats_cefr['samples']:,} | **{aug_stats_cefr['samples']/train_stats_level['samples']:.1f}x** |

    **Level-Aware Strategy:**
    - ✅ C2 → A2 (Proficiency to Elementary)
    - ✅ C1 → A2 (Advanced to Elementary)
    - ✅ B2 → B1 (Upper-Int to Intermediate)
    - ✅ C2 → B1 (Proficiency to Intermediate)
    - ✅ B2 → A2 (Upper-Int to Elementary)
    - ✅ C1 → B1 (Advanced to Intermediate)
    - ✅ Generic simplification (baseline)

    **Model will learn to:**
    - Understand CEFR level specifications
    - Target specific reading levels
    - Adjust complexity based on prompts
    """)
    return (dataset_train_augmented_cefr,)


@app.cell
def _(mo):
    mo.md(r"""
    ## CEFR Level-Aware Augmentation Complete! ✓

    | Metric  | Original | Augmented | Growth |
    |---------|----------|-----------|--------|
    | Samples | 16,000   | 64,000    | 4.0x   |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Loading (Level-Aware)
    This block loads the pre-trained Flan-T5 model and its tokenizer for sequence-to-sequence
    tasks. The model is moved to the appropriate device (GPU, MPS, or CPU) for computation
    and prepared for CEFR level-aware training using instruction-based prompts.
    """)
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, device_final, mo, torch):
    mo.md("## 🤖 Loading Model (Level-Aware)")

    # Load T5 model
    tokenizer_level = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)
    model_level = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    # Reuse device
    try:
        device_level = device_final
    except:
        if torch.cuda.is_available():
            device_level = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_level = torch.device("mps")
        else:
            device_level = torch.device("cpu")

    model_level = model_level.to(device_level)

    mo.md(f"""
    ### Model Loaded (Level-Aware) ✓

    - **Model**: google/flan-t5-base
    - **Parameters**: {sum(p.numel() for p in model_level.parameters()):,}
    - **Device**: {device_level}
    - **Training**: CEFR level-aware
    """)
    return device_level, model_level, tokenizer_level


@app.cell
def _(mo):
    mo.md(r"""
    ### Preprocessing (Level-Aware)
    This function tokenizes the source and target text using the Flan-T5 tokenizer,
    applies truncation and padding to ensure uniform sequence lengths, and converts
    padding tokens in labels to -100 so they are ignored in the loss function. The
    output is ready for Seq2Seq training with attention masks and labels.
    """)
    return


@app.cell
def _(mo, tokenizer_level):
    mo.md("## ⚙️ Preprocessing (Level-Aware)")

    def preprocess_function_level(examples, max_length=256):
        """Tokenize examples - Level version"""
        model_inputs = tokenizer_level(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer_level(
            examples["target"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        labels["input_ids"] = [
            [(l if l != tokenizer_level.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    mo.md("✓ Preprocessing function ready (level-aware)")
    return (preprocess_function_level,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Tokenizing Datasets (Level-Aware)
    This block applies the preprocessing function to training, validation, and test datasets.
    Training data contains CEFR-level instruction prompts and synonym-augmented text.
    Validation and test sets are prefixed with 'Simplify:' to maintain instruction style.
    """)
    return


@app.cell
def _(
    Dataset,
    dataset_test_level,
    dataset_train_augmented_cefr,
    dataset_validation_level,
    mo,
    preprocess_function_level,
):
    mo.md("## 🔄 Tokenizing Datasets (Level-Aware)")

    print("Tokenizing CEFR augmented training data...")
    tokenized_train_cefr = dataset_train_augmented_cefr.map(
        preprocess_function_level, 
        batched=True, 
        remove_columns=["source", "target"]
    )

    print("Tokenizing validation data...")
    dataset_validation_with_prefix_level = Dataset.from_dict({
        'source': ["Simplify: " + ex['source'] for ex in dataset_validation_level],
        'target': [ex['target'] for ex in dataset_validation_level]
    })

    tokenized_validation_cefr = dataset_validation_with_prefix_level.map(
        preprocess_function_level,
        batched=True,
        remove_columns=["source", "target"]
    )

    print("Tokenizing test data...")
    dataset_test_with_prefix_level = Dataset.from_dict({
        'source': ["Simplify: " + ex['source'] for ex in dataset_test_level],
        'target': [ex['target'] for ex in dataset_test_level]
    })

    tokenized_test_cefr = dataset_test_with_prefix_level.map(
        preprocess_function_level,
        batched=True,
        remove_columns=["source", "target"]
    )

    mo.md(f"""
    ### Tokenization Complete (Level-Aware) ✓

    | Split | Samples |
    |-------|---------|
    | **Training (CEFR)** | {len(tokenized_train_cefr):,} |
    | **Validation** | {len(tokenized_validation_cefr):,} |
    | **Test** | {len(tokenized_test_cefr):,} |
    """)
    return tokenized_test_cefr, tokenized_train_cefr, tokenized_validation_cefr


@app.cell
def _(mo):
    mo.md(r"""
    ### Metrics Function (Level-Aware)
    This function computes evaluation metrics for CEFR-level simplification:
    - **SARI** evaluates simplification edits against references and sources.
    - **BLEU** measures n-gram overlap with references.
    - **FKGL reduction** quantifies how much the text is simplified in readability.
    Predictions and labels are decoded from token IDs into text before metric calculation.
    """)
    return


@app.cell
def _(
    METRICS_AVAILABLE_LEVEL,
    bleu_metric_level,
    calculate_fkgl_level,
    dataset_test_level,
    mo,
    np,
    sari_metric_level,
    tokenizer_level,
):
    mo.md("## 📊 Metrics Function (Level-Aware)")

    def compute_metrics_cefr(eval_preds):
        """Compute metrics - CEFR version"""
        if not METRICS_AVAILABLE_LEVEL:
            return {}

        predictions, labels = eval_preds

        decoded_preds = tokenizer_level.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer_level.pad_token_id)
        decoded_labels = tokenizer_level.batch_decode(labels, skip_special_tokens=True)

        sources = [ex['source'] for ex in dataset_test_level[:len(decoded_preds)]]

        sari_score = sari_metric_level.compute(
            sources=sources,
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )

        bleu_score = bleu_metric_level.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )

        source_fkgl = np.mean([calculate_fkgl_level(s) for s in sources])
        pred_fkgl = np.mean([calculate_fkgl_level(p) for p in decoded_preds])

        return {
            "sari": sari_score["sari"],
            "bleu": bleu_score["bleu"] * 100,
            "fkgl_reduction": source_fkgl - pred_fkgl,
            "output_fkgl": pred_fkgl
        }

    mo.md("✓ Metrics function ready (level-aware)")
    return (compute_metrics_cefr,)


@app.cell
def _(mo):
    mo.md("## ⚙️ Training Hyperparameters (Level-Aware)")

    learning_rate_slider_cefr = mo.ui.slider(
        1e-6, 1e-4, value=2e-5, step=1e-6, label="Learning Rate (CEFR)"
    )
    epochs_slider_cefr = mo.ui.slider(1, 20, value=7, step=1, label="Epochs (CEFR)")
    batch_size_slider_cefr = mo.ui.slider(4, 32, value=16, step=4, label="Batch Size (CEFR)")
    grad_accum_slider_cefr = mo.ui.slider(1, 8, value=2, step=1, label="Gradient Accumulation (CEFR)")

    mo.vstack([
        mo.md("**CEFR Level-Aware Training Parameters:**"),
        mo.hstack([learning_rate_slider_cefr, epochs_slider_cefr]),
        mo.hstack([batch_size_slider_cefr, grad_accum_slider_cefr])
    ])
    return (
        batch_size_slider_cefr,
        epochs_slider_cefr,
        grad_accum_slider_cefr,
        learning_rate_slider_cefr,
    )


@app.cell
def _(mo):
    train_button_cefr = mo.ui.run_button(label="🚀 Start CEFR Level-Aware Training")
    train_button_cefr
    return (train_button_cefr,)


@app.cell
def _(mo):
    mo.md(r"""
    ### CEFR Level-Aware Trainer Configuration
    - **Data collator:** handles batch padding and label alignment
    - **TrainingArguments:** configure learning rate, batch size, epochs, evaluation, checkpointing
    - **Loss callback:** records training/eval loss and SARI score
    - **Seq2SeqTrainer:** integrates model, datasets, collator, tokenizer, callbacks, and metrics
    """)
    return


@app.cell
def _(
    DataCollatorForSeq2Seq,
    METRICS_AVAILABLE_LEVEL,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    batch_size_slider_cefr,
    compute_metrics_cefr,
    device_level,
    epochs_slider_cefr,
    grad_accum_slider_cefr,
    learning_rate_slider_cefr,
    mo,
    model_level,
    tokenized_train_cefr,
    tokenized_validation_cefr,
    tokenizer_level,
):
    mo.md("## 🎯 Training Configuration (CEFR)")

    # Data collator: prepares batches, ensures padding, and aligns labels for loss computation.
    data_collator_cefr = DataCollatorForSeq2Seq(
        tokenizer_level,
        model=model_level,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # Training arguments
    training_args_cefr = Seq2SeqTrainingArguments(
        output_dir="./results_cefr_level_aware",
        disable_tqdm=True,
        num_train_epochs=epochs_slider_cefr.value,
        per_device_train_batch_size=batch_size_slider_cefr.value,
        per_device_eval_batch_size=batch_size_slider_cefr.value,
        gradient_accumulation_steps=grad_accum_slider_cefr.value,
        learning_rate=learning_rate_slider_cefr.value,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=False,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        logging_dir="./logs_cefr_level_aware",
        logging_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="sari" if METRICS_AVAILABLE_LEVEL else "eval_loss",
        greater_is_better=True if METRICS_AVAILABLE_LEVEL else False,
        report_to="none",
        use_mps_device=(device_level.type == "mps"),
        push_to_hub=True,
        hub_model_id="X1in/anlp_t5_cefr_level_aware",
    )

    # Loss callback: define a callback to record losses and SARI scores at each step for monitoring and visualization.
    class LossCallbackCEFR(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.eval_sari = []
            self.train_steps = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                    self.train_steps.append(state.global_step)
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    self.eval_steps.append(state.global_step)
                if "eval_sari" in logs:
                    self.eval_sari.append(logs["eval_sari"])

    loss_callback_cefr = LossCallbackCEFR()

    # Initialize trainer: wraps all components together, handling training, evaluation, and metric computation seamlessly.
    trainer_cefr = Seq2SeqTrainer(
        model=model_level,
        args=training_args_cefr,
        train_dataset=tokenized_train_cefr,
        eval_dataset=tokenized_validation_cefr,
        data_collator=data_collator_cefr,
        tokenizer=tokenizer_level,
        callbacks=[loss_callback_cefr],
        compute_metrics=compute_metrics_cefr if METRICS_AVAILABLE_LEVEL else None,
    )

    mo.md("✓ Trainer initialized for CEFR level-aware training")
    return loss_callback_cefr, trainer_cefr


@app.cell
def _(mo):
    mo.md(r"""
    ### CEFR Level-Aware Training Execution
    - Training begins when the user clicks the **Start Training** button.
    - **Trainer.train()** updates model weights using tokenized CEFR-augmented data.
    - **Trainer.evaluate()** computes metrics (SARI, BLEU, FKGL) on the test set.
    - **Loss and SARI plots** visualize model learning progress.
    - Fine-tuned model and tokenizer are saved locally and can be pushed to Hugging Face Hub.
    """)
    return


@app.cell
def _(
    METRICS_AVAILABLE_LEVEL,
    Path,
    batch_size_slider_cefr,
    device_level,
    epochs_slider_cefr,
    grad_accum_slider_cefr,
    learning_rate_slider_cefr,
    loss_callback_cefr,
    mo,
    plt,
    tokenized_test_cefr,
    tokenized_train_cefr,
    tokenizer_level,
    train_button_cefr,
    trainer_cefr,
):
    mo.md("## 🏋️ Training Model (CEFR Level-Aware)")

    fig_cefr, axes_cefr = plt.subplots(1, 2, figsize=(14, 5))

    if train_button_cefr.value:
        print(f"\n{'='*70}")
        print(f"{'CEFR LEVEL-AWARE TRAINING':^70}")
        print(f"{'='*70}")
        print(f"Device: {device_level}")
        print(f"Training samples: {len(tokenized_train_cefr):,}")
        print(f"CEFR Levels: C2→A2, C1→A2, B2→B1, C2→B1, B2→A2, C1→B1")
        print(f"Learning Rate: {learning_rate_slider_cefr.value}")
        print(f"Epochs: {epochs_slider_cefr.value}")
        print(f"Batch Size: {batch_size_slider_cefr.value}")
        print(f"Gradient Accumulation: {grad_accum_slider_cefr.value}")
        print(f"Effective Batch: {batch_size_slider_cefr.value * grad_accum_slider_cefr.value}")
        print(f"{'='*70}\n")

        # Train
        train_result_cefr = trainer_cefr.train()

        # Evaluate: generates predicted simplifications and computes SARI, BLEU, and FKGL
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET (CEFR)")
        print("="*70)
        test_results_cefr = trainer_cefr.evaluate(tokenized_test_cefr)

        # Plot
        axes_cefr[0].clear()
        axes_cefr[1].clear()

        # Loss plot
        if loss_callback_cefr.train_losses:
            axes_cefr[0].plot(loss_callback_cefr.train_steps, loss_callback_cefr.train_losses,
                        'b-', label='Train Loss', linewidth=2, alpha=0.7)
        if loss_callback_cefr.eval_losses:
            axes_cefr[0].plot(loss_callback_cefr.eval_steps, loss_callback_cefr.eval_losses,
                        'r-', label='Val Loss', linewidth=2, marker='o', markersize=4)
        axes_cefr[0].set_xlabel("Steps", fontsize=11)
        axes_cefr[0].set_ylabel("Loss", fontsize=11)
        axes_cefr[0].set_title("Loss Curves (CEFR)", fontsize=12, fontweight="bold")
        axes_cefr[0].legend()
        axes_cefr[0].grid(True, alpha=0.3)

        # SARI plot
        if loss_callback_cefr.eval_sari:
            axes_cefr[1].plot(loss_callback_cefr.eval_steps, loss_callback_cefr.eval_sari,
                        'g-', label='SARI Score', linewidth=2, marker='s', markersize=4)
            axes_cefr[1].set_xlabel("Steps", fontsize=11)
            axes_cefr[1].set_ylabel("SARI Score", fontsize=11)
            axes_cefr[1].set_title("SARI Progress (CEFR)", fontsize=12, fontweight="bold")
            axes_cefr[1].legend()
            axes_cefr[1].grid(True, alpha=0.3)
        else:
            axes_cefr[1].text(0.5, 0.5, 'SARI metrics\nnot available',
                        ha='center', va='center', transform=axes_cefr[1].transAxes,
                        fontsize=12)

        plt.tight_layout()

        # Results
        results_md_cefr = f"""
        ## 🎉 CEFR Level-Aware Training Complete!

        ### Training Summary
        - **Training Loss**: {train_result_cefr.training_loss:.4f}
        - **Training Time**: {train_result_cefr.metrics['train_runtime']/60:.1f} minutes
        - **Samples/Second**: {train_result_cefr.metrics['train_samples_per_second']:.2f}

        ### Test Set Performance
        - **Test Loss**: {test_results_cefr['eval_loss']:.4f}
        """

        if METRICS_AVAILABLE_LEVEL and 'sari' in test_results_cefr:
            results_md_cefr += f"""
        - **SARI Score**: {test_results_cefr['sari']:.2f} {'✅ Excellent!' if test_results_cefr['sari'] >= 40 else '(Target: ≥40)'}
        - **BLEU Score**: {test_results_cefr['bleu']:.2f}
        - **FKGL Reduction**: {test_results_cefr['fkgl_reduction']:.2f} grade levels
        - **Output FKGL**: {test_results_cefr['output_fkgl']:.2f}
        """

        results_md_cefr += """

        ### ✨ CEFR Level-Aware Capabilities:
        - ✅ Understands C2, C1, B2, B1, A2, A1 levels
        - ✅ Can target specific reading levels
        - ✅ Adjusts complexity based on prompts
        - ✅ Trained on 6+ level combinations
        """

        mo.md(results_md_cefr)
        mo.output.append(fig_cefr)

        # Save
        save_path_cefr = Path("./fine_tuned_t5_cefr_level_aware")
        save_path_cefr.mkdir(exist_ok=True)
        trainer_cefr.save_model(save_path_cefr)
        tokenizer_level.save_pretrained(save_path_cefr)
        print(f"\n✓ CEFR model saved to {save_path_cefr}")

        # Push
        try:
            trainer_cefr.push_to_hub()
            print("✓ Model pushed: X1in/anlp_t5_cefr_level_aware")
        except Exception as e:
            print(f"⚠️ Could not push: {e}")
    else:
        mo.md("Click **🚀 Start CEFR Level-Aware Training** to begin")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Issue with Molab
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Testing with Cefr model
    """)
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, device_level, mo):
    mo.md("## 🔍 Load Trained CEFR Model for Testing")

    # Load from local path
    model_trained_cefr = None
    tokenizer_trained_cefr = None
    status_msgs_cefr = []

    local_model_path_cefr = "./fine_tuned_t5_cefr_level_aware"

    try:
        status_msgs_cefr.append(f"Loading CEFR model from: **{local_model_path_cefr}**...")

        model_trained_cefr = AutoModelForSeq2SeqLM.from_pretrained(local_model_path_cefr)
        tokenizer_trained_cefr = AutoTokenizer.from_pretrained(local_model_path_cefr)
        model_trained_cefr = model_trained_cefr.to(device_level)

        status_msgs_cefr.append(f"""
        ✅ Successfully loaded CEFR model!
        - **Path**: `{local_model_path_cefr}`
        - **Device**: {device_level}
        - **Parameters**: {sum(p.numel() for p in model_trained_cefr.parameters()):,}
        - **CEFR-Aware**: Yes ✓
        """)

    except Exception as e:
        status_msgs_cefr.append(f"❌ Could not load from local: {e}")
        status_msgs_cefr.append("Loading base model as fallback...")

        model_trained_cefr = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        tokenizer_trained_cefr = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)
        model_trained_cefr = model_trained_cefr.to(device_level)

        status_msgs_cefr.append("⚠️ Using untrained base model")

    mo.md("\n\n".join(status_msgs_cefr))
    return model_trained_cefr, tokenizer_trained_cefr


@app.cell
def _(mo, torch):
    mo.md("## 🎨 CEFR Simplification Function")

    def simplify_text_cefr(text, model, tokenizer, device, max_length=128):
        """Simplify text using CEFR-trained model"""
        model.eval()

        # Don't add prefix if already has level-aware prompt
        if not any(phrase in text for phrase in ["Simplify this", "level text to", "Simplify:"]):
            text = "Simplify: " + text

        inputs = tokenizer(
            text,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.0
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    mo.md("✓ CEFR simplification function ready")
    return (simplify_text_cefr,)


@app.cell
def _(
    calculate_fkgl_level,
    device_level,
    mo,
    model_trained_cefr,
    simplify_text_cefr,
    tokenizer_trained_cefr,
):
    mo.md("## 🧪 Test 1: C2 → A2 Simplification")

    # Test sentence (complex)
    test_sentence_c2_a2 = "Despite the initial setbacks, she remained resolute, recognizing that meaningful progress often emerges from sustained effort rather than immediate success."

    # Create CEFR-aware prompt
    prompt_c2_a2 = f"Simplify this C2 level text to A2 level: {test_sentence_c2_a2}"

    # Simplify
    simplified_c2_a2 = simplify_text_cefr(
        prompt_c2_a2, 
        model_trained_cefr, 
        tokenizer_trained_cefr, 
        device_level
    )

    # Calculate metrics
    original_fkgl_c2_a2 = calculate_fkgl_level(test_sentence_c2_a2)
    simplified_fkgl_c2_a2 = calculate_fkgl_level(simplified_c2_a2)
    reduction_c2_a2 = original_fkgl_c2_a2 - simplified_fkgl_c2_a2

    # A2 target range: 4-7
    target_min_a2 = 4
    target_max_a2 = 7
    on_target_c2_a2 = target_min_a2 <= simplified_fkgl_c2_a2 <= target_max_a2

    # Word counts
    orig_words_c2_a2 = len(test_sentence_c2_a2.split())
    simp_words_c2_a2 = len(simplified_c2_a2.split())

    result_md_c2_a2 = f"""
    ### Test 1: C2 → A2 (Proficiency to Elementary)

    **Target**: Elementary school level (FKGL: {target_min_a2}-{target_max_a2})

    ---

    **Original (C2, FKGL: {original_fkgl_c2_a2:.1f}, {orig_words_c2_a2} words):**
    > {test_sentence_c2_a2}

    **Simplified (A2, FKGL: {simplified_fkgl_c2_a2:.1f}, {simp_words_c2_a2} words):**
    > {simplified_c2_a2}

    ---

    **📊 Assessment:**
    - **FKGL Reduction**: {reduction_c2_a2:.1f} grade levels {'✅' if reduction_c2_a2 >= 3 else '⚠️'}
    - **Target Achievement**: {'✅ Within A2 Range!' if on_target_c2_a2 else f'⚠️ Outside range ({target_min_a2}-{target_max_a2})'}
    - **Output FKGL**: {simplified_fkgl_c2_a2:.1f}
    - **Compression**: {simp_words_c2_a2/orig_words_c2_a2:.1%} of original
    """

    mo.md(result_md_c2_a2)
    return


@app.cell
def _(
    calculate_fkgl_level,
    device_level,
    mo,
    model_trained_cefr,
    simplify_text_cefr,
    tokenizer_trained_cefr,
):
    mo.md("## 🧪 Test 2: B2 → B1 Simplification")

    # Test sentence (upper-intermediate)
    test_sentence_b2_b1 = "Contemporary educational institutions are implementing innovative pedagogical approaches to facilitate enhanced student engagement and academic achievement."

    # Create CEFR-aware prompt
    prompt_b2_b1 = f"Simplify this B2 level text to B1 level: {test_sentence_b2_b1}"

    # Simplify
    simplified_b2_b1 = simplify_text_cefr(
        prompt_b2_b1, 
        model_trained_cefr, 
        tokenizer_trained_cefr, 
        device_level
    )

    # Calculate metrics
    original_fkgl_b2_b1 = calculate_fkgl_level(test_sentence_b2_b1)
    simplified_fkgl_b2_b1 = calculate_fkgl_level(simplified_b2_b1)
    reduction_b2_b1 = original_fkgl_b2_b1 - simplified_fkgl_b2_b1

    # B1 target range: 7-10
    target_min_b1 = 7
    target_max_b1 = 10
    on_target_b2_b1 = target_min_b1 <= simplified_fkgl_b2_b1 <= target_max_b1

    # Word counts
    orig_words_b2_b1 = len(test_sentence_b2_b1.split())
    simp_words_b2_b1 = len(simplified_b2_b1.split())

    result_md_b2_b1 = f"""
    ### Test 2: B2 → B1 (Upper-Intermediate to Intermediate)

    **Target**: Middle school level (FKGL: {target_min_b1}-{target_max_b1})

    ---

    **Original (B2, FKGL: {original_fkgl_b2_b1:.1f}, {orig_words_b2_b1} words):**
    > {test_sentence_b2_b1}

    **Simplified (B1, FKGL: {simplified_fkgl_b2_b1:.1f}, {simp_words_b2_b1} words):**
    > {simplified_b2_b1}

    ---

    **📊 Assessment:**
    - **FKGL Reduction**: {reduction_b2_b1:.1f} grade levels {'✅' if reduction_b2_b1 >= 2 else '⚠️'}
    - **Target Achievement**: {'✅ Within B1 Range!' if on_target_b2_b1 else f'⚠️ Outside range ({target_min_b1}-{target_max_b1})'}
    - **Output FKGL**: {simplified_fkgl_b2_b1:.1f}
    - **Compression**: {simp_words_b2_b1/orig_words_b2_b1:.1%} of original
    """

    mo.md(result_md_b2_b1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Testing with Trained model for testing
    """)
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, device_final, mo, test_device):
    mo.md("## 🔍 Load Trained Model for Testing (Final)")

    # Try loading from local folder first, then HuggingFace
    model_trained_final = None
    tokenizer_trained_final = None
    status_messages = []

    # Local path (visible in your file browser)
    local_model_path = "./fine_tuned_t5_augmented_final_approach"

    try:
        status_messages.append(f"Loading model from local path: **{local_model_path}**...")

        model_trained_final = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
        tokenizer_trained_final = AutoTokenizer.from_pretrained(local_model_path)

        # Use existing device_final (already defined in cell-59)
        model_trained_final = model_trained_final.to(device_final)

        status_messages.append(f"""
        ✅ Successfully loaded model from local folder!
        - **Path**: `{local_model_path}`
        - **Device**: {device_final}
        - **Parameters**: {sum(p.numel() for p in model_trained_final.parameters()):,}
        """)

    except Exception as e:
        status_messages.append(f"❌ Could not load from local path: {e}")
        status_messages.append("Trying HuggingFace Hub...")

        # Fallback to HuggingFace
        try:
            model_id = "X1in/fine_tuned_t5_augmented_final_approach"
            model_trained_final = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            tokenizer_trained_final = AutoTokenizer.from_pretrained(model_id)
            model_trained_final = model_trained_final.to(test_device)

            status_messages.append(f"✅ Loaded from HuggingFace: **{model_id}**")

        except Exception as e2:
            status_messages.append(f"❌ Could not load from HuggingFace either: {e2}")
            status_messages.append("Loading base model as fallback...")

            model_trained_final = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            tokenizer_trained_final = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)
            model_trained_final = model_trained_final.to(device_final)

            status_messages.append("⚠️ Using untrained base model")

    # Display all status messages (THIS is what gets shown)
    mo.md("\n\n".join(status_messages))
    return model_trained_final, tokenizer_trained_final


@app.cell
def _(mo, torch):
    mo.md("## 🎨 Simplification Function (Final)")

    def simplify_text_final(text, model, tokenizer, device, max_length=128):
        """Simplify text using trained model - Final version"""
        model.eval()

        # Add instruction prefix if not present
        if not any(text.startswith(prefix) for prefix in ["Simplify:", "Make this simpler:", "Rewrite in simple words:"]):
            text = "Simplify: " + text

        inputs = tokenizer(
            text,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.0
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    mo.md("✓ Simplification function ready (final)")
    return (simplify_text_final,)


@app.cell
def _(
    calculate_fkgl_final,
    device_final,
    mo,
    model_trained_final,
    simplify_text_final,
    tokenizer_trained_final,
):
    mo.md("## 🧪 Single Text Simplification Test")

    # Predefined test sentence
    test_sentence_single = "The implementation of comprehensive regulatory frameworks necessitates substantial modifications to existing operational procedures and organizational structures."

    # Simplify
    simplified_single_final = simplify_text_final(
        test_sentence_single, 
        model_trained_final, 
        tokenizer_trained_final, 
        device_final
    )

    # Calculate metrics
    original_fkgl_single = calculate_fkgl_final(test_sentence_single)
    simplified_fkgl_single = calculate_fkgl_final(simplified_single_final)
    reduction_single = original_fkgl_single - simplified_fkgl_single

    # Word counts
    original_words_single = len(test_sentence_single.split())
    simplified_words_single = len(simplified_single_final.split())
    compression_single = simplified_words_single / original_words_single if original_words_single > 0 else 0

    result_md_single = f"""
    ### 📝 Single Simplification Result

    **Original (FKGL: {original_fkgl_single:.1f}, {original_words_single} words):**
    > {test_sentence_single}

    **Simplified (FKGL: {simplified_fkgl_single:.1f}, {simplified_words_single} words):**
    > {simplified_single_final}

    ---

    **📊 Metrics:**
    - **FKGL Reduction**: {reduction_single:.1f} grade levels {'✅ Excellent!' if reduction_single >= 3 else '⚠️ Could be better'}
    - **Compression Ratio**: {compression_single:.2%} of original length
    - **Words Saved**: {original_words_single - simplified_words_single} words
    - **Readability**: {'Easy' if simplified_fkgl_single <= 8 else 'Moderate' if simplified_fkgl_single <= 12 else 'Complex'}
    """

    mo.md(result_md_single)
    return


@app.cell
def _():
    return


@app.cell
def _(
    calculate_fkgl_final,
    device_final,
    mo,
    model_trained_final,
    simplify_text_final,
    tokenizer_trained_final,
):
    mo.md("## 📝 Batch Testing Across Domains")

    # Predefined test examples across domains
    test_examples_domains = [
        ("Academic", "The proliferation of sophisticated technological innovations has fundamentally transformed contemporary communication methodologies."),
        ("Medical", "Hypertension, characterized by persistently elevated arterial blood pressure, constitutes a significant cardiovascular risk factor."),
        ("Legal", "The aforementioned party shall indemnify and hold harmless the other party from any liabilities arising therefrom."),
        ("Technical", "The algorithm utilizes a convolutional neural network architecture with residual connections to optimize performance."),
        ("Business", "Our organization is implementing a comprehensive digital transformation strategy to enhance operational efficiency and stakeholder engagement."),
    ]

    # Process all examples
    batch_results_domains = []
    for category, example in test_examples_domains:
        simplified_domain = simplify_text_final(example, model_trained_final, tokenizer_trained_final, device_final)
        orig_fkgl_domain = calculate_fkgl_final(example)
        simp_fkgl_domain = calculate_fkgl_final(simplified_domain)
        reduction_domain = orig_fkgl_domain - simp_fkgl_domain

        batch_results_domains.append({
            "category": category,
            "original": example,
            "simplified": simplified_domain,
            "orig_fkgl": orig_fkgl_domain,
            "simp_fkgl": simp_fkgl_domain,
            "reduction": reduction_domain
        })

    # Build output
    batch_md_domains = "### 📊 Domain-Specific Simplification Results\n\n"
    for result in batch_results_domains:
        batch_md_domains += f"""
    #### {result['category']} Domain
    **FKGL Reduction**: {result['reduction']:.1f} levels {'✅' if result['reduction'] >= 2 else '⚠️'}

    **Original (Grade {result['orig_fkgl']:.1f}):**  
    {result['original']}

    **Simplified (Grade {result['simp_fkgl']:.1f}):**  
    {result['simplified']}

    ---

    """

    mo.md(batch_md_domains)
    return


@app.cell
def _(
    calculate_fkgl_final,
    device_final,
    mo,
    model_trained_final,
    simplify_text_final,
    tokenizer_trained_final,
):
    mo.md("## 🎯 Level-Aware: C2 → A2 Simplification")

    # Predefined test for C2 to A2
    test_text_c2_to_a2 = "The multifaceted implications of climate change necessitate immediate and coordinated international responses to mitigate anthropogenic greenhouse gas emissions."
    source_level_c2a2 = "C2"
    target_level_c2a2 = "A2"

    # Create level-aware prompt
    prompt_c2_to_a2 = f"Simplify this {source_level_c2a2} level text to {target_level_c2a2} level: {test_text_c2_to_a2}"

    # Simplify
    simplified_c2_to_a2 = simplify_text_final(
        prompt_c2_to_a2, 
        model_trained_final, 
        tokenizer_trained_final, 
        device_final
    )

    # Metrics
    orig_fkgl_c2a2 = calculate_fkgl_final(test_text_c2_to_a2)
    simp_fkgl_c2a2 = calculate_fkgl_final(simplified_c2_to_a2)

    # Target FKGL range for A2
    target_min_a2, target_max_a2 = (4, 7)
    on_target_c2a2 = target_min_a2 <= simp_fkgl_c2a2 <= target_max_a2

    level_md_c2a2 = f"""
    ### 🎯 C2 → A2 Simplification Result

    **Target FKGL Range**: {target_min_a2}-{target_max_a2} (Elementary level)

    ---

    **Original (C2, FKGL: {orig_fkgl_c2a2:.1f}):**
    > {test_text_c2_to_a2}

    **Simplified (A2, FKGL: {simp_fkgl_c2a2:.1f}):**
    > {simplified_c2_to_a2}

    ---

    **📊 Assessment:**
    - **FKGL Reduction**: {orig_fkgl_c2a2 - simp_fkgl_c2a2:.1f} grade levels
    - **Target Achievement**: {'✅ Within Target Range!' if on_target_c2a2 else f'⚠️ Outside range ({target_min_a2}-{target_max_a2})'}
    - **Output Complexity**: {simp_fkgl_c2a2:.1f}
    - **Appropriate for**: Elementary students (grades 4-7)
    """

    mo.md(level_md_c2a2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Running Evaluation Script
    """)
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, torch):


    import json
    import os
    # import torch
    # from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from tqdm import tqdm

    # ============ CONFIGURATION ============
    EVAL_MODEL_PATH = "./fine_tuned_t5_cefr_level_aware"
    EVAL_TEST_FILE = "./data/tsar2025_test.jsonl"  # Try this path
    EVAL_TEAM_NAME = "my_team"
    EVAL_MODEL_NAME = "cefr_t5_model"
    EVAL_BATCH_SIZE = 8
    EVAL_MAX_LENGTH = 128
    EVAL_NUM_BEAMS = 4
    # ======================================

    print("🚀 TSAR 2025 Submission Generator - DIAGNOSTIC MODE\n")

    # DIAGNOSTIC: Check file paths
    print("=" * 60)
    print("DIAGNOSTIC: Checking file paths...")
    print("=" * 60)

    # Check current directory
    print(f"\n📁 Current directory: {os.getcwd()}")

    # Check if test file exists
    print(f"\n🔍 Checking test file: {EVAL_TEST_FILE}")
    print(f"   File exists: {os.path.exists(EVAL_TEST_FILE)}")

    if os.path.exists(EVAL_TEST_FILE):
        _file_size = os.path.getsize(EVAL_TEST_FILE)
        print(f"   File size: {_file_size} bytes")

        # Read first few lines
        print(f"\n📄 First 3 lines of file:")
        with open(EVAL_TEST_FILE, 'r', encoding='utf-8') as _f:
            for _i, _line in enumerate(_f):
                if _i >= 3:
                    break
                print(f"   Line {_i+1} (len={len(_line)}): {_line[:100]}...")

        # Check for empty lines
        print(f"\n🔍 Checking for empty/whitespace lines:")
        with open(EVAL_TEST_FILE, 'r', encoding='utf-8') as _f:
            _lines = _f.readlines()
            _empty_lines = [_i+1 for _i, _l in enumerate(_lines) if not _l.strip()]
            if _empty_lines:
                print(f"   Found empty lines at: {_empty_lines[:10]}")
            else:
                print(f"   No empty lines found")
            print(f"   Total lines: {len(_lines)}")
            print(f"   Non-empty lines: {len([_l for _l in _lines if _l.strip()])}")
    else:
        # Try to find the file
        print(f"\n❌ File not found! Searching for tsar2025_test.jsonl...")

        _possible_paths = [
            "data/tsar2025_test.jsonl",
            "data/asset/tsar2025_test.jsonl",
            "./data/asset/tsar2025_test.jsonl",
            "evaluation/tsar2025_test.jsonl",
            "tsar2025_test.jsonl",
        ]

        for _path in _possible_paths:
            if os.path.exists(_path):
                print(f"   ✅ Found at: {_path}")
                EVAL_TEST_FILE = _path
                break
        else:
            print(f"   ❌ Not found in common locations")
            print(f"\n📂 Listing 'data' directory:")
            if os.path.exists("data"):
                for _root, _dirs, _files in os.walk("data"):
                    for _file in _files:
                        if "tsar" in _file.lower():
                            print(f"      {os.path.join(_root, _file)}")

    print("\n" + "=" * 60)

    # Only proceed if file was found
    if not os.path.exists(EVAL_TEST_FILE):
        print("\n❌ Cannot proceed: Test file not found!")
        print(f"   Please update EVAL_TEST_FILE to the correct path")
        raise FileNotFoundError(f"Test file not found: {EVAL_TEST_FILE}")

    # Load model
    print(f"\n📦 Loading model from: {EVAL_MODEL_PATH}")
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {eval_device}")

    eval_model = AutoModelForSeq2SeqLM.from_pretrained(EVAL_MODEL_PATH)
    eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_PATH)
    eval_model = eval_model.to(eval_device)
    eval_model.eval()
    print("✅ Model loaded\n")

    # Load test data with better error handling
    print(f"📖 Loading test data: {EVAL_TEST_FILE}")
    eval_test_data = []
    _line_num = 0

    try:
        with open(EVAL_TEST_FILE, 'r', encoding='utf-8') as _eval_f:
            for _line_num, _line in enumerate(_eval_f, 1):
                if _line.strip():  # Skip empty lines
                    try:
                        _data = json.loads(_line)
                        eval_test_data.append(_data)
                    except json.JSONDecodeError as _je:
                        print(f"   ⚠️  Warning: Invalid JSON at line {_line_num}")
                        print(f"      Line content: {_line[:100]}")
                        print(f"      Error: {_je}")
                        continue
    except Exception as _e:
        print(f"❌ ERROR reading file: {_e}")
        raise

    if not eval_test_data:
        raise ValueError(f"No valid data loaded from: {EVAL_TEST_FILE}")

    print(f"   ✅ Loaded {len(eval_test_data)} test instances")

    # Extract fields
    eval_text_ids = [item['text_id'] for item in eval_test_data]
    eval_originals = [item['original'] for item in eval_test_data]
    eval_target_cefrs = [item['target_cefr'] for item in eval_test_data]

    print(f"   Sample ID: {eval_text_ids[0]}")
    print(f"   Sample target: {eval_target_cefrs[0]}\n")

    # Generate simplifications
    print(f"🤖 Generating simplifications...")
    eval_simplified_texts = []

    with torch.no_grad():
        for eval_i in tqdm(range(0, len(eval_originals), EVAL_BATCH_SIZE)):
            eval_batch_originals = eval_originals[eval_i:eval_i+EVAL_BATCH_SIZE]
            eval_batch_targets = eval_target_cefrs[eval_i:eval_i+EVAL_BATCH_SIZE]

            # Match training format: "Simplify this [SOURCE] level text to [TARGET] level: {text}"
            # Assume source is always C2 (complex text) when not specified
            eval_prompts = [
                f"Simplify this C2 level text to {cefr} level: {text}"
                for text, cefr in zip(eval_batch_originals, eval_batch_targets)
            ]

            eval_inputs = eval_tokenizer(
                eval_prompts,
                max_length=EVAL_MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(eval_device)

            eval_outputs = eval_model.generate(
                **eval_inputs,
                max_length=EVAL_MAX_LENGTH,
                num_beams=EVAL_NUM_BEAMS,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

            eval_batch_simplified = eval_tokenizer.batch_decode(eval_outputs, skip_special_tokens=True)
            eval_simplified_texts.extend(eval_batch_simplified)

    print(f"✅ Generated {len(eval_simplified_texts)} simplifications\n")

    # Create submission file
    eval_submission_dir = f"evaluation/submissions/{EVAL_TEAM_NAME}"
    os.makedirs(eval_submission_dir, exist_ok=True)
    eval_submission_path = f"{eval_submission_dir}/{EVAL_MODEL_NAME}.jsonl"

    print(f"💾 Creating submission: {eval_submission_path}")
    with open(eval_submission_path, 'w', encoding='utf-8') as _eval_f:
        for eval_text_id, eval_simplified in zip(eval_text_ids, eval_simplified_texts):
            eval_entry = {
                "text_id": eval_text_id,
                "simplified": eval_simplified
            }
            _eval_f.write(json.dumps(eval_entry, ensure_ascii=False) + '\n')

    print(f"✅ Submission created!\n")

    # Show samples
    print("📊 Sample predictions:")
    for eval_i in range(min(3, len(eval_text_ids))):
        print(f"\n[{eval_i+1}] ID: {eval_text_ids[eval_i]} | Target: {eval_target_cefrs[eval_i]}")
        print(f"    Original:   {eval_originals[eval_i][:70]}...")
        print(f"    Simplified: {eval_simplified_texts[eval_i][:70]}...")

    print(f"\n{'='*60}")
    print("✅ DONE! Next steps:")
    print(f"1. Your submission is at: {eval_submission_path}")
    print(f"2. Copy test file to evaluation folder:")
    print(f"   cp {EVAL_TEST_FILE} evaluation/tsar2025_test.jsonl")
    print(f"3. Run evaluation:")
    print(f"   cd evaluation && python tsar2025_evaluation_script.py")
    print(f"{'='*60}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
