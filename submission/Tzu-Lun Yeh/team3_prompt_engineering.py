# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bert-score==0.3.13",
#     "evaluate==0.4.6",
#     "google-genai>=1.55.0",
#     "google-auth>=2.45.0",
#     "marimo>=0.19.0",
#     "matplotlib==3.10.8",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "protobuf==6.33.2",
#     "pyzmq",
#     "scikit-learn==1.8.0",
#     "torch==2.9.1",
#     "transformers==4.57.3",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(
    width="full",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import os
    import shutil

    # Clean up progress checkpoint file
    progress_file = "progress_checkpoint_prompt_eng.json"
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Deleted: {progress_file}")
    else:
        print(f"Progress file not found: {progress_file}")

    # Clean up submissions folder
    submissions_dir = "submissions"
    if os.path.exists(submissions_dir):
        shutil.rmtree(submissions_dir)
        print(f"Deleted submissions folder: {submissions_dir}")
    else:
        print(f"Submissions folder not found: {submissions_dir}")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import subprocess
    subprocess.run(["uv", "pip", "install", "google-cloud-storage", "google-cloud-aiplatform", "google-genai", "-q"])
    return


@app.cell
def _(mo):
    import os as _os

    # GCP Configuration
    PROJECT_ID = "anlp-483413"
    LOCATION = "us-central1"

    # Service account JSON file (local path from notebooks/)
    SA_JSON_FILE = "../data/anlp-483413-1818968712bd.json"

    _credentials_path = _os.path.abspath(SA_JSON_FILE)

    if _os.path.exists(_credentials_path):
        _os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _credentials_path
        _auth_status = f"OK - {_credentials_path}"
    else:
        _auth_status = f"NOT FOUND - {SA_JSON_FILE}"

    # Set project for Vertex AI
    _os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    _os.environ["VERTEXAI_PROJECT"] = PROJECT_ID
    _os.environ["VERTEXAI_LOCATION"] = LOCATION

    mo.md(f"""
    ### GCP Configuration

    - Project: {PROJECT_ID}
    - Location: {LOCATION}
    - Credentials: {_auth_status}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TSAR 2025 Prompt Engineering Experiments

    This notebook implements 4 prompt engineering strategies for text simplification (Ablation Study):

    1. **Baseline**: Zero-shot with minimal instruction
    2. **With Instructions**: Baseline + detailed instructions
    3. **With Examples**: Baseline + few-shot examples
    4. **Merged**: Instructions + examples (optimal)

    **All strategies use Google Generative AI (google-genai) exclusively.**
    """)
    return


@app.cell
def _():
    import json

    def read_jsonl(path):
        """Read JSONL file and return list of dictionaries"""
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    return json, read_jsonl


@app.cell
def _(read_jsonl):
    # Load TSAR trial data
    trial_data = read_jsonl("../data/tsar2025_test.jsonl")  # 200 samples for final evaluation

    # Dynamically group by target_cefr level
    cefr_levels = sorted(set(d["target_cefr"] for d in trial_data))
    data_by_cefr = {level: [d for d in trial_data if d["target_cefr"] == level] for level in cefr_levels}
    return cefr_levels, data_by_cefr, trial_data


@app.cell
def _(cefr_levels, data_by_cefr, mo):
    _level_counts = ", ".join([f"**{level.upper()}**: {len(data_by_cefr[level])}" for level in cefr_levels])
    mo.md(f"""
    ### Trial Data Loaded

    - **Total samples**: {sum(len(v) for v in data_by_cefr.values())}
    - **CEFR levels found**: {', '.join(level.upper() for level in cefr_levels)}
    - {_level_counts}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Prompt Engineering Strategies

    Defining 4 prompt engineering strategies for ablation study.
    """)
    return


@app.cell
def _():
    # 4 Prompt Engineering Strategies (Ablation Study)
    PROMPT_STRATEGIES = {
        "1_baseline": {
            "description": "Zero-shot baseline (minimal instruction)",
            "template": """Simplify this text to {cefr_level} CEFR level. Output only the simplified text.

    {original}"""
        },

        "2_with_instructions": {
            "description": "Instructions ONLY (No examples)",
            "template": """Simplify this text to {cefr_level} CEFR level.

    {instructions}

    Text: {original}

    Simplified:"""
        },

        "3_with_examples": {
            "description": "Examples ONLY (No instructions)",
            "template": """Simplify this text to {cefr_level} CEFR level. Output only the simplified text.

    {examples}

    Text: {original}

    Simplified:"""
        },

        "4_merged": {
            "description": "Instructions + Examples (Combined)",
            "template": """Simplify this text to {cefr_level} CEFR level.

    {instructions}

    {examples}

    Text: {original}

    Simplified:"""
        }
    }

    # === Separate INSTRUCTIONS (for Strategy 2 & 4) ===
    INSTRUCTIONS_A2 = """Instructions:
    - Split long sentences into shorter ones
    - Use one idea per sentence
    - Use simple words and add definitions for new concepts
    - Keep sentences under 12 words on average
    - Replace complex words with simpler alternatives
    - Output only the simplified text"""

    INSTRUCTIONS_B1 = """Instructions:
    - Do NOT split sentences into short fragments. (Critical!)
    - Maintain the original sentence flow and logic.
    - Use connectors (however, therefore, although, because) to join ideas.
    - Use intermediate vocabulary with average word length > 4.5 characters.
    - Keep precise, specific nouns (e.g., 'achievement', 'atmosphere', 'author').
    - Target: average sentence length should be 18-25 words.
    - Use relative clauses (who, which, that) to add detail.
    - Output only the simplified text."""

    # === Separate EXAMPLES (for Strategy 3 & 4) ===
    # Pure examples without rules (clean variable control)
    EXAMPLES_A2 = """<examples>
    <example>
    Original: "Earthquakes damage all structures, including bridges. Luckily, this kind of collapse is relatively infrequent, especially with modern bridges. Engineers have learned to design bridges in earthquake zones on areas that are much more resistant to movement."
    A2: "An earthquake means when the ground shakes a lot. Earthquakes can make buildings fall down, including bridges. Luckily, modern bridges don't fall down very often. In earthquake areas, engineers now choose to build bridges in places that do not move so much."
    </example>

    <example>
    Original: "Dreams of a phone call from an old friend or the death of someone close, for example, are more likely to be the result of coincidence than prophecy."
    A2: "Sometimes, we dream that an old friend will telephone us, or we dream that a friend or family member will die, and then those things happen. This is probably just because of chance, not an ability to see the future."
    </example>
    </examples>"""

    EXAMPLES_B1 = """<examples>
    <example>
    Original: "Earthquakes damage all structures, including bridges. Luckily, this kind of collapse is relatively infrequent, especially with modern bridges. Engineers have learned to design bridges in earthquake zones on areas that are much more resistant to movement."
    B1: "Earthquakes damage all kinds of buildings, including bridges. Luckily, however, bridges, especially modern ones, are not often completely destroyed. Engineers have learned to design bridges in earthquake zones on ground that moves less during earthquakes."
    </example>

    <example>
    Original: "Dreams of a phone call from an old friend or the death of someone close, for example, are more likely to be the result of coincidence than prophecy. And, of course, we probably choose to forget all the times we dream about such events but they don't happen."
    B1: "If we dream that we will get a phone call from an old friend or learn about the death of someone close, for example, this is more likely to be the result of chance rather than an ability to predict the future. And, of course, we probably choose to forget all the times we dream about events like that but then they don't happen."
    </example>
    </examples>"""

    num_strategies = len(PROMPT_STRATEGIES)
    return (
        EXAMPLES_A2,
        EXAMPLES_B1,
        INSTRUCTIONS_A2,
        INSTRUCTIONS_B1,
        PROMPT_STRATEGIES,
        num_strategies,
    )


@app.cell
def _(PROMPT_STRATEGIES, mo, num_strategies):
    _strategy_list = "\n".join([f"- **{name}**: {info['description']}" for name, info in PROMPT_STRATEGIES.items()])

    mo.md(f"""
    ### {num_strategies} Strategies Defined

    {_strategy_list}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## API Client Setup

    Initialize Google Generative AI client for all 4 strategies.
    """)
    return


@app.cell
def _(mo):
    from google import genai

    # Create genai client
    genai_client = genai.Client(vertexai=True)

    # List all available models
    _available_models = []
    for _model in genai_client.models.list():
        if "gemini" in _model.name.lower():
            _available_models.append(_model.name)

    # Sort models (newest first)
    _available_models = sorted(_available_models, reverse=True)

    # Create dropdown for model selection
    model_dropdown = mo.ui.dropdown(
        options=_available_models,
        value="publishers/google/models/gemini-2.5-flash",
        label="Select Model"
    )

    mo.md(f"""
    ### Google Generative AI Client

    Found **{len(_available_models)}** Gemini models.

    {model_dropdown}
    """)
    return genai_client, model_dropdown


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    All strategies use the google-genai client exclusively.
    No additional API wrappers needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Unified Strategy Execution Function

    This function routes to the appropriate API based on strategy configuration.
    """)
    return


@app.cell
def _(
    EXAMPLES_A2,
    EXAMPLES_B1,
    INSTRUCTIONS_A2,
    INSTRUCTIONS_B1,
    PROMPT_STRATEGIES,
    genai_client,
):
    import time as _time_module

    def call_strategy(
        strategy_name: str,
        original: str,
        cefr_level: str,
        model: str,
        temperature: float = 0.0
    ) -> str:
        """
        Unified strategy execution function for Ablation Study

        Args:
            strategy_name: Strategy key (e.g., "1_baseline", "2_with_instructions")
            original: Original text to simplify
            cefr_level: Target CEFR level (A2/B1)
            model: Model name
            temperature: Temperature parameter

        Returns:
            Simplified text
        """
        strategy_info = PROMPT_STRATEGIES[strategy_name]
        template = strategy_info["template"]
        level_key = cefr_level.upper()

        # Select CEFR-specific instructions and examples
        instructions = INSTRUCTIONS_A2 if level_key == "A2" else INSTRUCTIONS_B1
        examples = EXAMPLES_A2 if level_key == "A2" else EXAMPLES_B1

        # Build format arguments
        format_args = {
            "cefr_level": level_key,
            "original": original
        }

        # Add instructions if template uses them (Strategy 2 & 4)
        if "{instructions}" in template:
            format_args["instructions"] = instructions

        # Add examples if template uses them (Strategy 3 & 4)
        if "{examples}" in template:
            format_args["examples"] = examples

        # Format prompt
        prompt = template.format(**format_args)

        # Call google-genai with retry
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = genai_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": 4096
                    }
                )
                result_text = response.text.strip()
                return result_text

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    _time_module.sleep(wait_time)
                else:
                    return f"ERROR: {str(e)}"

        return "ERROR: Max retries exceeded"
    return (call_strategy,)


@app.cell
def _(PROMPT_STRATEGIES, call_strategy, data_by_cefr, mo, model_dropdown):
    import os as _os
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # --- Configuration ---
    MAX_WORKERS = 20  # Adjust down to 5-10 if rate limited
    SAVE_INTERVAL = 20  # Save every 20 items to reduce I/O
    MAX_SAMPLES_PER_LEVEL = None  # None = all samples

    _progress_file = "progress_checkpoint_prompt_eng.json"
    _jsonl_file = "progress_checkpoint_prompt_eng.jsonl"  # Append-only backup

    # Load existing progress (with error handling)
    all_results = {}
    if _os.path.exists(_progress_file):
        try:
            with open(_progress_file, 'r', encoding='utf-8') as _pf:
                all_results = _json.load(_pf)
        except _json.JSONDecodeError:
            print("Warning: Progress file corrupted, starting fresh.")
            all_results = {}

    _save_lock = threading.Lock()

    def process_single_item(args):
        """Process single item - relies on call_strategy's internal retry"""
        _strategy, _cefr_level, _item, _model = args
        _exp_name = f"{_cefr_level}_{_strategy}"

        # call_strategy already has retry logic inside
        _simplified = call_strategy(
            _strategy,
            _item["original"],
            _cefr_level.upper(),
            _model,
            temperature=0.0
        )

        is_error = _simplified.startswith("ERROR:")
        return {
            "exp_name": _exp_name,
            "text_id": _item["text_id"],
            "simplified": _simplified,
            "error": is_error
        }

    # Build task list
    _tasks = []
    for _strategy in PROMPT_STRATEGIES.keys():
        for _cefr_level, _data in data_by_cefr.items():
            if not _data:
                continue

            _exp_name = f"{_cefr_level}_{_strategy}"
            _processed_ids = {r["text_id"] for r in all_results.get(_exp_name, [])}

            for _item in _data[:MAX_SAMPLES_PER_LEVEL] if MAX_SAMPLES_PER_LEVEL else _data:
                if _item["text_id"] not in _processed_ids:
                    _tasks.append((_strategy, _cefr_level, _item, model_dropdown.value))

    _total = len(_tasks)
    _completed = 0
    _error_count = 0

    if _total == 0:
        mo.md("### All tasks already completed! Nothing to process.")
    else:
        with mo.status.spinner(title=f"Processing {_total} items (Parallel)", subtitle=f"Workers: {MAX_WORKERS}") as _spinner:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_single_item, task): task for task in _tasks}

                for future in as_completed(futures):
                    result = future.result()
                    _completed += 1

                    if result["error"]:
                        _error_count += 1

                    _exp_name = result["exp_name"]
                    with _save_lock:
                        if _exp_name not in all_results:
                            all_results[_exp_name] = []
                        all_results[_exp_name].append({
                            "text_id": result["text_id"],
                            "simplified": result["simplified"]
                        })

                        # Append to JSONL (fast, atomic writes)
                        with open(_jsonl_file, 'a', encoding='utf-8') as _jl:
                            _jl.write(_json.dumps({
                                "exp_name": result["exp_name"],
                                "text_id": result["text_id"],
                                "simplified": result["simplified"]
                            }, ensure_ascii=False) + '\n')

                        # Save full JSON periodically
                        if _completed % SAVE_INTERVAL == 0:
                            with open(_progress_file, 'w', encoding='utf-8') as _pf:
                                _json.dump(all_results, _pf, ensure_ascii=False, indent=2)

                    _spinner.update(subtitle=f"Progress: {_completed}/{_total} (Errors: {_error_count})")

            # Final save
            with open(_progress_file, 'w', encoding='utf-8') as _pf:
                _json.dump(all_results, _pf, ensure_ascii=False, indent=2)

        mo.md(f"""
        ### Batch Processing Complete ✅

        - **Model**: {model_dropdown.value}
        - **Completed**: {_completed}/{_total}
        - **Errors**: {_error_count}
        - **Progress saved to**: `{_progress_file}`
        - **JSONL backup**: `{_jsonl_file}`
        """)
    return (all_results,)


@app.cell
def _(all_results, json):
    import os as _os

    # Save results as JSONL files
    aligned_submissions = {}

    for _exp_name, _results in all_results.items():
        _aligned_results = []
        for _result in _results:
            _aligned_results.append({
                "text_id": _result["text_id"],
                "simplified": _result["simplified"]
            })

        aligned_submissions[_exp_name] = _aligned_results

    # Save as JSONL files
    _os.makedirs("submissions", exist_ok=True)

    for _exp_name, _results in aligned_submissions.items():
        _output_path = f"submissions/{_exp_name}.jsonl"
        with open(_output_path, 'w', encoding='utf-8') as _f:
            for _result in _results:
                _f.write(json.dumps(_result, ensure_ascii=False) + '\n')
    return (aligned_submissions,)


@app.cell
def _(aligned_submissions, mo):
    mo.md(f"""
    ### Submissions Saved

    Successfully saved **{len(aligned_submissions)}** submission files in JSONL format.

    Location: `submissions/`

    Files are ready for evaluation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluation Pipeline

    Loading evaluation models and computing metrics...
    """)
    return


@app.cell
def _():
    # Note: Offline mode moved to later cell (after model download)

    from transformers import pipeline
    import evaluate
    import torch

    # Determine device
    device = 0 if torch.cuda.is_available() else -1

    # Load CEFR classification models (ensemble of 3)
    cefr_labeler1 = pipeline(
        "text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
        device=device,
        torch_dtype="auto"
    )

    cefr_labeler2 = pipeline(
        "text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
        device=device,
        torch_dtype="auto"
    )

    cefr_labeler3 = pipeline(
        "text-classification",
        model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2",
        device=device,
        torch_dtype="auto"
    )

    # Load semantic similarity metrics
    meaning_bert = evaluate.load("davebulaval/meaningbert")
    bertscore = evaluate.load("bertscore")
    return bertscore, cefr_labeler1, cefr_labeler2, cefr_labeler3, meaning_bert


@app.cell
def _(
    bertscore,
    cefr_labeler1,
    cefr_labeler2,
    cefr_labeler3,
    meaning_bert,
    os,
):
    # Enable offline mode NOW (after models are loaded)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_EVALUATE_OFFLINE"] = "1"

    import numpy as np
    from sklearn.metrics import f1_score, root_mean_squared_error

    CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
    LEVEL_TO_INDEX = {level: idx for idx, level in enumerate(CEFR_LEVELS)}

    def cefr_classify(texts, batch_size=32):
        """Classify CEFR levels using ensemble of 3 models with majority voting"""
        from collections import Counter

        pred1 = cefr_labeler1(texts, batch_size=batch_size, truncation=True)
        pred2 = cefr_labeler2(texts, batch_size=batch_size, truncation=True)
        pred3 = cefr_labeler3(texts, batch_size=batch_size, truncation=True)

        final_labels = []
        for p1, p2, p3 in zip(pred1, pred2, pred3):
            def get_top(x):
                if isinstance(x, dict):
                    return x
                if isinstance(x, list) and x:
                    return max(x, key=lambda d: d["score"])

            # Get predictions from all 3 models
            top1 = get_top(p1)
            top2 = get_top(p2)
            top3 = get_top(p3)

            label1 = top1["label"].strip().upper()
            label2 = top2["label"].strip().upper()
            label3 = top3["label"].strip().upper()

            # Majority voting
            vote_counts = Counter([label1, label2, label3])
            most_common_label, count = vote_counts.most_common(1)[0]

            # If tie (all different predictions), use confidence score
            if count == 1:
                best = max((top1, top2, top3), key=lambda d: d["score"])
                final_labels.append(best["label"].strip().upper())
            else:
                final_labels.append(most_common_label)

        return final_labels

    def evaluate_cefr(predictions, targets):
        """Calculate CEFR metrics: F1, adjacent accuracy, RMSE"""
        gold = [str(t).strip().upper() for t in targets]
        pred = [str(p).strip().upper() for p in predictions]

        weighted_f1 = f1_score(gold, pred, average="weighted")

        gold_idx = np.array([LEVEL_TO_INDEX[l] for l in gold])
        pred_idx = np.array([LEVEL_TO_INDEX[l] for l in pred])
        adj_accuracy = (np.abs(gold_idx - pred_idx) <= 1).mean()

        rmse = root_mean_squared_error(gold_idx, pred_idx)

        return {
            "weighted_f1": round(float(weighted_f1), 4),
            "adj_accuracy": round(float(adj_accuracy), 4),
            "rmse": round(float(rmse), 4)
        }

    def evaluate_meaning_preservation(predictions, references):
        """Calculate MeaningBERT score"""
        result = meaning_bert.compute(
            predictions=predictions,
            references=references
        )
        mean_score = np.mean(result["scores"]) / 100.0
        return round(float(mean_score), 4)

    def evaluate_bertscore(predictions, references):
        """Calculate BERTScore F1"""
        result = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en"
        )
        mean_f1 = np.mean(result["f1"])
        return round(float(mean_f1), 4)
    return (
        cefr_classify,
        evaluate_bertscore,
        evaluate_cefr,
        evaluate_meaning_preservation,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluation Context & Methodology

    ### Key Finding: Classifier Ceiling (from Trial Data)

    When evaluating the **gold B1 references** (human-written) from `tsar2025_trialdata.jsonl` 
    with our CEFR classifier ensemble, only **65%** were correctly classified as B1. 
    This establishes an important **upper bound** for our evaluation:

    | Gold Reference | Predicted Level | Count | Percentage |
    |----------------|-----------------|-------|------------|
    | B1 | B1 (✓ Correct) | 13/20 | **65%** |
    | B1 | B2 (Over) | 6/20 | 30% |
    | B1 | A2 (Under) | 1/20 | 5% |

    **Implication**: Even perfect B1 text cannot achieve 100% accuracy with this classifier.
    Our strategies achieving **50%+ B1 accuracy** are approaching this theoretical ceiling.

    ### Ablation Study Design

    | Strategy | Instructions | Examples | Purpose |
    |----------|-------------|----------|---------|
    | 1_baseline | ✗ | ✗ | Baseline (zero-shot) |
    | 2_with_instructions | ✓ | ✗ | Test instruction effectiveness |
    | 3_with_examples | ✗ | ✓ | Test example effectiveness |
    | 4_merged | ✓ | ✓ | Combined effect |
    """)
    return


@app.cell
def _(
    aligned_submissions,
    cefr_classify,
    evaluate_bertscore,
    evaluate_cefr,
    evaluate_meaning_preservation,
    mo,
    trial_data,
):
    import pandas as pd

    # Execute full evaluation
    gold_dict = {item["text_id"]: item for item in trial_data}
    eval_results_list = []

    with mo.status.spinner(
        title="Running Full Evaluation",
        subtitle="This may take 3-5 minutes..."
    ) as eval_spinner:

        for idx, (_exp_name, predictions) in enumerate(aligned_submissions.items(), 1):
            _cefr_level, _strategy = _exp_name.split("_", 1)

            eval_spinner.update(
                subtitle=f"Evaluating {_exp_name} ({idx}/{len(aligned_submissions)})..."
            )

            # Align predictions with gold data
            hyps, refs, targets, origs = [], [], [], []
            _skipped_errors = 0
            _has_reference = False

            for pred in predictions:
                # Skip error results
                if pred["simplified"].startswith("ERROR:"):
                    _skipped_errors += 1
                    continue

                gold = gold_dict.get(pred["text_id"])
                if gold:
                    hyps.append(pred["simplified"])
                    targets.append(gold["target_cefr"])
                    origs.append(gold["original"])

                    # Check if reference exists
                    if "reference" in gold and gold["reference"]:
                        refs.append(gold["reference"])
                        _has_reference = True
                    else:
                        refs.append("")  # placeholder

            if not hyps:
                continue

            # 1. CEFR Classification
            predicted_levels = cefr_classify(hyps, batch_size=32)
            cefr_metrics = evaluate_cefr(predicted_levels, targets)

            # Compute prediction distribution
            from collections import Counter
            pred_distribution = Counter(predicted_levels)

            # 2. Meaning preservation (vs original)
            mb_orig = evaluate_meaning_preservation(hyps, origs)
            bs_orig = evaluate_bertscore(hyps, origs)

            # 3. Similarity to reference (only if reference exists)
            if _has_reference and len(refs) == len(hyps):
                mb_ref = evaluate_meaning_preservation(hyps, refs)
                bs_ref = evaluate_bertscore(hyps, refs)
            else:
                mb_ref = None
                bs_ref = None

            eval_results_list.append({
                "experiment": _exp_name,
                "cefr_level": _cefr_level,
                "strategy": _strategy,
                "num_samples": len(hyps),
                "pred_distribution": dict(pred_distribution),
                "weighted_f1": cefr_metrics["weighted_f1"],
                "adj_accuracy": cefr_metrics["adj_accuracy"],
                "rmse": cefr_metrics["rmse"],
                "meaningbert_orig": mb_orig,
                "bertscore_orig": bs_orig,
                "meaningbert_ref": mb_ref,
                "bertscore_ref": bs_ref
            })

    eval_df = pd.DataFrame(eval_results_list)
    return eval_df, pd


@app.cell
def _(eval_df, mo):
    # ===== SECTION 1: Prediction Distribution (Primary) =====
    _summary_lines = ["## Evaluation Results - Prediction Distribution\n"]

    # Group by CEFR level
    for _level in sorted(eval_df["cefr_level"].unique()):
        _level_data = eval_df[eval_df["cefr_level"] == _level].sort_values("strategy")

        _summary_lines.append(f"### {_level.upper()} Level Results\n")
        _summary_lines.append("```")
        _summary_lines.append("Strategy              │ Prediction Distribution")
        _summary_lines.append("──────────────────────┼────────────────────────────────────────")

        for _, _row in _level_data.iterrows():
            _strat = _row["strategy"]
            _dist = _row["pred_distribution"]
            _total = _row["num_samples"]
            _target = _row["cefr_level"].upper()

            # Sort by count descending
            _sorted_preds = sorted(_dist.items(), key=lambda x: x[1], reverse=True)

            # Build distribution string
            _dist_parts = []
            for _pred_level, _count in _sorted_preds:
                _pct = round((_count / _total) * 100)
                _mark = "(✓)" if _pred_level.lower() == _target.lower() else ""
                _dist_parts.append(f"{_pred_level} {_mark} {_count}/{_total} {_pct}%")

            _dist_str = "  │ ".join(_dist_parts)
            _summary_lines.append(f"{_strat:21s} │ {_dist_str}")

        _summary_lines.append("```\n")

    _summary_lines.append("\n**Legend:** (✓) = Correct target level\n")

    # ===== SECTION 2: Detailed Metrics (Secondary - for reference) =====
    _summary_lines.append("\n---\n")
    _summary_lines.append("## Detailed Metrics (Reference)\n")
    _summary_lines.append("<details><summary>Click to expand</summary>\n")

    _has_ref = eval_df["meaningbert_ref"].notna().any()

    for _level in sorted(eval_df["cefr_level"].unique()):
        _level_data = eval_df[eval_df["cefr_level"] == _level].sort_values("strategy")

        _summary_lines.append(f"\n### {_level.upper()} Level\n")
        _summary_lines.append("```")

        if _has_ref:
            _summary_lines.append("Strategy              │  F1  │ Adj │ RMSE │ MB_orig│ MB_ref│ BS_orig│ BS_ref")
            _summary_lines.append("──────────────────────┼──────┼─────┼──────┼────────┼───────┼────────┼───────")
        else:
            _summary_lines.append("Strategy              │  F1  │ Adj │ RMSE │ MB_orig│ BS_orig")
            _summary_lines.append("──────────────────────┼──────┼─────┼──────┼────────┼────────")

        for _, _row in _level_data.iterrows():
            _strat = _row["strategy"]
            _f1 = _row["weighted_f1"]
            _adj = _row["adj_accuracy"]
            _rmse = _row["rmse"]
            _mb_o = _row["meaningbert_orig"]
            _bs_o = _row["bertscore_orig"]

            if _has_ref and _row["meaningbert_ref"] is not None:
                _mb_r = _row["meaningbert_ref"]
                _bs_r = _row["bertscore_ref"]
                _line = f"{_strat:21s} │{_f1:.3f} │{_adj:.3f} │{_rmse:.3f} │ {_mb_o:.3f} │{_mb_r:.3f} │ {_bs_o:.3f} │{_bs_r:.3f} "
            else:
                _line = f"{_strat:21s} │{_f1:.3f} │{_adj:.3f} │{_rmse:.3f} │ {_mb_o:.3f} │ {_bs_o:.3f} "

            _summary_lines.append(_line)

        _summary_lines.append("```")

    _summary_lines.append("\n</details>")

    _summary_text = "\n".join(_summary_lines)
    mo.md(_summary_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Raw Data Deep Dive

    Complete sample-level analysis with all metrics.
    """)
    return


@app.cell
def _(
    aligned_submissions,
    bertscore,
    cefr_labeler1,
    meaning_bert,
    mo,
    pd,
    trial_data,
):
    # Raw Data Report
    with mo.status.spinner("Computing detailed metrics for all experiments... (Please wait)"):
        _all_raw_data = []
        _gold_lookup = {item["text_id"]: item for item in trial_data}

        for _exp_name, _preds in aligned_submissions.items():
            _cefr_level, _strategy = _exp_name.split("_", 1)

            _valid_preds = [p for p in _preds if not p['simplified'].startswith("ERROR:")]

            _curr_hyps = [p['simplified'] for p in _valid_preds]
            _curr_ids = [p['text_id'] for p in _valid_preds]
            _curr_origs = [_gold_lookup[tid]['original'] for tid in _curr_ids]
            _curr_refs = [_gold_lookup[tid]['reference'] for tid in _curr_ids]
            _curr_targets = [_gold_lookup[tid]['target_cefr'] for tid in _curr_ids]

            if not _curr_hyps:
                continue

            _cefr_results = cefr_labeler1(_curr_hyps, top_k=None, truncation=True)

            _mb_res = meaning_bert.compute(predictions=_curr_hyps, references=_curr_origs)
            _mb_scores = _mb_res["scores"]

            _bs_res = bertscore.compute(predictions=_curr_hyps, references=_curr_refs, lang="en")
            _bs_scores = _bs_res["f1"]

            for _i, (_tid, _hyp, _orig, _ref, _target) in enumerate(zip(_curr_ids, _curr_hyps, _curr_origs, _curr_refs, _curr_targets)):
                _c_scores = {item['label']: item['score'] for item in _cefr_results[_i]}
                _pred_label = max(_c_scores, key=_c_scores.get)

                _all_raw_data.append({
                    "experiment": _exp_name,
                    "strategy": _strategy,
                    "target_cefr": _cefr_level,
                    "text_id": _tid,
                    "original": _orig,
                    "simplified": _hyp,
                    "reference": _ref,
                    "gold_cefr": _target,
                    "pred_cefr": _pred_label,
                    "conf_score": round(_c_scores[_pred_label], 4),
                    "meaningbert": round(_mb_scores[_i], 2),
                    "bertscore": round(_bs_scores[_i], 4),
                    "prob_A1": round(_c_scores.get("A1", 0), 4),
                    "prob_A2": round(_c_scores.get("A2", 0), 4),
                    "prob_B1": round(_c_scores.get("B1", 0), 4),
                    "prob_B2": round(_c_scores.get("B2", 0), 4)
                })

        _df_raw = pd.DataFrame(_all_raw_data)

    mo.vstack([
        mo.md(f"### Full Raw Data Report ({len(_df_raw)} samples)"),
        mo.md("Below is the complete raw scoring data. Use the table controls to filter or download."),
        mo.ui.table(_df_raw, selection=None)
    ])
    return


if __name__ == "__main__":
    app.run()
