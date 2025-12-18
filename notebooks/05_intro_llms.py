import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # The env file needs
    # KITEGG_API_KEY
    # GEMINI_API_KEY
    # GOOGLE_APPLICATION_CREDENTIALS (path to json)

    # API configuration
    API_BASE = "https://chat1.kitegg.de/api"
    API_KEY = os.environ.get("KITEGG_API_KEY")
    return API_BASE, API_KEY, os


@app.cell(hide_code=True)
def _(mo):
    # Temperature: Controls randomness (0.0 = deterministic, 2.0 = very random)
    temperature_slider = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.1,
        value=0.7,
        label="Temperature",
        show_value=True,
    )

    # Top P: Nucleus sampling (0.0 = most conservative, 1.0 = most diverse)
    top_p_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.9,
        label="Top P (Nucleus Sampling)",
        show_value=True,
    )

    # Max Tokens: Maximum length of generated response
    max_tokens_slider = mo.ui.slider(
        start=128,
        stop=1024,
        step=128,
        value=512,
        label="Max Tokens",
        show_value=True,
    )

    # Frequency Penalty: Reduces repetition (-2.0 to 2.0)
    frequency_penalty_slider = mo.ui.slider(
        start=-2.0,
        stop=2.0,
        step=0.1,
        value=0.0,
        label="Frequency Penalty",
        show_value=True,
    )

    # Presence Penalty: Encourages topic diversity (-2.0 to 2.0)
    presence_penalty_slider = mo.ui.slider(
        start=-2.0,
        stop=2.0,
        step=0.1,
        value=0.0,
        label="Presence Penalty",
        show_value=True,
    )

    mo.vstack(
        [
            temperature_slider,
            top_p_slider,
            max_tokens_slider,
            frequency_penalty_slider,
            presence_penalty_slider,
        ]
    )
    return (
        frequency_penalty_slider,
        max_tokens_slider,
        presence_penalty_slider,
        temperature_slider,
        top_p_slider,
    )


@app.cell(hide_code=True)
def _(mo):
    # Default example sentence - a complex sentence that needs simplification
    default_text = "Nevertheless, Tagore emulated numerous styles, including craftwork from northern New Ireland, Haida carvings from the west coast of Canada (British Columbia), and woodcuts by Max Pechstein."

    text_input = mo.ui.text_area(
        value=default_text, label="Text to Simplify", full_width=True, rows=4
    )
    text_input
    return (text_input,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Using Gemini via Vertex AI

    First, we test the Gemini models directly. We use the package [LiteLLM](https://docs.litellm.ai/) that gives us universal access to most of the LLM providers. Also, we use the Gemini API via Vertex AI to make use of our Educational Credits. There is also the Google GenAI platform, but this uses a different budget.
    """)
    return


@app.cell
def _(mo):
    # List of available Gemini models
    gemini_models = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    gemini_dropdown = mo.ui.dropdown(
        options=gemini_models,
        value=gemini_models[0],
        label="Select Gemini Model",
    )

    gemini_dropdown
    return (gemini_dropdown,)


@app.cell
def _(
    frequency_penalty_slider,
    gemini_dropdown,
    max_tokens_slider,
    presence_penalty_slider,
    temperature_slider,
    text_input,
    top_p_slider,
):
    from litellm import completion as gemini_completion

    # Use the selected Gemini model
    if gemini_dropdown.value and text_input.value:
        gemini_response = gemini_completion(
            model=f"vertex_ai/{gemini_dropdown.value}",
            messages=[
                {
                    "role": "system",
                    "content": "You are a text simplification assistant. Simplify the following text while preserving its meaning. Do not say anything else.",
                },
                {"role": "user", "content": f"Simplify: {text_input.value}"},
            ],
            vertex_project="chatbot-183407",
            vertex_location="us-central1",
            temperature=temperature_slider.value,
            top_p=top_p_slider.value,
            max_tokens=max_tokens_slider.value,
            frequency_penalty=frequency_penalty_slider.value,
            presence_penalty=presence_penalty_slider.value,
        )

        gemini_simplified = gemini_response.choices[0].message.content
    return (gemini_simplified,)


@app.cell
def _(gemini_simplified, mo, text_input):
    mo.md(f"""
    ### Original Text

    {text_input.value}

    ---

    ### Simplified Text (Gemini)

    {gemini_simplified}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Batch Prediction

    Next, we can process many sentences at once by utilizing the batch prediction feature. As an example, we simplify the first 20 sentences in the ASSET test set. Batch processing offers 50% cost savings compared to real-time inference.


    **Prerequisites:**
    1. Create a service account on your Google Cloud Platform.
    2. The service account needs these roles:
       - Vertex AI Batch Prediction Service
       - Vertex AI Platform Express User
       - Vertex AI Tuning Service Agent
    3. Download a key for the service account as json and add the path as `GOOGLE_APPLICATION_CREDENTIALS` to your `.env` file.
    4. Update the `bucket_name` variable below if using a different bucket

    **How it works:**
    1. Load 20 sentences from the ASSET test file
    2. Create a JSONL file with batch requests
    3. Upload to Google Cloud Storage
    4. Submit batch job to Gemini
    5. Monitor job status and retrieve results when complete

    **Note**

    I noticed in my first runs that gemini-flash-2.5 used thought tokens and somehow the `thinkingConfig` parameter is not accpected for batch jobs (there is no documentation on this, might be a bug) and I couldn't turn off thinking, so I switchted to gemini-flash-2.5-lite for now.

    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini
    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-cloud-storage#create-batch-job-python_genai_sdk
    """)
    return


@app.cell
def _():
    from google.cloud import storage

    # TODO: Enter your project id here
    client = storage.Client(project="chatbot-183407")
    bucket_name = "anlp-ws202526-batch"

    try:
        bucket = client.get_bucket(bucket_name)
        print(f"✓ Bucket {bucket_name} already exists")
    except Exception as e:
        print(f"Creating bucket {bucket_name}...")
        bucket = client.create_bucket(bucket_name, location="us-central1")
        print(f"✓ Bucket {bucket_name} created successfully in us-central1")
    return bucket, bucket_name


@app.cell
def _():
    # Read the first 20 sentences from the ASSET test file
    import json

    asset_file_path = "/Users/richard/Lehre/anlp-ws202526/repos/anlp-ws202526/data/asset/asset.test.orig"

    with open(asset_file_path, "r") as f:
        lines = f.readlines()

    # Extract just the text (remove line numbers)
    sentences = [line.strip() for line in lines[:20]]

    print(f"Loaded {len(sentences)} sentences for batch processing")
    return json, sentences


@app.cell
def _(
    frequency_penalty_slider,
    json,
    max_tokens_slider,
    presence_penalty_slider,
    sentences,
    temperature_slider,
    top_p_slider,
):
    # Create JSONL file for batch processing
    import tempfile

    batch_requests = []
    for idx, sentence in enumerate(sentences):
        request = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": f"Simplify the following text while preserving its meaning. Don't say anythin else. \n\n{sentence}"
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": temperature_slider.value,
                    "topP": top_p_slider.value,
                    "maxOutputTokens": max_tokens_slider.value,
                    "frequencyPenalty": frequency_penalty_slider.value,
                    "presencePenalty": presence_penalty_slider.value,
                    "thinkingConfig": {"thinkingBudget": 0},  # Disable thinking
                },
            }
        }
        batch_requests.append(request)

    # Create temporary JSONL file
    jsonl_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    for request in batch_requests:
        jsonl_file.write(json.dumps(request) + "\n")
    jsonl_file.close()

    print(
        f"Created JSONL file with {len(batch_requests)} requests at: {jsonl_file.name}"
    )
    return (jsonl_file,)


@app.cell
def _(bucket, bucket_name, jsonl_file, os):
    # Upload input file
    input_blob_name = "batch_input/simplification_batch_input.jsonl"
    input_blob = bucket.blob(input_blob_name)
    input_blob.upload_from_filename(jsonl_file.name)

    input_gcs_uri = f"gs://{bucket_name}/{input_blob_name}"

    print(f"Uploaded input file to: {input_gcs_uri}")

    # Clean up temp file
    os.unlink(jsonl_file.name)
    return (input_gcs_uri,)


@app.cell
def _(bucket_name, gemini_dropdown, input_gcs_uri):
    # Create and submit batch job using google-genai SDK
    from google import genai
    from google.genai.types import CreateBatchJobConfig
    from datetime import datetime

    # Create client and submit batch job
    genai_client = genai.Client(vertexai=True)

    # Create unique output directory using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_gcs_uri = f"gs://{bucket_name}/batch_output/{timestamp}/"

    batch_job = genai_client.batches.create(
        model=gemini_dropdown.value,
        src=input_gcs_uri,
        config=CreateBatchJobConfig(dest=output_gcs_uri),
    )

    print(f"Batch job created successfully!")
    print(f"Job name: {batch_job.name}")
    print(f"Job state: {batch_job.state}")
    print(f"Job create time: {batch_job.create_time}")
    print(f"Output will be saved to: {output_gcs_uri}")
    return batch_job, genai_client, output_gcs_uri


@app.cell
def _(batch_job, genai_client, mo):
    # Monitor batch job status with auto-refresh
    from google.genai.types import JobState
    import time

    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
    }

    # Poll job status every 5 seconds until completion
    current_job = genai_client.batches.get(name=batch_job.name)
    check_count = 0

    if current_job.state not in completed_states:
        with mo.status.spinner(
            title="Monitoring Batch Job",
            subtitle=f"Job state: {current_job.state.name}",
        ) as spinner:
            while current_job.state not in completed_states:
                check_count += 1
                elapsed_time = check_count * 5

                # Update spinner with current status
                spinner.update(
                    subtitle=f"Job state: {current_job.state.name} | Checks: {check_count} | Elapsed: {elapsed_time}s"
                )

                # Wait 5 seconds before next check
                time.sleep(5)

                # Fetch updated job status
                current_job = genai_client.batches.get(name=batch_job.name)

                # Safety limit: stop checking after ~5 minutes (60 checks)
                if check_count >= 60:
                    break

    # Display final status
    if current_job.state == JobState.JOB_STATE_SUCCEEDED:
        status_icon = "✅"
        status_color = "green"
    elif current_job.state == JobState.JOB_STATE_FAILED:
        status_icon = "❌"
        status_color = "red"
    elif current_job.state == JobState.JOB_STATE_CANCELLED:
        status_icon = "⚠️"
        status_color = "orange"
    else:
        status_icon = "🔄"
        status_color = "blue"

    mo.md(f"""
    ### Batch Job Status {status_icon}

    **State:** {current_job.state.name}
    **Job Name:** {current_job.name}
    **Create Time:** {current_job.create_time}
    """)
    return JobState, current_job


@app.cell
def _(JobState, bucket, current_job, genai_client, json, mo, output_gcs_uri):
    job_status = genai_client.batches.get(name=current_job.name)

    if job_status.state == JobState.JOB_STATE_SUCCEEDED:
        # Extract the path from the GCS URI (remove gs://bucket_name/)
        output_path = "/".join(output_gcs_uri.split("/")[3:])

        blobs = list(bucket.list_blobs(prefix=output_path))

        results = []
        errors = []
        for blob in blobs:
            if blob.name.endswith(".jsonl"):
                content = blob.download_as_text()
                for idx2, line in enumerate(content.strip().split("\n")):
                    if line:
                        result = json.loads(line)

                        # Check for status errors first
                        if result.get("status"):
                            error_msg = result["status"]
                            errors.append(
                                f"Sentence {idx2 + 1}: API Error - {error_msg}"
                            )
                            results.append(f"[Error: {error_msg}]")
                            continue

                        if (
                            "response" in result
                            and "candidates" in result["response"]
                        ):
                            candidate = result["response"]["candidates"][0]
                            finish_reason = candidate.get(
                                "finishReason", "UNKNOWN"
                            )

                            # Check for different response structures
                            if (
                                "content" in candidate
                                and "parts" in candidate["content"]
                            ):
                                # Standard response with parts array
                                simplified = candidate["content"]["parts"][0][
                                    "text"
                                ]
                                if finish_reason == "MAX_TOKENS":
                                    errors.append(
                                        f"Sentence {idx2 + 1}: Truncated (MAX_TOKENS) - partial result shown"
                                    )
                                    results.append(f"[TRUNCATED] {simplified}")
                                else:
                                    results.append(simplified)
                            elif finish_reason == "MAX_TOKENS":
                                # Response was truncated and no text generated
                                # This happens when thinking tokens consume the entire budget
                                usage = result["response"].get("usageMetadata", {})
                                thinking_tokens = usage.get(
                                    "thoughtsTokenCount", 0
                                )
                                errors.append(
                                    f"Sentence {idx2 + 1}: MAX_TOKENS - thinking used {thinking_tokens} tokens, no output generated"
                                )
                                results.append(
                                    f"[Error: Response truncated - model thinking consumed all {thinking_tokens} tokens]"
                                )
                            else:
                                # Other error or unexpected structure
                                errors.append(
                                    f"Sentence {idx2 + 1}: {finish_reason}"
                                )
                                results.append(f"[Error: {finish_reason}]")
                        elif "error" in result:
                            errors.append(
                                f"Sentence {idx2 + 1}: {result['error']}"
                            )
                            results.append(
                                f"[Error: {result['error'].get('message', 'Unknown error')}]"
                            )

        results_display = "\n\n".join(
            [f"**{i + 1}.** {result}" for i, result in enumerate(results)]
        )

        error_display = ""
        if errors:
            error_display = (
                f"\n\n**Errors encountered:** {len(errors)}\n"
                + "\n".join([f"- {e}" for e in errors])
            )

        md = mo.md(f"""
        ### Batch Processing Results

        Successfully processed {len(results)} sentences:{error_display}

        {results_display}
        """)
    else:
        md = mo.md(f"""
        ### Job Status: {job_status.state.name}

        Job is still processing or has not completed successfully.
        Please wait and refresh this cell to check for results.
        """)

    md
    return


@app.cell
def _(mo):
    mo.md("""
    ## Using the KITeGG Cluster

    Configure the LLM API connection:
    """)
    return


@app.cell
def _(API_BASE, API_KEY, mo):
    import requests


    # IMPORTANT: Only works from inside campus environment or via VPN
    # Fetch available models
    def get_models():
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.get(f"{API_BASE}/models", headers=headers)
        models_data = response.json()
        # Extract model IDs from the response
        return [model["id"] for model in models_data.get("data", [])]


    available_models = get_models()

    # Create dropdown for model selection
    model_dropdown = mo.ui.dropdown(
        options=available_models,
        value=available_models[0] if available_models else None,
        label="Select Model",
    )

    model_dropdown
    return model_dropdown, requests


@app.cell
def _(
    API_BASE,
    API_KEY,
    frequency_penalty_slider,
    max_tokens_slider,
    model_dropdown,
    presence_penalty_slider,
    requests,
    temperature_slider,
    text_input,
    top_p_slider,
):
    def simplify_text(
        text,
        model,
        temperature,
        top_p,
        max_tokens,
        frequency_penalty,
        presence_penalty,
    ):
        """Simplify text using the selected model with specified parameters"""
        url = f"{API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        system_message = """You are a text simplification assistant. Your task is to simplify complex sentences while preserving their meaning. Dont say anything else."""

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Simplify the following text:\n\n{text}",
                },
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()


    # Use selected model and parameters from UI
    if model_dropdown.value and text_input.value:
        api_result = simplify_text(
            text=text_input.value,
            model=model_dropdown.value,
            temperature=temperature_slider.value,
            top_p=top_p_slider.value,
            max_tokens=max_tokens_slider.value,
            frequency_penalty=frequency_penalty_slider.value,
            presence_penalty=presence_penalty_slider.value,
        )

        # Extract the simplified text from the response
        simplified_text = (
            api_result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    return (simplified_text,)


@app.cell
def _(mo, simplified_text, text_input):
    mo.md(f"""
    ### Original Text

    {text_input.value}

    ---

    ### Simplified Text

    {simplified_text}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Text Simplification with LiteLLM

    Using LiteLLM for more flexible model access:
    """)
    return


@app.cell
def _(
    API_BASE,
    API_KEY,
    frequency_penalty_slider,
    max_tokens_slider,
    model_dropdown,
    presence_penalty_slider,
    temperature_slider,
    text_input,
    top_p_slider,
):
    from litellm import completion

    # Use the selected model from dropdown with parameters
    if model_dropdown.value and text_input.value:
        litellm_response = completion(
            model=f"openai/{model_dropdown.value}",
            messages=[
                {
                    "role": "system",
                    "content": "You are a text simplification assistant. Simplify the following text while preserving its meaning. Dont say anything else.",
                },
                {"role": "user", "content": f"Simplify: {text_input.value}"},
            ],
            api_base=API_BASE,
            api_key=API_KEY,
            temperature=temperature_slider.value,
            top_p=top_p_slider.value,
            max_tokens=max_tokens_slider.value,
            frequency_penalty=frequency_penalty_slider.value,
            presence_penalty=presence_penalty_slider.value,
        )

        litellm_simplified = litellm_response.choices[0].message.content
    return completion, litellm_simplified


@app.cell
def _(litellm_simplified, mo, text_input):
    mo.md(f"""
    ### Original Text

    {text_input.value}

    ---

    ### Simplified Text (LiteLLM)

    {litellm_simplified}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Structured Output with Instructor

    Using Instructor to get structured simplification results with metadata:
    """)
    return


@app.cell
def _(
    API_BASE,
    API_KEY,
    completion,
    frequency_penalty_slider,
    max_tokens_slider,
    model_dropdown,
    presence_penalty_slider,
    temperature_slider,
    text_input,
    top_p_slider,
):
    import instructor
    from pydantic import BaseModel, Field

    instructor_client = instructor.from_litellm(completion)


    class SimplificationResult(BaseModel):
        """Structured result for text simplification"""

        simplified_text: str = Field(
            description="The simplified version of the input text"
        )


    def simplify_with_structure(
        text: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
    ):
        """Simplify text and return structured result with metadata"""
        return instructor_client.chat.completions.create(
            model=f"openai/{model}",
            api_base=API_BASE,
            api_key=API_KEY,
            response_model=SimplificationResult,
            messages=[
                {
                    "role": "system",
                    "content": "You are a text simplification assistant. ",
                },
                {"role": "user", "content": f"Simplify this text:\n\n{text}"},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_retries=3,
        )


    # Use selected model from dropdown
    if model_dropdown.value and text_input.value:
        structured_result = simplify_with_structure(
            text=text_input.value,
            model=model_dropdown.value,
            temperature=temperature_slider.value,
            top_p=top_p_slider.value,
            max_tokens=max_tokens_slider.value,
            frequency_penalty=frequency_penalty_slider.value,
            presence_penalty=presence_penalty_slider.value,
        )
    return (structured_result,)


@app.cell
def _(mo, structured_result, text_input):
    mo.md(f"""
    ### Original Text

    {text_input.value}

    ---

    ### Simplified Text (Structured)

    {structured_result.simplified_text}
    """)
    return


if __name__ == "__main__":
    app.run()
