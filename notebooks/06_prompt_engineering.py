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

    # The env file needs:
    # GEMINI_API_KEY
    # GOOGLE_APPLICATION_CREDENTIALS (path to json)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prompt Engineering Fundamentals

    In this tutorial we go over some basic prompt engineering techniques.

    ## References

    This tutorial is based on some resources on prompt engineering:
    - [Anthropic's Prompt Engineering Tutorial](https://github.com/anthropics/courses/tree/master/prompt_engineering_interactive_tutorial/Anthropic%201P)
    - [Lilian Weng's Prompt Engineering Guide](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
    - [Eugene Yan's Prompting Fundamentals](https://eugeneyan.com/writing/prompting/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example Sentence

    Throughout this tutorial, we'll use this complex sentence that needs simplification:
    """)
    return


@app.cell
def _(mo):
    # Default example sentence
    default_sentence = "In architectural decoration Small pieces of colored and iridescent shell have been used to create mosaics and inlays, which have been used to decorate walls, furniture and boxes."

    sentence_input = mo.ui.text_area(
        value=default_sentence, label="Input Sentence", full_width=True, rows=3
    )
    sentence_input
    return (sentence_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: LLM Configuration

    As in the previous tutorial, we'll use Gemini via Vertex AI with LiteLLM for all examples.
    """)
    return


@app.cell
def _(mo):
    # Model selection
    gemini_models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    model_selector = mo.ui.dropdown(
        options=gemini_models, value=gemini_models[-2], label="Select Model"
    )

    # Temperature slider
    temperature_slider = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.1,
        value=0.0,
        label="Temperature",
        show_value=True,
    )

    mo.vstack([model_selector, temperature_slider])
    return model_selector, temperature_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 1. Being Clear and Direct

    ## The Foundation of Good Prompts

    The most fundamental principle in prompt engineering is to be **clear and direct**. Vague or ambiguous instructions lead to unpredictable outputs.

    ### Key Principles:
    - **Say exactly what you want**: Don't make the model guess your intent
    - **Be specific about format**: Tell the model exactly how to structure output
    - **Use concrete examples**: Show, don't just tell
    - **Avoid ambiguity**: Use precise language
    - **Use bullet or numbered lists** for instructions

    Let's see this in action with text simplification.
    """)
    return


@app.cell
def _(sentence_input):
    clear_prompt = f"""Your task is to simplify complex text for a general audience.

    Input text:
    {sentence_input.value}

    Instructions:
    - Replace complex words with simpler alternatives
    - Keep the same meaning
    - Use everyday language
    - Output only the simplified text, nothing else"""

    print(clear_prompt)
    return (clear_prompt,)


@app.cell
def _(clear_prompt, model_selector, temperature_slider):
    from litellm import completion


    clear_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[{"role": "user", "content": clear_prompt}],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
    )
    clear_result = clear_response.choices[0].message.content
    return clear_result, completion


@app.cell(hide_code=True)
def _(clear_result, mo, sentence_input):
    mo.md(f"""
    **Result with clear prompt:**

    Original: {sentence_input.value}

    Simplified: {clear_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. System, User, and Assistant Roles

    ## Understanding Message Roles

    Modern LLMs use a conversation structure with three main roles:

    - **System**: Sets the overall behavior and context (the "operating instructions")
    - **User**: The person asking questions or giving tasks
    - **Assistant**: The AI's responses

    The **system message** is crucial for defining how the model should behave. It's like hiring instructions for an employee - you specify their role, responsibilities, and expectations.

    Usually it is best to describe the overall task in the system prompt and then use the user message to give the specific input.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: System Message

    Now let's create a focused system message that clearly defines the task and expected behavior:
    """)
    return


@app.cell
def _():
    simplification_system_prompt = """You are a text simplification expert. Your job is to rewrite complex sentences into simpler versions that are easier to understand.

    Rules:
    1. Preserve all original meaning
    2. Use simpler words and shorter sentences
    3. Remove unnecessary jargon
    4. Output only the simplified text, no explanations"""
    print(simplification_system_prompt)
    return (simplification_system_prompt,)


@app.cell
def _(
    completion,
    model_selector,
    sentence_input,
    simplification_system_prompt,
    temperature_slider,
):
    response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {"role": "system", "content": simplification_system_prompt},
            # Here is the user message
            {"role": "user", "content": f"Simplify:q {sentence_input.value}"},
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
    )
    result = response.choices[0].message.content
    return


@app.cell(hide_code=True)
def _(clear_result, mo, sentence_input):
    mo.md(f"""
    **Result with system message:**

    Original: {sentence_input.value}

    Simplified: {clear_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Separating Data from Instructions

    ## Why Separation Matters

    One crucial prompt engineering principle is **clearly separating your instructions from the data you want processed**. When these are mixed together, models can:
    - Confuse data for instructions (or vice versa)
    - Process instructions as if they were data
    - Miss important details

    ## Using XML Tags

    **XML tags are particularly effective** (especially with Claude/Gemini) for creating clear boundaries. They're unambiguous and easy for both humans and models to parse.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Using XML Tags for Clear Separation

    XML tags create unambiguous boundaries between instructions and data:
    """)
    return


@app.cell
def _(sentence_input):
    xml_prompt = f"""Your task is to simplify complex text for a general audience.

    Rules:
    - Replace complex words with simpler alternatives
    - Keep the same meaning
    - Use everyday language
    - Output only the simplified text


    <text_to_simplify>
    {sentence_input.value}
    </text_to_simplify>"""
    print(xml_prompt)
    return (xml_prompt,)


@app.cell
def _(completion, model_selector, temperature_slider, xml_prompt):
    xml_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification assistant.",
            },
            {"role": "user", "content": xml_prompt},
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
    )
    xml_result = xml_response.choices[0].message.content
    return (xml_result,)


@app.cell(hide_code=True)
def _(mo, sentence_input, xml_result):
    mo.md(f"""
    **Result:**

    Original: {sentence_input.value}

    Simplified: {xml_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Formatting Best Practices

    **1. Use XML tags for clear separation:**
    ```
    <instructions>
    [what to do]
    </instructions>

    <data>
    [content to process]
    </data>
    ```

    **2. Triple quotes for text blocks:**
    ```
    Input text:
    \"\"\"
    [multi-line text here]
    \"\"\"
    ```

    **3. Structured headers:**
    ```
    Task: [what to do]
    Input: [data]
    Requirements: [constraints]
    Output format: [expected format]
    ```

    **Common XML tag patterns:**
    - `<document>`, `<text>`, `<input>` for data
    - `<instructions>`, `<rules>`, `<guidelines>` for directions
    - `<example>`, `<examples>` for demonstrations
    - `<context>`, `<background>` for additional info
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Prefilling



    Prefilling means starting the assistant's response with specific text. This ensures:
    - The output begins exactly as you want
    - The model continues in the established pattern
    - More predictable, parseable outputs

    ### Prefilling Use Cases

    **When to prefill:**
    - Forcing specific output format (JSON, XML, CSV) - nowadays most models have a json mode as we see below so this is no longer that important
    - Skipping preambles or explanations
    - Ensuring outputs are immediately parseable
    - Controlling response style or tone

    **Common prefill patterns:**
    - `{` for JSON objects
    - `[` for JSON arrays
    - `<result>` for XML output
    - `Sure,` to make responses sound agreeable
    - `"` for quoted strings

    **Note:** Prefilling counts as part of the assistant's response and uses output tokens.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Using Prefilling for JSON Output

    Let's prefill the response to start with `{` to ensure clean JSON output:
    """)
    return


@app.cell
def _(completion, model_selector, sentence_input, temperature_slider):
    prefill_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification assistant.",
            },
            {
                "role": "user",
                "content": f"""Simplify this text and return the result as JSON with fields "original" and "simplified":

    <text>
    {sentence_input.value}
    </text>""",
            },
            {"role": "assistant", "content": "{"},  # Prefill the response
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
    )
    prefill_result = "{" + prefill_response.choices[0].message.content
    return (prefill_result,)


@app.cell(hide_code=True)
def _(mo, prefill_result, sentence_input):
    mo.md(f"""
    **Result:**

    Original: {sentence_input.value}

    JSON Output:
    ```
    {prefill_result}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Few-Shot Learning

    ## Learning by Example

    Few-shot prompting means providing examples of the task before asking the model to perform it. This is especially powerful for:
    - Establishing a specific style or format
    - Demonstrating complex transformations
    - Showing edge cases

    For text simplification, examples help the model understand the desired simplification level.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zero-Shot (No Examples)

    First, let's try without any examples:
    """)
    return


@app.cell
def _(completion, model_selector, sentence_input, temperature_slider):
    zero_shot_prompt = f"""Simplify this sentence:

    {sentence_input.value}"""

    zero_shot_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification assistant. Output only the simplified text.",
            },
            {"role": "user", "content": zero_shot_prompt},
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
    )
    zero_shot_result = zero_shot_response.choices[0].message.content
    return (zero_shot_result,)


@app.cell(hide_code=True)
def _(mo, sentence_input, zero_shot_result):
    mo.md(f"""
    **Zero-shot result:**

    Original: {sentence_input.value}

    Simplified: {zero_shot_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Few-Shot (With Examples)

    Now with 3 examples showing the desired simplification style.

    You can also use the examples directly from a simplificationd dataset like ASSET or our trial data.
    """)
    return


@app.cell
def _(completion, model_selector, sentence_input, temperature_slider):
    few_shot_system_prompt = """You are a text simplification expert. Your job is to rewrite complex sentences into simpler versions that are easier to understand.

    Rules:
    1. Preserve all original meaning
    2. Use simpler words and shorter sentences
    3. Remove unnecessary jargon
    4. Output only the simplified text, no explanations

    <examples>
    <example>
    Original: "The physician administered medication to ameliorate the patient's symptoms."
    Simplified: "The doctor gave medicine to help the patient feel better."
    </example>

    <example>
    Original: "Despite the inclement weather conditions, the expedition proceeded as scheduled."
    Simplified: "Even though the weather was bad, the trip went ahead as planned."
    </example>

    <example>
    Original: "The corporation implemented a comprehensive restructuring initiative to optimize operational efficiency."
    Simplified: "The company changed how it works to do things better and faster."
    </example>
    </examples>
    """

    few_shot_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": few_shot_system_prompt,
            },
            {
                "role": "user",
                "content": f"Simplify the following sentence: {sentence_input.value}",
            },
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
    )
    few_shot_result = few_shot_response.choices[0].message.content
    return (few_shot_result,)


@app.cell(hide_code=True)
def _(few_shot_result, mo, sentence_input):
    mo.md(f"""
    **Few-shot result:**

    Original: {sentence_input.value}

    Simplified: {few_shot_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using the Assistant Role for Examples

    You can also format examples using the conversation structure:
    """)
    return


@app.cell
def _(completion, model_selector, sentence_input, temperature_slider):
    conversational_few_shot = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification assistant. Output only the simplified text.",
            },
            {
                "role": "user",
                "content": "Simplify: The physician administered medication to ameliorate the patient's symptoms.",
            },
            {
                "role": "assistant",
                "content": "The doctor gave medicine to help the patient feel better.",
            },
            {
                "role": "user",
                "content": "Simplify: Despite the inclement weather conditions, the expedition proceeded as scheduled.",
            },
            {
                "role": "assistant",
                "content": "Even though the weather was bad, the trip went ahead as planned.",
            },
            {"role": "user", "content": f"Simplify: {sentence_input.value}"},
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
    )
    conversational_result = conversational_few_shot.choices[0].message.content
    return (conversational_result,)


@app.cell(hide_code=True)
def _(conversational_result, mo, sentence_input):
    mo.md(f"""
    **Conversational few-shot result:**

    Original: {sentence_input.value}

    Simplified: {conversational_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Chain of Thought (CoT) Reasoning

    ## Making Thinking Explicit

    Chain of Thought prompting encourages the model to break down complex problems into steps. This typically improves:
    - Reasoning accuracy
    - Output quality for complex tasks
    - Transparency (you can see the "thinking")

    For text simplification, CoT can help identify which parts need simplification and why.

    Many models as of January 2026 are reasoning models or decide based on the prompt whether to use reasoning (GPT5). These models take more time and tokens during inference to automatically solve a task step by step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zero-Shot Chain of Thought

    Simply adding "Let's think step by step" can trigger reasoning:
    """)
    return


@app.cell
def _(completion, model_selector, sentence_input, temperature_slider):
    zero_shot_cot_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification assistant.",
            },
            {
                "role": "user",
                "content": f"""Simplify this sentence:

    {sentence_input.value}

    Let's think step by step about how to simplify it.""",
            },
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=5000,
    )
    zero_shot_cot_result = zero_shot_cot_response.choices[0].message.content
    return (zero_shot_cot_result,)


@app.cell(hide_code=True)
def _(mo, sentence_input, zero_shot_cot_result):
    mo.md(f"""
    **Zero-shot CoT:**

    Original: {sentence_input.value}

    Result with reasoning:

    {zero_shot_cot_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standard Chain of Thought

    Provide an example that includes the reasoning steps:
    """)
    return


@app.cell
def _(completion, model_selector, sentence_input, temperature_slider):
    standard_cot_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification assistant.",
            },
            {
                "role": "user",
                "content": """Simplify this sentence:

    "The physician administered medication to ameliorate the patient's symptoms."

    Let me think through this:
    1. "physician" is a complex medical term → "doctor"
    2. "administered" is formal → "gave"
    3. "medication" is clear, keep it → "medicine" is simpler
    4. "ameliorate" is very formal → "help" or "improve"
    5. "patient's symptoms" is clear enough → "patient feel better"

    Simplified: "The doctor gave medicine to help the patient feel better."

    Now simplify this sentence:

    """
                + sentence_input.value,
            },
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=5000,
    )
    standard_cot_result = standard_cot_response.choices[0].message.content
    return (standard_cot_result,)


@app.cell(hide_code=True)
def _(mo, sentence_input, standard_cot_result):
    mo.md(f"""
    **Standard CoT with example:**

    Original: {sentence_input.value}

    Result with reasoning:

    {standard_cot_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Structured Chain of Thought

    Request a specific reasoning format:
    """)
    return


@app.cell
def _(completion, model_selector, sentence_input, temperature_slider):
    structured_cot_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification assistant.",
            },
            {
                "role": "user",
                "content": f"""Simplify this sentence using this format:

    Original: {sentence_input.value}

    Analysis:
    1. Identify complex words: [list them]
    2. Identify complex structures: [list them]
    3. Proposed replacements: [list them]

    Simplified version: [final output]""",
            },
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=5000,
    )
    structured_cot_result = structured_cot_response.choices[0].message.content
    return (structured_cot_result,)


@app.cell(hide_code=True)
def _(mo, sentence_input, structured_cot_result):
    mo.md(f"""
    **Structured CoT:**

    Original: {sentence_input.value}

    Result with analysis:

    {structured_cot_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 7. Structured Outputs

    ## From Text to Data

    Often we need more than just text - we want structured data we can process programmatically. Modern LLMs can output:
    - Valid JSON
    - Structured objects matching schemas
    - Data with specific fields and types
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.1 Native JSON Mode

    Many LLM APIs now support JSON mode natively. With LiteLLM, you can use `response_format` to request JSON output.

    **References:**
    - [LiteLLM JSON Mode Documentation](https://docs.litellm.ai/docs/completion/json_mode)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: JSON Mode

    Request the output in a specific JSON structure:
    """)
    return


@app.cell
def _(completion, model_selector, sentence_input, temperature_slider):
    import json

    json_mode_response = completion(
        model=f"vertex_ai/{model_selector.value}",
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification assistant. Always respond with valid JSON.",
            },
            {
                "role": "user",
                "content": f"""Simplify this sentence and return the result as JSON with these fields:
    - original: the original sentence
    - simplified: the simplified version
    - changes_made: list of key changes

    Sentence: {sentence_input.value}""",
            },
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    json_mode_raw = json_mode_response.choices[0].message.content
    try:
        json_mode_parsed = json.loads(json_mode_raw)
    except:
        json_mode_parsed = {"error": "Failed to parse JSON", "raw": json_mode_raw}
    return (json_mode_parsed,)


@app.cell(hide_code=True)
def _(json_mode_parsed, mo):
    mo.md(f"""
    **JSON Mode Output:**

    {mo.json(json_mode_parsed)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Limitations of JSON Mode

    While JSON mode is useful, it has some limitations:
    - No schema validation (model might return wrong field names)
    - No type checking (strings where you expect numbers)
    - Manual parsing and error handling required
    - No automatic retries on invalid outputs

    **This is where Instructor comes in!**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.2 Instructor: Type-Safe Structured Outputs

    [Instructor](https://github.com/jxnl/instructor) is a library that wraps LLM APIs to provide:
    - Type-safe Pydantic models
    - Automatic validation
    - Automatic retries on failures
    - Better error messages

    **References:**
    - [LiteLLM + Instructor Tutorial](https://docs.litellm.ai/docs/tutorials/instructor)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define the Output Schema

    First, we define what we want using Pydantic:
    """)
    return


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing import List


    class SimplificationResult(BaseModel):
        """Structured result for text simplification with metadata"""

        original_text: str = Field(description="The original input sentence")
        simplified_text: str = Field(description="The simplified version")
        complexity_score_before: int = Field(
            description="Estimated complexity before (1-10, 10=most complex)",
            ge=1,
            le=10,
        )
        complexity_score_after: int = Field(
            description="Estimated complexity after (1-10, 10=most complex)",
            ge=1,
            le=10,
        )
        changes_made: List[str] = Field(
            description="List of specific changes made during simplification"
        )
        difficult_words_replaced: dict[str, str] = Field(
            description="Mapping of difficult words to their simpler replacements"
        )
    return (SimplificationResult,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Use Instructor to Get Structured Output

    Now we use Instructor to enforce this schema:
    """)
    return


@app.cell
def _(
    SimplificationResult,
    completion,
    model_selector,
    sentence_input,
    temperature_slider,
):
    import instructor

    # Wrap LiteLLM with Instructor
    instructor_client = instructor.from_litellm(completion)

    # Get structured output
    instructor_result = instructor_client.chat.completions.create(
        model=f"vertex_ai/{model_selector.value}",
        response_model=SimplificationResult,
        messages=[
            {
                "role": "system",
                "content": "You are a text simplification expert. Analyze and simplify the given text.",
            },
            {
                "role": "user",
                "content": f"Simplify this sentence:\n\n{sentence_input.value}",
            },
        ],
        vertex_project="chatbot-183407",
        vertex_location="us-central1",
        temperature=temperature_slider.value,
        max_tokens=1024,
        max_retries=3,
    )
    return (instructor_result,)


@app.cell(hide_code=True)
def _(instructor_result, mo):
    mo.md(f"""
    ### Structured Output with Instructor

    {mo.json(instructor_result.model_dump_json())}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Benefits of Using Instructor

    **Type Safety:**
    ```python
    # This is a proper Python object, not a dict!
    result.simplified_text  # Type-checked
    result.complexity_score_before  # Guaranteed to be int between 1-10
    result.changes_made  # Guaranteed to be a list
    ```

    **Automatic Validation:**
    - If the model outputs invalid JSON → Automatic retry
    - If fields are missing → Automatic retry
    - If types are wrong → Automatic retry
    - If validation rules fail (like `ge=1, le=10`) → Automatic retry

    **Better Error Messages:**
    - Pydantic provides clear error messages about what's wrong
    - You can see exactly which field failed validation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Summary: Prompt Engineering Best Practices

    ## 1. Be Clear and Direct
    - State exactly what you want
    - Be specific about output format
    - Use concrete examples, not vague descriptions
    - Avoid ambiguous language

    ## 2. Assign Roles (System/User/Assistant)
    - Use system messages to define behavior
    - Set clear expectations and constraints
    - Specify output requirements upfront

    ## 3. Separate Data from Instructions
    - Use XML tags to mark boundaries: `<instructions>`, `<data>`
    - Prevent confusion between content and commands
    - Make prompts unambiguous and parseable

    ## 4. Control Output Formatting
    - Specify desired output structure in instructions
    - Use prefilling to guarantee response start
    - Common prefills: `{` for JSON, `<result>` for XML

    ## 5. Few-Shot Learning
    - Provide 1-5 representative examples
    - Use consistent format across examples
    - Show exact output style you want
    - Include edge cases if relevant

    ## 6. Chain of Thought
    - Use for complex reasoning tasks
    - Zero-shot: "Let's think step by step"
    - Standard: Provide reasoning examples
    - Structured: Request specific format (e.g., `<sketchpad>`)

    ## 7. Structured Outputs
    - Native JSON mode for simple structured data
    - Instructor for type-safe, validated outputs
    - Define clear Pydantic schemas
    - Let Instructor handle retries and validation

    ## General Tips

    **Do:**
    - Iterate and test your prompts
    - Be specific about what you want
    - Use XML tags to separate concerns
    - Validate outputs programmatically
    - Use prefilling for consistent formats
    """)
    return


if __name__ == "__main__":
    app.run()
