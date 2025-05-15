"""BARPROMPT - A tool for evaluating prompts with Langfuse."""

import concurrent.futures
import json
import os
import sys

from pathlib import Path
from typing import Any

import yaml

from langfuse import Langfuse
from langfuse.api.resources.commons.errors.not_found_error import NotFoundError
from langfuse.client import DatasetClient, DatasetItemClient, PromptClient
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai

langfuse = Langfuse()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))


def print_welcome() -> None:
    """Display welcome message for the application."""
    # Welcome message is printed to console intentionally as part of UI
    print("#####################################")
    print("# Welcome to BARPROMPT")
    print(
        "# The app that integrates with Langfuse (prompt management) to help you test and compare "
        "different prompts effectively.",
    )
    print("#####################################")
    print()


def get_user_input(prompt_text: str) -> str:
    """Get input from the user with a formatted prompt.

    Args:
        prompt_text: The text to display as prompt

    Returns:
        The user input as a string
    """
    return input(f"> {prompt_text}\n$ ").strip()


def validate_llm_judges(eval_config: dict[str, list[dict[str, Any]]]) -> tuple[str, str] | None:
    """Validate LLM judges in evaluation config.

    Args:
        eval_config: The evaluation configuration

    Returns:
        Tuple of (judge_name, judge_version) if exactly one judge is found, None otherwise
    """
    llm_judges = []
    for eval_item in eval_config.get("evaluations", []):
        if eval_item.get("function") == "llm_judge_evaluation":
            judge_prompt_name = eval_item.get("args", {}).get("judge_prompt_name")
            if judge_prompt_name is None:
                missing_judge_name = "judge_prompt_name is required in evaluation config"
                raise ValueError(missing_judge_name)
            judge_prompt_version = eval_item.get("args", {}).get("judge_prompt_version")
            if judge_prompt_version is None:
                missing_judge_version = "judge_prompt_version is required in evaluation config"
                raise ValueError(missing_judge_version)
            llm_judges.append((judge_prompt_name, judge_prompt_version))

    if len(llm_judges) > 1:
        error_msg = (
            "Error: Multiple LLM judges found in evaluation config. Currently, only one LLM judge "
            "can be used per experiment run because the judge name and version are included in the "
            "experiment name to prevent duplicate runs. Found judges:\n"
        )
        for name, version in llm_judges:
            error_msg += f"- {name} (v{version})\n"
        raise ValueError(error_msg)

    return llm_judges[0] if llm_judges else None


def validate_against_langfuse(
    prompt_name: str,
    prompt_version: str,
    dataset_name: str,
) -> tuple[PromptClient, DatasetClient, str]:
    """Validate prompt and dataset existence in Langfuse.

    Args:
        prompt_name: Name of the prompt to evaluate
        prompt_version: Version of the prompt
        dataset_name: Name of the dataset to use

    Returns:
        Tuple containing prompt, dataset and experiment name
    """
    try:
        prompt = langfuse.get_prompt(prompt_name, version=prompt_version)
    except NotFoundError:
        sys.exit()
    try:
        dataset = langfuse.get_dataset(dataset_name)
    except NotFoundError as e:
        print(f"Error while fetching dataset '{dataset_name}': {e}")
        sys.exit()

    # Load evaluation configuration to check for LLM judges
    eval_config = load_evaluation_config()

    # Validate LLM judges and get judge info if present
    judge_info = ""
    judge_result = validate_llm_judges(eval_config)
    if judge_result:
        judge_name, judge_version = judge_result
        judge_info = f"_judge_{judge_name}_v{judge_version}"

    experiment_name = f"{prompt_name}_v{prompt_version}{judge_info}"
    try:
        langfuse.get_dataset_run(dataset_name, experiment_name)
        print("Experiment already exists. Exiting.")
        sys.exit()
    except NotFoundError:
        pass
    return prompt, dataset, experiment_name


@observe()
def run_prompt_evaluation(input_data: str | dict[str, Any], prompt: PromptClient) -> str | None:
    """Run the evaluation of a prompt with input data.

    This implementation compiles the Langfuse prompt with the variables coming
    from the dataset item (``input_data``). The compiled prompt is then passed
    as the **user** message to the model - mirroring the deployment pattern
    used in the *codel-text* project.

    In addition, the OpenAI generation parameters (model name and temperature)
    are automatically picked up from the prompt's ``config`` field (if
    provided). Otherwise, the process falls back to the environment defaults
    configured at startup.

    Args:
        input_data: The input data (variables) for the prompt. Must be a
            ``dict`` when the prompt expects template variables.
        prompt: The Langfuse ``PromptClient`` instance to evaluate.

    Returns:
        The generated output from the model.
    """
    # 1. Resolve model parameters: prefer prompt.config > env defaults
    try:
        model_config: dict[str, Any] = getattr(prompt, "config", {})
    except AttributeError:  # Just in case the SDK changes
        model_config = {}

    # 2. Compile the prompt with the provided variables (if any)
    compiled_prompt: str
    if isinstance(input_data, dict):
        try:
            compiled_prompt = prompt.compile(**input_data)
        except TypeError:
            # If the prompt does not accept the given variables, fall back to a
            # simple compile (this also supports static prompts without
            # variables).
            compiled_prompt = prompt.compile()
    else:
        compiled_prompt = prompt.compile()

    # 3. Build the request parameters for the OpenAI completion
    request_params = {
        "model": model_config.get("model", "gpt-4o"),
        "messages": [
            {"role": "system", "content": model_config.get("system_message", "")},
            {"role": "user", "content": compiled_prompt},
        ],
        "max_tokens": model_config.get("max_tokens", 1_000),
        "temperature": model_config.get("temperature", 0.7),
        "response_format": model_config.get("response_format", {"type": "json_object"}),
        "timeout": model_config.get("timeout", model_config.get("request_timeout", 60)),
    }

    # 4. Execute the OpenAI completion
    completion: str | None = (
        openai.chat.completions.create(
            langfuse_prompt=prompt,
            **request_params,
        )
        .choices[0]
        .message.content
    )
    return completion


def load_evaluation_config(config_file: str = "evaluation_config.yaml") -> dict[str, list[dict[str, Any]]]:
    """Load evaluation configuration from YAML file.

    Args:
        config_file: Path to the YAML configuration file

    Returns:
        The evaluation configuration
    """
    try:
        with Path(config_file).open() as file:
            config = yaml.safe_load(file)
            if config is None:
                return {"evaluations": []}
            # Ensure proper typing for return value
            return config if isinstance(config, dict) else {"evaluations": []}
    except FileNotFoundError:
        print(f"Warning: Evaluation config file '{config_file}' not found. Using default evaluation.")
        # Return a default configuration
        return {
            "evaluations": [
                {
                    "name": "exact_match",
                    "function": "simple_exact_comparison",
                    "key": "affect",
                },
            ],
        }
    except (OSError, yaml.YAMLError, PermissionError) as e:
        print(f"Error loading evaluation config: {e}")
        return {"evaluations": []}


def extract_expected_value(
    expected_output: str | dict[str, Any] | list[Any],
    eval_key: str | None,
) -> str | dict[str, Any] | list[Any]:
    """Extract the expected value from the expected output.

    Args:
        expected_output: The expected output
        eval_key: The key to extract from the expected output

    Returns:
        The extracted value from the expected output
    """
    expected_value = expected_output
    if eval_key is not None and isinstance(expected_value, str):
        try:
            expected_json = json.loads(expected_value)
            if isinstance(expected_json, dict) and eval_key in expected_json:
                expected_value = expected_json[eval_key]
        except json.JSONDecodeError:
            pass
    elif eval_key is not None and isinstance(expected_value, dict) and eval_key in expected_value:
        expected_value = expected_value[eval_key]
    return expected_value


def process_item(item: DatasetItemClient, prompt: PromptClient, experiment: str) -> None:
    """Process a single dataset item with the prompt.

    Args:
        item: The dataset item to process
        prompt: The prompt to evaluate
        experiment: The name of the experiment
    """
    with item.observe(run_name=experiment) as trace_id:
        output = run_prompt_evaluation(item.input, prompt)

        # Load evaluation configuration
        eval_config = load_evaluation_config()

        # Apply the configured evaluations
        for eval_item in eval_config.get("evaluations", []):
            eval_name = eval_item.get("name", "unnamed_evaluation")
            eval_function = eval_item.get("function", "simple_exact_comparison")
            eval_key = eval_item.get("key", None)
            score_only_for = eval_item.get("score_only_for", None)
            filter_by = eval_item.get("filter_by", "output")  # 'output' or 'expected'

            # Determine if the score should be recorded based on the score_only_for parameter
            should_score = True
            if score_only_for is not None:
                if filter_by == "expected":
                    # Filter based on expected output
                    expected_value = extract_expected_value(item.expected_output, eval_key)
                    # Apply different filtering logic based on the evaluation function
                    if eval_function == "simple_exact_comparison" or not isinstance(expected_value, list):
                        should_score = expected_value in score_only_for
                    else:
                        should_score = any(item in score_only_for for item in expected_value)
                else:
                    # Filter based on actual output (default)
                    output_value = extract_output_value(output, eval_key)
                    should_score = output_value in score_only_for

            # Record the score in Langfuse if it should be scored
            if should_score:
                # Determine which evaluation function to use
                if eval_function == "llm_judge_evaluation":
                    judge_prompt_name = eval_item.get("args", {}).get("judge_prompt_name", "default_judge")
                    judge_prompt_version = eval_item.get("args", {}).get("judge_prompt_version", 1)

                    eval_score = llm_judge_evaluation(
                        output,
                        item,
                        judge_prompt_name,
                        judge_prompt_version,
                        eval_key,
                    )
                elif eval_function == "list_inclusion_comparison":
                    eval_score = list_inclusion_comparison(output, item.expected_output, eval_key)
                else:  # Default to simple_exact_comparison
                    eval_score = simple_exact_comparison(output, item.expected_output, eval_key)
                langfuse.score(
                    trace_id=trace_id,
                    name=eval_name,
                    value=eval_score,
                )


def run_experiment(prompt: PromptClient, dataset: DatasetClient, experiment: str) -> None:
    """Run experiment with prompt and dataset.

    Args:
        prompt: The prompt to evaluate
        dataset: The dataset to use
        experiment: The name of the experiment
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_item, item, prompt, experiment) for item in dataset.items]
        # Wait for all futures to complete
        concurrent.futures.wait(futures)


def extract_output_value(output: str | None, key: str | None = None) -> str | dict[str, Any] | None:
    """Extract the value from the output.

    Args:
        output: The output to extract the value from
        key: The key to extract from the output
    """
    output_value = output
    if isinstance(output, str):
        try:
            output_json = json.loads(output)
            if isinstance(output_json, dict) and key in output_json:
                output_value = output_json[key]
        except json.JSONDecodeError:
            # Not valid JSON, use as is
            pass
    return output_value


@observe()
def llm_judge_evaluation(
    output: str | None,
    item: DatasetItemClient,
    judge_prompt_name: str,
    judge_prompt_version: int = 1,
    key: str | None = None,
) -> float:
    """Use an LLM to evaluate the quality of the output compared to expected output.

    This version mirrors `run_prompt_evaluation` and therefore:
      1. Compiles the judge prompt with the evaluation input variables.
      2. Extracts model parameters from `prompt.config` (fallback to env defaults).
      3. Executes the OpenAI call with those dynamic parameters.

    Args:
        output: The actual output from the LLM
        item: The dataset item to evaluate
        judge_prompt_name: Name of the prompt in Langfuse to use for evaluation
        judge_prompt_version: Version of the prompt to use
        key: Optional key to extract from output and expected_output if they are JSON objects

    Returns:
        A score between 0-1 indicating the quality of the match
    """
    # Extract values if key is provided and values are valid JSON
    output_value: str | dict[str, Any] | None = output
    expected_value = item.expected_output

    if key is not None:
        # Try to extract the key from output if it's JSON
        output_value = extract_output_value(output, key)
        # Try to extract the key from expected_output if it's JSON
        expected_value = extract_expected_value(item.expected_output, key)

    try:
        judge_prompt = langfuse.get_prompt(judge_prompt_name, version=judge_prompt_version)
    except NotFoundError as e:
        print(f"Error: Judge prompt '{judge_prompt_name}' not found: {e}")
        return 0.0

    # ---------------------------------------------
    # Build evaluation input and compile prompt
    # ---------------------------------------------
    evaluation_input = {
        "output": output_value,
        "expected_output": expected_value,
        "input": item.input,
    }

    # 1. Resolve model parameters from prompt config (fallback to env defaults)
    try:
        model_config: dict[str, Any] = getattr(judge_prompt, "config", {})
    except AttributeError:
        model_config = {}

    # 2. Compile the judge prompt with variables (if accepted)
    try:
        compiled_prompt = judge_prompt.compile(**evaluation_input)
    except TypeError:
        compiled_prompt = judge_prompt.compile()

    # 3. Construct request params similarly to `run_prompt_evaluation`
    request_params = {
        "model": model_config.get("model", "gpt-4o"),
        "messages": [
            {"role": "system", "content": model_config.get("system_message", "")},
            {"role": "user", "content": compiled_prompt},
        ],
        "max_tokens": model_config.get("max_tokens", 1000),
        "temperature": model_config.get("temperature", 0.0),  # judges default to deterministic
        "response_format": model_config.get("response_format", {"type": "json_object"}),
        "timeout": model_config.get("timeout", model_config.get("request_timeout", 60)),
    }

    # ---------------------------------------------
    # Execute OpenAI call
    # ---------------------------------------------
    try:
        completion = (
            openai.chat.completions.create(
                langfuse_prompt=judge_prompt,
                **request_params,
            )
            .choices[0]
            .message.content
        )

        # Parse the score from the completion (expects JSON with "score")
        try:
            result = json.loads(completion)
            score = float(result.get("score", 0.0))
            return max(0.0, min(1.0, score))  # Clamp to [0,1]
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing judge output: {e}")
            print(f"Raw output: {completion}")
            return 0.0

    except (openai.OpenAIError, ValueError, KeyError) as e:
        print(f"Error during evaluation: {e}")
        return 0.0


def simple_exact_comparison(
    output: str | None,
    expected_output: str | dict[str, Any],
    key: str | None = None,
) -> int:
    """Perform a simple exact comparison between output and expected_output.

    If key is provided, extracts that key from JSON objects before comparison.

    Args:
        output: The actual output to evaluate
        expected_output: The expected output to compare against
        key: Optional key to extract from output and expected_output if they are JSON objects

    Returns:
        1 if the output matches expected_output, 0 otherwise
    """
    # Extract values if key is provided and values are valid JSON
    output_value: str | dict[str, Any] | None = output
    expected_value: str | dict[str, Any] | list[Any] = expected_output

    if key is not None:
        output_value = extract_output_value(output, key)
        expected_value = extract_expected_value(expected_output, key)

    return 1 if output_value == expected_value else 0


def list_inclusion_comparison(
    output: str | None,
    expected_output: str | dict[str, Any] | list[Any],
    key: str | None = None,
) -> int:
    """Check if the output is included in the expected_output list.

    If key is provided, extracts that key from JSON objects before comparison.

    Args:
        output: The actual output to evaluate
        expected_output: A list of acceptable outputs
        key: Optional key to extract from output and expected_output if they are JSON objects

    Returns:
        1 if the output is in the expected_output list, 0 otherwise
    """
    # Extract values if key is provided and values are valid JSON
    output_value: str | dict[str, Any] | None = output

    if key is not None:
        output_value = extract_output_value(output, key)

    # Ensure expected_output is a list
    if not isinstance(expected_output, list):
        expected_output = [expected_output]

    return 1 if output_value in expected_output else 0


def main() -> None:
    """Execute the main application flow."""
    # Load environment variables
    print_welcome()

    # Get user inputs
    prompt_name = get_user_input("Please input the name of the prompt:")
    prompt_version = get_user_input("Great! Please input the version:")
    dataset_name = get_user_input("Great! Please input the name of the dataset:")

    prompt, dataset, experiment = validate_against_langfuse(prompt_name, prompt_version, dataset_name)

    print(f"\nGreat, we'll evaluate {prompt_name} (v{prompt_version}) with dataset {dataset_name},")
    print(f"which will be {len(dataset.items)} LLM queries.")

    confirm = get_user_input("Continue (y/n): ").lower()
    if confirm != "y":
        print("Experiment cancelled.")
        sys.exit()

    # Run the experiment
    run_experiment(prompt, dataset, experiment)

    # Ensure all events are sent to Langfuse
    langfuse_context.flush()
    langfuse.flush()

    print("\nExperiment complete. You can find this experiment in your Langfuse dashboard.")
    print(f"{os.getenv('LANGFUSE_HOST')}/project/{dataset.project_id}/datasets/{dataset.id}")


if __name__ == "__main__":
    main()
