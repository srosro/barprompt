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
    experiment_name = f"{prompt_name}_v{prompt_version}"
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

    Args:
        input_data: The input data for the prompt
        prompt: The prompt to evaluate

    Returns:
        The generated output from the model
    """
    messages = [
        {"role": "system", "content": prompt.compile()},
        {"role": "user", "content": str(input_data)},
    ]

    completion: str | None = (
        openai.chat.completions.create(model="gpt-4o", messages=messages, langfuse_prompt=prompt)
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

            # Determine if the score should be recorded based on the score_only_for parameter
            should_score = True
            if score_only_for is not None:
                # Extract expected output value for comparison
                expected_value = extract_expected_value(item.expected_output, eval_key)
                # Apply different filtering logic based on the evaluation function
                if eval_function == "simple_exact_comparison" or not isinstance(expected_value, list):
                    should_score = expected_value in score_only_for
                else:
                    should_score = any(item in score_only_for for item in expected_value)

            # Record the score in Langfuse if it should be scored
            if should_score:
                # Determine which evaluation function to use
                if eval_function == "llm_judge_evaluation":
                    judge_prompt_name = eval_item.get("args", {}).get("judge_prompt_name", "default_judge")
                    judge_prompt_version = eval_item.get("args", {}).get("judge_prompt_version", 1)

                    eval_score = llm_judge_evaluation(
                        output,
                        item.expected_output,
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
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
    expected_output: str | dict[str, Any] | list[Any],
    judge_prompt_name: str,
    judge_prompt_version: int = 1,
    key: str | None = None,
) -> float:
    """Use an LLM to evaluate the quality of the output compared to expected output.

    Args:
        output: The actual output from the LLM
        expected_output: The expected output to compare against
        judge_prompt_name: Name of the prompt in langfuse to use for evaluation
        judge_prompt_version: Version of the prompt to use
        key: Optional key to extract from output and expected_output if they are JSON objects

    Returns:
        A score between 0-1 indicating the quality of the match
    """
    # Extract values if key is provided and values are valid JSON
    output_value: str | dict[str, Any] | None = output
    expected_value = expected_output

    if key is not None:
        # Try to extract the key from output if it's JSON
        output_value = extract_output_value(output, key)
        # Try to extract the key from expected_output if it's JSON
        expected_value = extract_expected_value(expected_output, key)

    try:
        judge_prompt = langfuse.get_prompt(judge_prompt_name, version=judge_prompt_version)
    except NotFoundError as e:
        print(f"Error: Judge prompt '{judge_prompt_name}' not found: {e}")
        return 0.0

    # Prepare evaluation context
    evaluation_input = {"output": output_value, "expected_output": expected_value}

    # Call the LLM with the judge prompt
    messages = [
        {"role": "system", "content": judge_prompt.compile()},
        {"role": "user", "content": json.dumps(evaluation_input)},
    ]

    try:
        completion = (
            openai.chat.completions.create(model="gpt-4o", messages=messages, langfuse_prompt=judge_prompt)
            .choices[0]
            .message.content
        )

        # Parse the score from the completion
        # Expecting a JSON response with a "score" field (value 0-1)
        try:
            result = json.loads(completion)
            score = float(result.get("score", 0.0))
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
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

    print("\nExperiment complete. You can find this experiment in your Langfuse dashboard.")
    print(f"{os.getenv('LANGFUSE_HOST')}/project/{dataset.project_id}/datasets/{dataset.id}")

    # Ensure all events are sent to Langfuse
    langfuse_context.flush()
    langfuse.flush()


if __name__ == "__main__":
    main()
