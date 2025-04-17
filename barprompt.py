import os
import json
import concurrent.futures
import yaml
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import openai
from langfuse.decorators import observe, langfuse_context
from langfuse.api.resources.commons.errors.not_found_error import NotFoundError


load_dotenv()

langfuse = Langfuse()


def print_welcome():
    print("#####################################")
    print("# Welcome to BARPROMPT")
    print(
        "# The app that integrates with Langfuse (prompt management) to help you test and compare different prompts effectively."
    )
    print("#####################################")
    print()


def get_user_input(prompt):
    return input(f"> {prompt}\n$ ").strip()


def validate_against_langfuse(prompt_name, prompt_version, dataset_name):
    try:
        prompt = langfuse.get_prompt(prompt_name, version=prompt_version)
    except NotFoundError:
        exit()
    try:
        dataset = langfuse.get_dataset(dataset_name)
    except NotFoundError as e:
        print(f"Error while fetching dataset '{dataset_name}': {e}")
        exit()
    experiment_name = f"{prompt_name}_v{prompt_version}"
    try:
        langfuse.get_dataset_run(dataset_name, experiment_name)
        print("Experiment already exists. Exiting.")
        exit()
    except NotFoundError:
        pass
    return prompt, dataset, experiment_name


@observe()
def run_prompt_evaluation(input_data, prompt):
    messages = [
        {"role": "system", "content": prompt.compile()},
        {"role": "user", "content": str(input_data)},
    ]

    completion = (
        openai.chat.completions.create(
            model="gpt-4o", messages=messages, langfuse_prompt=prompt
        )
        .choices[0]
        .message.content
    )

    return completion


# def simple_evaluation(output, expected_output):
#     print(output)
#     affect = output.get("affect", None)
#     if not (affect and isinstance(affect, str)):
#         return 0
#     if affect.startswith("Other-"):
#         affect = "Neutral"
#     elif affect.startswith("Partner-"):
#         affect = affect.replace("Partner-", "", 1)
#     return affect in expected_output


def load_evaluation_config(config_file="evaluation_config.yaml"):
    """
    Load evaluation configuration from YAML file.

    Args:
        config_file: Path to the YAML configuration file

    Returns:
        dict: The evaluation configuration
    """
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(
            f"Warning: Evaluation config file '{config_file}' not found. Using default evaluation."
        )
        # Return a default configuration
        return {
            "evaluations": [
                {
                    "name": "exact_match",
                    "function": "simple_exact_comparison",
                    "key": "affect",
                }
            ]
        }
    except Exception as e:
        print(f"Error loading evaluation config: {e}")
        return {"evaluations": []}


def process_item(item, prompt, experiment):
    with item.observe(run_name=experiment) as trace_id:
        output = run_prompt_evaluation(item.input, prompt)
        # output_dict = json.loads(output)
        # affect = output_dict.get("affect", None)
        # print(affect)
        # if affect.startswith("Other-"):
        #     affect = "Neutral"
        # elif affect.startswith("Partner-"):
        #     affect = affect.replace("Partner-", "", 1)
        # is_correct = affect in item.expected_output

        # Load evaluation configuration
        eval_config = load_evaluation_config()

        # Apply the configured evaluations
        for eval_item in eval_config.get("evaluations", []):
            eval_name = eval_item.get("name", "unnamed_evaluation")
            eval_function = eval_item.get("function", "simple_exact_comparison")
            eval_key = eval_item.get("key", None)

            # Determine which evaluation function to use
            if eval_function == "llm_judge_evaluation":
                judge_prompt_name = eval_item.get("args", {}).get(
                    "judge_prompt_name", "default_judge"
                )
                judge_prompt_version = eval_item.get("args", {}).get(
                    "judge_prompt_version", "latest"
                )

                eval_score = llm_judge_evaluation(
                    output,
                    item.expected_output,
                    judge_prompt_name,
                    judge_prompt_version,
                    eval_key,
                )
            elif eval_function == "list_inclusion_comparison":
                eval_score = list_inclusion_comparison(
                    output, item.expected_output, eval_key
                )
            else:  # Default to simple_exact_comparison
                eval_score = simple_exact_comparison(
                    output, item.expected_output, eval_key
                )

            # Record the score in Langfuse
            langfuse.score(
                trace_id=trace_id,
                name=eval_name,
                value=eval_score,
            )

        # Keep the original scores for backward compatibility
        # langfuse.score(
        #     trace_id=trace_id,
        #     name="exact_match",
        #     value=is_correct,
        # )
        # langfuse.score(
        #     trace_id=trace_id,
        #     name=affect.lower(),
        #     value=is_correct,
        # )


def run_experiment(prompt, dataset, experiment):
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(process_item, item, prompt, experiment)
            for item in dataset.items
        ]
        # Wait for all futures to complete
        concurrent.futures.wait(futures)


@observe()
def llm_judge_evaluation(
    output, expected_output, judge_prompt_name, judge_prompt_version=1, key=None
):
    """
    Use an LLM to evaluate the quality of the output compared to expected output.

    Args:
        output: The actual output from the LLM
        expected_output: The expected output to compare against
        judge_prompt_name: Name of the prompt in langfuse to use for evaluation
        judge_prompt_version: Version of the prompt to use (default: 1)
        key: Optional key to extract from output and expected_output if they are JSON objects

    Returns:
        float: A score between 0-1 indicating the quality of the match
    """
    # Extract values if key is provided and values are valid JSON
    output_value = output
    expected_value = expected_output

    if key is not None:
        # Try to extract the key from output if it's JSON
        if isinstance(output, str):
            try:
                output_json = json.loads(output)
                if isinstance(output_json, dict) and key in output_json:
                    output_value = output_json[key]
            except json.JSONDecodeError:
                # Not valid JSON, use as is
                pass
        elif isinstance(output, dict) and key in output:
            output_value = output[key]

        # Try to extract the key from expected_output if it's JSON
        if isinstance(expected_output, str):
            try:
                expected_json = json.loads(expected_output)
                if isinstance(expected_json, dict) and key in expected_json:
                    expected_value = expected_json[key]
            except json.JSONDecodeError:
                # Not valid JSON, use as is
                pass
        elif isinstance(expected_output, dict) and key in expected_output:
            expected_value = expected_output[key]

    try:
        judge_prompt = langfuse.get_prompt(
            judge_prompt_name, version=judge_prompt_version
        )
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
            openai.chat.completions.create(
                model="gpt-4o", messages=messages, langfuse_prompt=judge_prompt
            )
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

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0.0


def simple_exact_comparison(output, expected_output, key=None):
    """
    Performs a simple exact comparison between output and expected_output.
    If key is provided, extracts that key from JSON objects before comparison.

    Args:
        output: The actual output to evaluate
        expected_output: The expected output to compare against
        key: Optional key to extract from output and expected_output if they are JSON objects

    Returns:
        int: 1 if the output matches expected_output, 0 otherwise
    """
    # Extract values if key is provided and values are valid JSON
    output_value = output
    expected_value = expected_output

    if key is not None:
        # Try to extract the key from output if it's JSON
        if isinstance(output, str):
            try:
                output_json = json.loads(output)
                if isinstance(output_json, dict) and key in output_json:
                    output_value = output_json[key]
            except json.JSONDecodeError:
                # Not valid JSON, use as is
                pass
        elif isinstance(output, dict) and key in output:
            output_value = output[key]

        # Try to extract the key from expected_output if it's JSON
        if isinstance(expected_output, str):
            try:
                expected_json = json.loads(expected_output)
                if isinstance(expected_json, dict) and key in expected_json:
                    expected_value = expected_json[key]
            except json.JSONDecodeError:
                # Not valid JSON, use as is
                pass
        elif isinstance(expected_output, dict) and key in expected_output:
            expected_value = expected_output[key]

    return 1 if output_value == expected_value else 0


def list_inclusion_comparison(output, expected_output, key=None):
    """
    Checks if the output is included in the expected_output list.
    If key is provided, extracts that key from JSON objects before comparison.

    Args:
        output: The actual output to evaluate
        expected_output: A list of acceptable outputs
        key: Optional key to extract from output and expected_output if they are JSON objects

    Returns:
        int: 1 if the output is in the expected_output list, 0 otherwise
    """
    # Extract values if key is provided and values are valid JSON
    output_value = output

    if key is not None:
        # Try to extract the key from output if it's JSON
        if isinstance(output, str):
            try:
                output_json = json.loads(output)
                if isinstance(output_json, dict) and key in output_json:
                    output_value = output_json[key]
            except json.JSONDecodeError:
                # Not valid JSON, use as is
                pass
        elif isinstance(output, dict) and key in output:
            output_value = output[key]

    # Ensure expected_output is a list
    if not isinstance(expected_output, list):
        expected_output = [expected_output]

    return 1 if output_value in expected_output else 0


def main():
    # Load environment variables
    print_welcome()

    # Get user inputs
    prompt_name = get_user_input("Please input the name of the prompt:")
    prompt_version = get_user_input("Great! Please input the version:")
    dataset_name = get_user_input("Great! Please input the name of the dataset:")

    prompt, dataset, experiment = validate_against_langfuse(
        prompt_name, prompt_version, dataset_name
    )

    print(
        f"\nGreat, we'll evaluate {prompt_name} (v{prompt_version}) with dataset {dataset_name}, which will be {len(dataset.items)} LLM queries."
    )

    confirm = get_user_input("Continue (y/n): ").lower()
    if confirm != "y":
        print("Experiment cancelled.")
        exit()

    # Run the experiment
    run_experiment(prompt, dataset, experiment)

    print(
        f"\nExperiment complete. You can find this experiment in your Langfuse dashboard. {os.getenv('LANGFUSE_HOST')}/project/{dataset.project_id}/datasets/{dataset.id}"
    )
    # Ensure all events are sent to Langfuse
    langfuse_context.flush()
    langfuse.flush()


if __name__ == "__main__":
    main()
