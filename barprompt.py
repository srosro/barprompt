import os
import json
import concurrent.futures
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
        "# The app that integrates langfuse (prompt management) into promptfoo (prompt evaluation)"
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
    experiment_name = f"{prompt_name}_v{prompt_version}_test"
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


def process_item(item, prompt, experiment):
    with item.observe(run_name=experiment) as trace_id:
        output = run_prompt_evaluation(item.input, prompt)
        output_dict = json.loads(output)
        affect = output_dict.get("affect", None)
        print(affect)
        if affect.startswith("Other-"):
            affect = "Neutral"
        elif affect.startswith("Partner-"):
            affect = affect.replace("Partner-", "", 1)
        is_correct = affect in item.expected_output
        langfuse.score(
            trace_id=trace_id,
            name="exact_match",
            value=is_correct,
        )
        langfuse.score(
            trace_id=trace_id,
            name=affect.lower(),
            value=is_correct,
        )


def run_experiment(prompt, dataset, experiment):
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(process_item, item, prompt, experiment)
            for item in dataset.items
        ]
        # Wait for all futures to complete
        concurrent.futures.wait(futures)


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
