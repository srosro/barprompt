# BarPrompt

BarPrompt is a tool that integrates with Langfuse (prompt management) to help you test and compare different prompts effectively. It provides a streamlined workflow for evaluating prompts against datasets and tracking the results in Langfuse.

## Features

- Seamless integration with Langfuse for prompt management and tracking
- Concurrent evaluation of prompts against datasets
- Detailed scoring and analysis of prompt performance
- Multiple evaluation methods including exact matching, list inclusion, and LLM-based evaluation
- Flexible configuration via YAML for customizing evaluation strategies
- Conditional scoring based on expected output values

## Prerequisites

- Python 3.12 or higher
- Poetry (Python package manager)
- OpenAI API key
- Langfuse account and credentials

## Poetry Setup

Before proceeding with the installation, you need to set up pipx and Poetry:

### Install pipx

**macOS:**
```bash
brew install pipx
pipx ensurepath
```

**Linux:**
```bash
# Debian/Ubuntu
sudo apt update
sudo apt install pipx
pipx ensurepath

# Fedora
sudo dnf install pipx
pipx ensurepath
```

### Install Poetry using pipx

```bash
pipx install poetry
```

### Install required Poetry plugins

```bash
pipx inject poetry poetry-plugin-shell
pipx inject poetry poetry-plugin-dotenv
```


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/barprompt.git
cd barprompt
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Set Up Environment Variables

Copy the `.env.template` file to `.env` and fill in your credentials:

```bash
cp .env.template .env
```

Edit the `.env` file with your:
- OpenAI API key
- Langfuse credentials (public key, secret key, host)

## Usage

### Running Prompt Evaluations

1. Activate the Poetry environment:
```bash
poetry shell
```

2. Run the main script:
```bash
barprompt
```

3. Follow the interactive prompts to:
   - Enter the prompt name
   - Specify the prompt version
   - Select the dataset to evaluate against

The script will run the evaluation and provide results through the Langfuse dashboard.

### Configuring Evaluations

BarPrompt uses a YAML configuration file (`evaluation_config.yaml`) to define how outputs should be evaluated. If this file is not found, a default evaluation will be used.

#### Example Configuration

```yaml
evaluations:
  # Simple exact match comparison
  - name: exact_match_affect
    function: simple_exact_comparison
    key: affect
  
  # List inclusion comparison
  - name: negative_affects
    function: list_inclusion_comparison
    key: affect
    score_only_for:
      - "Anger"
      - "Sadness"
      - "Disgust"
  
  # LLM-based evaluation
  - name: llm_quality_score
    function: llm_judge_evaluation
    key: affect
    args:
      judge_prompt_name: quality_judge
      judge_prompt_version: 5
```

#### Configuration Options

Each evaluation item can include:

- `name`: The name of the score in Langfuse
- `function`: The evaluation function to use
  - `simple_exact_comparison`: Checks if output exactly matches expected output
  - `list_inclusion_comparison`: Checks if output is in a list of expected outputs
  - `llm_judge_evaluation`: Uses an LLM to evaluate the quality (returns 0-1 score)
- `key`: (Optional) Field to extract from JSON output for evaluation
- `score_only_for`: (Optional) List of expected output values to score (others will be skipped)
- `args`: Additional arguments for specific evaluation functions
  - For `llm_judge_evaluation`:
    - `judge_prompt_name`: Name of the prompt in Langfuse to use for evaluation
    - `judge_prompt_version`: Version of the prompt to use

## Evaluation Methods

### Simple Exact Comparison

Compares the output with the expected output for an exact match. Returns a binary score (1 for match, 0 for no match).

### List Inclusion Comparison

Checks if the output is included in a list of expected outputs. Returns a binary score (1 if included, 0 if not).

### LLM Judge Evaluation

Uses an LLM as a judge to evaluate the quality of the output compared to the expected output. Returns a continuous score between 0 and 1.

## Project Structure

- `barprompt.py`: Main script for running prompt evaluations
- `evaluation_config.yaml`: Configuration for evaluation methods
- `pyproject.toml`: Poetry project configuration
- `.env`: Environment variables (not tracked in git)
- `.env.template`: Template for environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the terms specified in the LICENSE file.
