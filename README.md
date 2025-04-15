# BarPrompt

BarPrompt is a tool that integrates with Langfuse (prompt management) to help you test and compare different prompts effectively. It provides a streamlined workflow for evaluating prompts against datasets and tracking the results in Langfuse.

## Features

- Seamless integration with Langfuse for prompt management and tracking
- Concurrent evaluation of prompts against datasets
- Detailed scoring and analysis of prompt performance

## Prerequisites

- Python 3.12 or higher
- Poetry (Python package manager)
- OpenAI API key
- Langfuse account and credentials

## Installation

### 1. Install Poetry

If you don't have Poetry installed, you can install it using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/barprompt.git
cd barprompt
```

### 3. Install Dependencies

```bash
poetry install
```

### 4. Set Up Environment Variables

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
python barprompt.py
```

3. Follow the interactive prompts to:
   - Enter the prompt name
   - Specify the prompt version
   - Select the dataset to evaluate against

The script will run the evaluation and provide results through the Langfuse dashboard.

## Project Structure

- `barprompt.py`: Main script for running prompt evaluations
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
