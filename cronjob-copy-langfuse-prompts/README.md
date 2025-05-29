# Langfuse Prompt Synchronization

This project provides tools to synchronize production prompts between Langfuse HIPAA and NON-PII environments. It includes scripts for copying prompts, verifying their content, and automated synchronization.

## Prerequisites

- Python 3.8 or higher
- Access to both Langfuse HIPAA and NON-PII environments
- Langfuse API credentials for both environments

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp template.env .env
```
Edit `.env` with your Langfuse credentials:
- HIPAA environment credentials
- NON-PII environment credentials

## Usage

### Manual Synchronization

Run the synchronization script:
```bash
./sync_prompts.sh
```

This will:
1. Copy production prompts from HIPAA to NON-PII
2. Verify the copied prompts
3. Log all operations to `prompt_sync.log`

### Automated Synchronization (Cronjob)

To set up automatic synchronization every 5 minutes:

1. Open the crontab editor:
```bash
crontab -e
```

2. Add the following line (adjust the path to match your setup):
```bash
*/5 * * * * cd /path/to/project && ./sync_prompts.sh
```

3. Make sure the script is executable:
```bash
chmod +x sync_prompts.sh
```

4. Verify the cronjob is set:
```bash
crontab -l
```

## Project Structure

- `copy_prompts.py`: Script to copy prompts from HIPAA to NON-PII
- `verify_prompts.py`: Script to verify prompt synchronization
- `sync_prompts.sh`: Bash script to run both Python scripts
- `template.env`: Template for environment variables
- `prompt_sync.log`: Log file for synchronization operations

## Log Management

The `prompt_sync.log` file:
- Contains timestamps for all operations
- Is automatically trimmed when it exceeds 5000 lines
- Keeps the most recent 4000 lines when trimmed

## Error Handling

The scripts include comprehensive error handling:
- Failed copies are logged but don't stop the process
- Verification failures are reported
- All errors are logged with timestamps
- Exit codes indicate success/failure

## Security Notes

- Never commit the `.env` file
- Keep `template.env` in version control
- Ensure proper permissions on the scripts
- Monitor the log file for any issues