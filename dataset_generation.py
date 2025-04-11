"""
Common utilities for processing cleaned message data across different formats.
"""

import csv
from datetime import datetime
import logging
import os
from typing import Any, Dict
import pandas as pd
import pytz

from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# init
langfuse = Langfuse()

# Configure logger
logger = logging.getLogger(__name__)

# Add constant with environment variable fallback
NUM_WORKERS = int(os.getenv("MESSAGE_PROCESSOR_WORKERS", 10))


def format_time(seconds: float) -> str:
    """
    Format a time duration given in seconds into a human-readable string.

    Args:
        seconds (float): Time duration in seconds.

    Returns:
        str: Formatted time string in the format "HH:MM:SS".
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def calculate_time_difference(
    prev_timestamp: str | None, current_timestamp: str | None
) -> float | None:
    """
    Calculate the time difference between two ISO format timestamps in seconds.

    Args:
        prev_timestamp (str | None): The previous timestamp in ISO format
        current_timestamp (str | None): The current timestamp in ISO format

    Returns:
        float | None: The time difference in seconds, or None if either timestamp is None
    """
    if prev_timestamp is None or current_timestamp is None:
        return None

    prev_time = datetime.fromisoformat(prev_timestamp)
    current_time = datetime.fromisoformat(current_timestamp)
    return (current_time - prev_time).total_seconds()


def time_difference_to_str(total_seconds: float | None) -> str | None:
    """
    Convert a time difference in seconds to a human-readable string.

    Args:
        total_seconds (float | None): The time difference in seconds

    Returns:
        str | None: A human-readable string describing the time difference, or None if total_seconds is None
    """
    if total_seconds is None:
        return None

    days = total_seconds // (24 * 3600)

    if days > 0:
        return f"{int(days)} days ago."
    elif total_seconds > 3600:
        hours = int(total_seconds // 3600)
        return f"{hours} hours ago."
    elif total_seconds > 60:
        minutes = int(total_seconds // 60)
        return f"{minutes} minutes ago."
    else:
        return "Very recently."


def process_message_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the context dictionary to calculate time_diff_info, response_time,
    last_message_from_other_partner, and build the context string.
    """
    current_message = context["message"]
    previous_messages = context["previous_messages"]
    all_messages = previous_messages + [current_message] + context["next_messages"]

    sender_id = current_message["person_id"]
    last_message_from_other_partner = None
    other_partner_timestamp = None
    time_diff_info = None
    response_time = None

    # Find the last message from the other partner
    for i, msg in enumerate(reversed(previous_messages)):
        if msg["person_id"] != sender_id:
            last_message_from_other_partner = msg["message"]
            # Only calculate time_diff if this is the most recent message (i == 0)
            if i == 0:
                other_partner_timestamp = msg["timestamp"]
                response_time = calculate_time_difference(
                    other_partner_timestamp, current_message["timestamp"]
                )
                time_diff_info = time_difference_to_str(response_time)
            break

    # Build the context string
    context_str = "; ".join(
        [
            f"{msg['timestamp'].strip()} - Person-{msg['person_id'].strip()}: {msg['message'].strip()}"
            for msg in all_messages
        ]
    )

    return {
        "time_diff_info": time_diff_info,
        "response_time": response_time,
        "last_message_from_other_partner": last_message_from_other_partner,
        "context_str": context_str,
    }


def load_csv_in_chunks(file_path, chunk_size=100):
    """
    Generator function to load CSV files from a list of file paths in chunks of a given size.

    Args:
        csv_files (list): List of paths to CSV files.
        chunk_size (int): Number of rows to load at a time.

    Yields:
        list: A list of dictionaries, where each dictionary represents a row in the CSV.
    """
    if not os.path.isfile(file_path):
        logger.error(f"The file {file_path} does not exist.")
        exit()

    with open(file_path, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        # Check if the first row is the header
        first_row = next(reader)
        if first_row == ["person_id", "timestamp", "message"]:
            # If it's the header, skip it
            fieldnames = first_row
        else:
            # If it's not the header, use it as data
            fieldnames = ["person_id", "timestamp", "message"]
            file.seek(0)  # Reset file pointer to the beginning

        dict_reader = csv.DictReader(file, fieldnames=fieldnames)
        chunk = []
        for row in dict_reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        # Yield remaining rows if any
        if chunk:
            yield chunk


def get_context(
    messages: list, idx: int, prev_context_size: int, next_context_size: int
) -> dict:
    """
    Get the context for a given message by taking the previous messages, current message, and next messages.

    Args:
        messages (list): List of message dictionaries.
        idx (int): Index of the current message in the list.
        prev_context_size (int): Number of previous messages to include in the context.
        next_context_size (int): Number of next messages to include in the context.

    Returns:
        dict: A dictionary containing lists of previous messages, the current message, and next messages.
    """
    # Get the previous messages
    prev_messages = messages[max(0, idx - prev_context_size) : idx]

    # Get the current message
    current_message = messages[idx]

    # Get the next messages, ensuring we don't go beyond the end of the list
    next_messages = messages[idx + 1 : min(idx + 1 + next_context_size, len(messages))]

    # Create the context dictionary
    context = {
        "previous_messages": prev_messages,
        "message": current_message,
        "next_messages": next_messages,
    }

    return context


def preprocess_conversation(
    messages: list,
    previous_messages: list,
    prev_context_size: int,
    next_context_size: int,
) -> list:
    """
    Preprocess a conversation by combining previous messages with current messages and adding context-dependent variables.

    Args:
        messages (list): List of current message dictionaries.
        previous_messages (list): List of previous message dictionaries.
        prev_context_size (int): Number of previous messages to include in the context.
        next_context_size (int): Number of next messages to include in the context.

    Returns:
        list: List of fully processed message dictionaries.
    """
    # Start by combining previous messages with current messages
    messages = previous_messages + messages
    starting_idx = len(previous_messages)

    # List to store fully processed rows
    processed_rows = []

    # Loop from starting_idx to len(messages):
    for i in range(starting_idx, len(messages)):
        # Handle context: Take up to prev_context_size previous messages, current message, and next_context_size next messages
        context = get_context(
            messages=messages,
            idx=i,
            prev_context_size=prev_context_size,
            next_context_size=next_context_size,
        )

        # Combine message data and context-dependent variables into one dictionary
        processed_row = {
            "person_id": messages[i]["person_id"].strip(),
            "message": messages[i]["message"].strip(),
            "timestamp": messages[i]["timestamp"].strip(),
            "context": context,  # Add the combined context string
        }
        # Add the processed row to the list
        processed_rows.append(processed_row)
    return processed_rows


def enrich_and_load_messages(
    file_path: str,
    result_dict: dict = None,
):
    """
    Enrich and load message data from CSV files into the database.

    Args:
        file_path (str): Path to cleaned CSV file.
        messages_per_chunk (int): Number of messages to process in each chunk.
        prev_context_size (int): Number of previous messages to include in the context.
        next_context_size (int): Number of next messages to include in the context.

    Raises:
        ValueError: If prev_context_size is greater than messages_per_chunk.
    """
    previous_messages = []  # Track the last context_size messages
    total_messages = 0

    result = {}
    for chunk in load_csv_in_chunks(file_path, chunk_size=100):
        processed_batch = preprocess_conversation(
            messages=chunk,
            previous_messages=previous_messages,
            prev_context_size=10,
            next_context_size=1,
        )

        total_messages += len(processed_batch)
        previous_messages = chunk[-10:]

        for row in processed_batch:
            processed_message = process_message_context(row["context"])
            processed_message.update({"message": row["message"]})
            result[row["timestamp"]] = {
                "input": {
                    "context": processed_message["context_str"],
                    "message": processed_message["message"],
                },
                "expected_output": result_dict[row["timestamp"]],
            }
    return result


def convert_mock_data_to_dict(file_path: str) -> dict:
    """
    Convert mock data CSV to a dictionary where keys are datetime objects and values are lists of SPAFF codes.

    Args:
        file_path (str): Path to the mock data CSV file

    Returns:
        dict: Dictionary with datetime keys and SPAFF code lists as values
    """
    PACIFIC_TZ = pytz.timezone("US/Pacific")
    UTC_TZ = pytz.UTC
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Initialize the result dictionary
    result_dict = {}

    # Process each row
    for _, row in df.iterrows():
        # Convert Message Date to datetime
        local_dt = PACIFIC_TZ.localize(
            datetime.strptime(row["Message Date"], "%Y-%m-%d %H:%M:%S")
        )
        date_key = local_dt.astimezone(UTC_TZ).isoformat()

        # Get SPAFF values and filter out empty ones
        spaff_values = []
        for col in ["SPAFF - 1", "SPAFF - 2", "SPAFF - 3"]:
            if pd.notna(row[col]) and row[col].strip():
                spaff_values.append(row[col].strip())

        # Store in dictionary
        result_dict[date_key] = spaff_values

    return result_dict


if __name__ == "__main__":
    result_dict = convert_mock_data_to_dict(
        "/home/ubuntu/Projects/barprompt/mock_data.csv"
    )
    result = enrich_and_load_messages(
        file_path="/home/ubuntu/Projects/barprompt/20250409_185158_mock_data_clean.csv",
        result_dict=result_dict,
    )
    for key, item in result.items():
        langfuse.create_dataset_item(
            dataset_name="check_performance",
            # any python object or value
            input=item["input"],
            # any python object or value, optional
            expected_output=item["expected_output"],
        )
