# analyzer_utils.py
"""
General utility functions for the AI Response Analyzer application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import logging
import base64 # Needed for logo loading function
from pathlib import Path # Needed for logo path construction
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import specific config variables and constants
from analyzer_config import (
    LOGO_FILENAME, OVERALL_THRESHOLDS, METRIC_THRESHOLDS, SENTIMENT_THRESHOLDS,
    MAIN_TITLE_COLOR, BODY_TEXT_COLOR, SUBTITLE_COLOR, PRIMARY_ACCENT_COLOR,
    CHART_SUCCESS_COLOR, CHART_WARNING_COLOR, CHART_ERROR_COLOR # Use these for score colors
)

# --- Logging Configuration (Setup primarily in main app) ---
# logging.basicConfig(...)

# --- Batching Function ---
def batch_responses(responses, batch_size):
    """Split responses into batches."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        logging.warning(f"Invalid batch_size '{batch_size}', defaulting to 1.")
        batch_size = 1
    if not responses:
        return []
    return [responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]

# --- JSON Parsing (Specific for List Output) ---
def clean_and_parse_json_list(json_text):
    """
    Clean and parse JSON text expected to be a list of dictionaries.

    Handles markdown code blocks, leading 'json' text, and wraps single
    dictionaries in a list if needed. Filters out non-dictionary items
    within a list.

    Args:
        json_text (str): Raw text from the API potentially containing JSON.

    Returns:
        list: A list of dictionaries, or an empty list if parsing fails
              or no valid dictionary items are found.
    """
    if not json_text:
        logging.warning("Received empty JSON text to parse.")
        return []
    try:
        # Remove markdown code blocks and strip whitespace
        cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', json_text.strip(), flags=re.MULTILINE | re.DOTALL).strip()
        # Handle potential leading 'json' prefix
        if cleaned_text.lower().startswith('json'):
             cleaned_text = cleaned_text[4:].lstrip()

        # Attempt parsing
        parsed_json = json.loads(cleaned_text)

        # --- Type Handling ---
        if isinstance(parsed_json, list):
            # Ensure all items are dictionaries, filter if necessary
            valid_items = [item for item in parsed_json if isinstance(item, dict)]
            if len(valid_items) < len(parsed_json):
                non_dict_count = len(parsed_json) - len(valid_items)
                logging.warning(f"Parsed JSON list contained {non_dict_count} non-dictionary items. Filtering them out.")
                # Optionally inform user if partial data is returned
                # st.warning(f"Warning: Filtered out {non_dict_count} invalid items from the AI response list.")
            if not valid_items:
                 logging.warning("Parsed JSON list contained no valid dictionary items.")
                 st.error("AI response list did not contain any valid result objects.")
                 return []
            return valid_items
        elif isinstance(parsed_json, dict):
             # If API returns a single dict, wrap it in a list for consistency
             logging.warning("API returned a single dictionary, but expected a list. Wrapping it.")
             st.toast("Note: API returned a single object, processed as a list of one.", icon="ℹ️")
             return [parsed_json]
        else:
            # Handle unexpected types
            logging.error(f"Parsed JSON is not a list or dictionary: {type(parsed_json)}. Content snippet: {str(parsed_json)[:100]}")
            st.error(f"Unexpected JSON structure received (Type: {type(parsed_json)}). Expected a list of results.")
            return [] # Return empty list for failure

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response as JSON: {e}")
        # Limit problematic text display
        st.code(f"{cleaned_text[:500]}{'...' if len(cleaned_text)>500 else ''}", language='text')
        logging.error(f"JSONDecodeError: {e}. Problematic text snippet: {cleaned_text[:500]}...")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during JSON parsing: {e}")
        st.code(f"{cleaned_text[:500]}{'...' if len(cleaned_text)>500 else ''}", language='text')
        logging.error(f"Unexpected JSON parsing error: {e}. Problematic text snippet: {cleaned_text[:500]}...")
        return []


# --- Data Loading Function ---
# (Identical to the one used in the Themer app)
def load_data_from_file(uploaded_file):
    """
    Loads data from an uploaded CSV or Excel file into a pandas DataFrame.

    Handles basic cleaning (dropping empty rows/columns) and common
    CSV encoding issues (UTF-8, fallback to latin1). Reads all columns
    as strings initially to avoid type inference issues.

    Args:
        uploaded_file: The file object uploaded via st.file_uploader.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame or None if loading fails.
    """
    df = None
    if uploaded_file is None:
        return None
    try:
        file_name = uploaded_file.name
        logging.info(f"Attempting to load file: {file_name}")

        if file_name.endswith('.csv'):
            try: # Try UTF-8 first
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
                logging.info(f"Read CSV '{file_name}' with UTF-8.")
            except UnicodeDecodeError:
                logging.warning(f"UTF-8 failed for '{file_name}', trying latin1.")
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1', dtype=str, keep_default_na=False)
                    logging.info(f"Read CSV '{file_name}' with latin1.")
                except Exception as e_latin:
                     logging.error(f"Failed to read CSV '{file_name}' with latin1: {e_latin}")
                     st.error(f"Error reading CSV (tried UTF-8 and latin1): {e_latin}")
                     return None
            except Exception as e_csv:
                 logging.error(f"General error reading CSV '{file_name}': {e_csv}")
                 st.error(f"Error reading CSV: {e_csv}")
                 return None
        elif file_name.endswith(('.xls', '.xlsx')):
            try:
                 uploaded_file.seek(0)
                 df = pd.read_excel(uploaded_file, dtype=str, keep_default_na=False)
                 logging.info(f"Read Excel file '{file_name}'.")
            except Exception as e_excel:
                 logging.error(f"Error reading Excel '{file_name}': {e_excel}")
                 st.error(f"Error reading Excel file: {e_excel}")
                 return None
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            logging.warning(f"Unsupported file format: {file_name}")
            return None

        # --- Post-load Cleaning ---
        if df is not None:
             original_shape = df.shape
             df.replace("", np.nan, inplace=True)
             df.dropna(axis=0, how='all', inplace=True)
             df.dropna(axis=1, how='all', inplace=True)
             df.fillna('', inplace=True)
             cleaned_shape = df.shape
             logging.info(f"Cleaned DataFrame shape: {cleaned_shape} (Original: {original_shape})")
             if df.empty:
                 st.warning("The file appears empty or contains only empty rows/columns after cleaning.")
                 logging.warning(f"File '{file_name}' resulted in empty DataFrame.")
             return df
        else:
             st.error("Failed to create DataFrame from file.")
             logging.error(f"DataFrame creation failed unexpectedly for {file_name}")
             return None

    except Exception as e:
        st.error(f"An critical error occurred during file loading: {e}")
        logging.exception(f"Critical file loading error for {getattr(uploaded_file, 'name', 'N/A')}")
        return None

# --- Tag Handling Function ---
def add_tags(existing_tags, tags_to_add):
    """
    Adds new tags to existing comma-separated tag string, avoiding duplicates.
    Ensures consistent formatting (Title Case, sorted).

    Args:
        existing_tags (str | float | None): Current tags (might be NaN).
        tags_to_add (str): New tags to add, comma-separated.

    Returns:
        str: Updated, formatted, comma-separated tag string.
    """
    current_tags_str = str(existing_tags) if pd.notna(existing_tags) else ""
    new_tags_str = str(tags_to_add) if pd.notna(tags_to_add) else ""

    # Split, strip whitespace, convert to lowercase for comparison
    current_tags = set(tag.strip().lower() for tag in current_tags_str.split(',') if tag.strip())
    new_tags = set(tag.strip().lower() for tag in new_tags_str.split(',') if tag.strip())

    # Update the set (duplicates handled automatically)
    current_tags.update(new_tags)

    # Format for output: Remove empty strings, title case, sort alphabetically
    formatted_tags = sorted([tag.strip().title() for tag in current_tags if tag.strip()])

    return ', '.join(formatted_tags)

# --- API Key Validation ---
# (Identical to the one used in the Themer app)
def validate_api_key(api_key_to_validate):
    """
    Validates the Gemini API key by trying to list available models.

    Args:
        api_key_to_validate (str): The API key string.

    Returns:
        tuple: (bool, str) indicating (is_valid, message)
    """
    if not api_key_to_validate:
        return False, "API key is empty."
    try:
        genai.configure(api_key=api_key_to_validate)
        models = genai.list_models()
        # Check if *any* model supporting 'generateContent' exists
        if not any('generateContent' in m.supported_generation_methods for m in models):
            logging.warning("Could not find any models supporting generateContent via list_models(). Key might be restricted or invalid.")
            # Consider this a potential failure depending on requirements
            # return False, "No models supporting content generation found. Check key permissions."
        logging.info("API Key validated successfully via list_models().")
        return True, "API Key Validated"
    except Exception as e:
        logging.error(f"API Key validation failed: {e}")
        error_msg = str(e).lower()
        if "api key not valid" in error_msg or "invalid api key" in error_msg:
            return False, "Invalid API Key provided."
        elif "permission denied" in error_msg:
             return False, "Permission denied. Check API key permissions."
        elif "quota" in error_msg:
             return False, "Quota exceeded or billing issue. Check Google Cloud Console."
        else:
            return False, f"API Connection Error ({type(e).__name__}). Check network or key details."

# --- Score Coloring Function ---
# Use specific colors from config
SCORE_COLOR_MAP = {
    'error': CHART_ERROR_COLOR,
    'orange': '#FF9800', # Keep original orange or use CHART_WARNING_COLOR? Let's use a distinct orange.
    'warning': CHART_WARNING_COLOR,
    'success': CHART_SUCCESS_COLOR,
    'text': BODY_TEXT_COLOR # Fallback color
}

def get_color_for_score(score, thresholds):
    """
    Gets the color hex code based on score and defined thresholds.

    Args:
        score (numeric | str | None): The score to evaluate.
        thresholds (dict): Dictionary mapping color names (keys from SCORE_COLOR_MAP)
                           to (lower_bound, upper_exclusive_bound) tuples.

    Returns:
        str: Hex color code.
    """
    default_color = SCORE_COLOR_MAP['text']
    if not thresholds or score is None or score == '':
        return default_color
    try:
        score_num = float(score)
        if pd.isna(score_num): # Handle numpy NaN etc.
            return default_color
    except (ValueError, TypeError):
        return default_color # Return default if score is not numeric

    # Find the highest threshold upper bound to handle scores >= top threshold
    max_upper_bound = -float('inf')
    top_color_name = None
    for color_name, (lower, upper) in thresholds.items():
         if upper > max_upper_bound:
              max_upper_bound = upper
              top_color_name = color_name

    # Check against thresholds
    for color_name, (lower, upper) in thresholds.items():
        # Check if score falls within the range [lower, upper)
        if lower <= score_num < upper:
            return SCORE_COLOR_MAP.get(color_name, default_color)

    # Handle score >= the highest upper bound (belongs to the top category)
    # Allow a small tolerance for floating point comparisons if needed, e.g., 101 vs 100.999
    if top_color_name and score_num >= thresholds[top_color_name][0]: # Check >= lower bound of top category
         return SCORE_COLOR_MAP.get(top_color_name, default_color)

    # Fallback if no range matches (should ideally not happen with well-defined thresholds)
    logging.warning(f"Score {score_num} did not fall into any defined threshold range: {thresholds}")
    return default_color


# --- Logo Loading Function ---
# (Identical to the one used in the Themer app's main_app.py)
def get_base64_of_bin_file(bin_file):
    """Reads a binary file and returns its base64 encoded string."""
    try:
        with open(str(bin_file), 'rb') as f: # Ensure path is string
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        # Log warning, user feedback handled in main app sidebar
        logging.warning(f"Logo file not found at {bin_file}")
        return None
    except Exception as e:
        # Log error, user feedback handled in main app sidebar
        logging.error(f"Error loading logo file {bin_file}: {e}")
        return None