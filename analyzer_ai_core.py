# analyzer_ai_core.py
"""
Core functions for interacting with the Google Generative AI API
for the Response Analyzer application.
"""

import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import pandas as pd
import numpy as np
import json
import logging
import time

# Import utilities and configuration from analyzer modules
from analyzer_utils import batch_responses, clean_and_parse_json_list
from analyzer_config import SAFETY_SETTINGS # Use specific safety settings if needed

# --- Model Initialization Helper ---
# (Optional: Reuse the helper from the Themer app's ai_core if desired,
# or define it here if keeping apps completely separate)
_ANALYZER_MODEL_CACHE = {}

def get_analyzer_model(model_name="gemini-1.5-flash-latest", api_key=None):
    """Initializes and returns a GenerativeModel instance for the Analyzer."""
    cache_key = (model_name, bool(api_key))
    if cache_key in _ANALYZER_MODEL_CACHE:
        return _ANALYZER_MODEL_CACHE[cache_key]

    if not api_key:
        st.error("API Key is missing. Cannot initialize AI model.")
        logging.error("Analyzer: Attempted to get model without API key.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        _ANALYZER_MODEL_CACHE[cache_key] = model
        logging.info(f"Analyzer: Initialized GenerativeModel: {model_name}")
        return model
    except Exception as e:
        st.error(f"Analyzer: Failed to initialize Gemini Model ({model_name}): {e}")
        logging.error(f"Analyzer: Gemini Model init failed for {model_name}: {e}")
        return None

# --- Response Evaluation Function ---
def evaluate_responses(survey_question, responses, is_batch, batch_size, generation_config, api_key):
    """
    Evaluate survey responses using Gemini API based on predefined criteria.

    Args:
        survey_question (str): The survey question asked.
        responses (list): A list of response strings to analyze.
        is_batch (bool): Flag indicating if batch processing should be used.
        batch_size (int): Number of responses per API call in batch mode.
        generation_config (GenerationConfig): AI model generation parameters.
        api_key (str): The API key for authentication.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each response,
                      including scores, status, sentiment, explanation, etc. Returns an
                      empty DataFrame on critical failure.
    """
    model = get_analyzer_model("gemini-1.5-flash-latest", api_key) # Flash often faster for structured output
    if not model:
        return pd.DataFrame() # Error already shown

    # Ensure responses is a list of non-empty strings
    if not isinstance(responses, list):
        logging.warning("evaluate_responses received non-list input for responses.")
        responses = [] # Treat as empty
    responses_list = [str(r).strip() for r in responses if r and isinstance(r, str) and str(r).strip()]
    if not responses_list:
        st.error("No valid, non-empty responses found to analyze.")
        logging.error("evaluate_responses: No valid responses after cleaning.")
        return pd.DataFrame()

    # Determine batches (even if not is_batch, creates a single batch of one item)
    batches = batch_responses(responses_list, batch_size) if is_batch else [[responses_list[0]]]
    all_evaluations = [] # Stores dicts for each response
    total_responses = len(responses_list)
    processed_count = 0
    responses_processed_successfully = 0 # Count successful API results

    # --- Evaluation Prompt Template ---
    # Using the revised prompt structure from the original full code
    evaluation_prompt_template = """
    Analyze EACH response in the provided list based on the survey question and the criteria below.
    Return a JSON list where EACH element corresponds to ONE response analysis.

    **Survey Question:**
    "{survey_question}"
**Evaluation Criteria & Keys for EACH response object in the JSON list:**

    1.  **relevance:** (0-10) How directly does the response address the specific question?
    2.  **completeness:** (0-10) How thoroughly does the response answer all aspects implied?
    3.  **specificity:** (0-10) Level of detail and concrete information.
    4.  **language_quality:** (0-10) Coherence, clarity, grammar, vocabulary.
    5.  **sentiment_alignment:** (0-10) Appropriateness of tone/sentiment for the question's context.
    6.  **topic_status:** ("On-topic" or "Off-topic") Does it address the question's core subject? Be critical using the examples provided.
        * Off-topic examples: Wrong brand category (luxury perfume vs Axe), wrong time period (80s vs 2010s movies), wrong geography (Italian vs French cuisine), wrong price tier (budget vs premium phone), wrong industry (commercial vs military aircraft), wrong field (cardiologist vs dermatologist), wrong product type (gaming laptop vs tablet), answers completely different question.
    7.  **sentiment:** (Dictionary with "label" and "score")
        *   "label": Classify sentiment as "Positive", "Negative", or "Neutral".
        *   "score": Provide a confidence score (0.0 to 1.0) for the assigned label.
    8.  **overall_score:** (0-100) Calculate this as EXACTLY (Sum of relevance, completeness, specificity, language_quality, sentiment_alignment) * 2. Integer output required.
    9.  **explanation:** (1 concise sentence, max 15 words) Justify the overall evaluation, highlighting key strengths/weaknesses.
    10. **bot_likelihood:** (0-10) Analyze the following response for signs of AI authorship, focusing on formality, lack of personal experience, absence of specific examples, and a neutral or generic tone. Assign a bot likelihood score from 1 (definitely human) to 10 (definitely bot), and explain your reasoning in one sentence.

    ---
    **Example:**

    **Input Responses for this Batch:**
    ["It was okay, but lacked detail.", "Amazing! Super helpful."]

    **Output Format Example:**
    [
      {{
        "relevance": 7,
        "completeness": 5,
        "specificity": 3,
        "language_quality": 8,
        "sentiment_alignment": 7,
        "topic_status": "On-topic",
        "sentiment": {{"label": "Neutral", "score": 0.7}},
        "overall_score": 60,
        "explanation": "Addresses question somewhat but lacks specifics.",
        "bot_likelihood": 1
      }},
      {{
        "relevance": 9,
        "completeness": 8,
        "specificity": 7,
        "language_quality": 9,
        "sentiment_alignment": 9,
        "topic_status": "On-topic",
        "sentiment": {{"label": "Positive", "score": 0.95}},
        "overall_score": 84,
        "explanation": "Clearly answers with positive sentiment and detail.",
        "bot_likelihood": 0
      }}
    ]
    ---

    **Input Responses for this Batch:**
    {json_batch}

    **Output Format:**
    Provide the analysis as a **single JSON list** containing multiple JSON objects, one for each response analyzed in this batch. Each object must contain exactly the keys specified above. Do not use markdown formatting (like ```json). Ensure all numeric scores are integers/floats within their specified ranges. Respond ONLY with the JSON list, no introductory text.
    """
    # Safety Settings (Keep relaxed or adjust as needed)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Note: Safety settings are defined in config and passed during API call

    # --- Progress Bar Setup ---
    progress_bar = None
    if is_batch and total_responses > 1:
        progress_bar = st.progress(0, text="Initializing Analysis...")
    start_time = time.time()

    # --- Batch Processing Loop ---
    for i, batch in enumerate(batches):
        if not batch: continue # Skip empty batches

        prompt = evaluation_prompt_template.format(
            survey_question=survey_question,
            json_batch=json.dumps(batch) # Pass the current batch
        )

        try:
            logging.info(f"Sending Batch {i+1}/{len(batches)} ({len(batch)} responses) to Gemini for evaluation.")
            result = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=SAFETY_SETTINGS # Use settings from config
            )

            # --- Response Handling & Parsing ---
            block_reason = None
            response_text = None
            try: # Safe access to feedback/candidates
                if hasattr(result, 'prompt_feedback') and result.prompt_feedback and result.prompt_feedback.block_reason:
                     block_reason = result.prompt_feedback.block_reason.name
                elif result.candidates and result.candidates[0].finish_reason.name != 'STOP':
                      block_reason = f"Finish Reason: {result.candidates[0].finish_reason.name}"
                      if result.candidates[0].finish_reason.name == 'MAX_TOKENS':
                          logging.warning(f"Batch {i+1} hit MAX_TOKENS limit.")
                          st.warning(f"Batch {i+1} response may have been truncated.")

                # Try getting text only if not clearly blocked
                if not block_reason or block_reason == 'Finish Reason: MAX_TOKENS':
                    response_text = result.text

            except ValueError as e: # Often indicates blocked content
                 logging.warning(f"ValueError accessing result.text for Batch {i+1}: {e}. Assuming blocked.")
                 if not block_reason: block_reason = "CONTENT_BLOCKED"
            except Exception as e: # Catch other potential errors
                 logging.error(f"Unexpected error accessing result parts for Batch {i+1}: {e}")
                 if not block_reason: block_reason = f"ACCESS_ERROR: {e}"

            # --- Handle Errors / Blocking ---
            if block_reason and block_reason != 'Finish Reason: MAX_TOKENS':
                 st.error(f"API response blocked/error for Batch {i+1}. Reason: {block_reason}. Skipping batch.")
                 logging.error(f"API response blocked/error for Batch {i+1}. Reason: {block_reason}")
                 # Add error placeholders for each item in the failed batch
                 for response_in_batch in batch:
                      all_evaluations.append({'full_response': response_in_batch, 'error': f'API Error: {block_reason}'})
                 processed_count += len(batch)
                 continue # Skip to next batch

            if not response_text:
                st.error(f"API returned no response text for Batch {i+1}. Skipping batch.")
                logging.error(f"API returned empty response text for Batch {i+1}.")
                # Add error placeholders
                for response_in_batch in batch:
                    all_evaluations.append({'full_response': response_in_batch, 'error': 'API Error: Empty Response'})
                processed_count += len(batch)
                continue # Skip to next batch

            # --- Parse Valid JSON Response ---
            # Uses the specific list parser from utils
            parsed_batch = clean_and_parse_json_list(response_text)

            # --- Validate Parsed Results ---
            validated_batch = []
            metrics_to_sum = ['relevance', 'completeness', 'specificity', 'language_quality', 'sentiment_alignment']

            if len(parsed_batch) != len(batch):
                 logging.warning(f"Batch {i+1}: Mismatch requested ({len(batch)}) vs received ({len(parsed_batch)}) results.")
                 # Continue processing received items, placeholders added later

            for idx, item in enumerate(parsed_batch):
                # Basic check if item is a dictionary
                if not isinstance(item, dict):
                     logging.warning(f"Item {idx} in parsed batch {i+1} is not a dict ({type(item)}), skipping.")
                     continue # Skip non-dict items

                calculated_score = 0
                valid_metrics_count = 0
                validation_issues = [] # Store issues for the 'error' column

                # Recalculate Overall Score for consistency
                for key in metrics_to_sum:
                    value = item.get(key)
                    try:
                        score = int(float(value)) # Allow float then convert
                        if 0 <= score <= 10:
                            calculated_score += score
                            valid_metrics_count += 1
                        else:
                            validation_issues.append(f"{key}({value}) out of range [0-10]")
                            item[key] = pd.NA # Mark invalid score as NA
                    except (ValueError, TypeError, KeyError):
                        issue = f"'{key}' missing" if key not in item else f"'{key}'('{value}') invalid"
                        validation_issues.append(issue)
                        item[key] = pd.NA # Mark invalid score as NA

                if valid_metrics_count == len(metrics_to_sum):
                    final_overall_score = min(calculated_score * 2, 100) # Cap at 100
                    item['overall_score'] = final_overall_score # Overwrite/set score
                else:
                    item['overall_score'] = pd.NA # Set to NA if metrics missing
                    validation_issues.append("Overall score calc failed")

                # Validate Sentiment Structure
                sentiment_value = item.get('sentiment')
                if not isinstance(sentiment_value, dict) or 'label' not in sentiment_value or 'score' not in sentiment_value:
                    validation_issues.append("Sentiment structure invalid/missing")
                    item['sentiment'] = None # Use None for object column holding dicts/NAs
                else:
                    try: # Validate sentiment score is float 0-1
                        score = float(sentiment_value['score'])
                        if not (0.0 <= score <= 1.0):
                             validation_issues.append(f"Sentiment score ({score}) out of range [0-1]")
                             item['sentiment']['score'] = max(0.0, min(1.0, score)) # Clamp
                    except (ValueError, TypeError):
                         validation_issues.append(f"Sentiment score ('{sentiment_value.get('score')}') invalid")
                         item['sentiment']['score'] = pd.NA # Set score to NA

                # Validate Bot Likelihood Range
                try:
                    bot_lik_val = item.get('bot_likelihood')
                    if bot_lik_val is not None:
                        bot_lik_score = int(float(bot_lik_val))
                        if not (0 <= bot_lik_score <= 10):
                            validation_issues.append(f"Bot likelihood ({bot_lik_score}) out of range [0-10]")
                            item['bot_likelihood'] = pd.NA
                except (ValueError, TypeError):
                     validation_issues.append(f"Bot likelihood ('{item.get('bot_likelihood')}') invalid")
                     item['bot_likelihood'] = pd.NA

                # Add original response text back
                if idx < len(batch): # Match based on index within batch
                    item['full_response'] = batch[idx]
                else: # Should not happen if mismatch check done correctly, but safety net
                    item['full_response'] = f"Response Association Error (Index {idx})"
                    validation_issues.append("Response association failed")

                # Add error column
                item['error'] = ", ".join(validation_issues) if validation_issues else ""

                # Ensure other expected fields exist (even if empty/NA from API)
                item.setdefault('topic_status', 'Unknown')
                item.setdefault('explanation', '')
                # Fields for manual editing later
                item.setdefault('manual_tags', "")
                item.setdefault('action_item', "")

                validated_batch.append(item)
                responses_processed_successfully += 1
                # End of item validation loop

            all_evaluations.extend(validated_batch)

            # Add placeholders for responses requested but not returned by API
            if len(parsed_batch) < len(batch):
                missing_count = len(batch) - len(parsed_batch)
                st.warning(f"Batch {i+1}: API returned {missing_count} fewer results than expected. Adding placeholders.")
                logging.warning(f"Batch {i+1}: Adding {missing_count} placeholders for missing API results.")
                # Add placeholders for indices that were requested but not in parsed_batch
                for missing_idx in range(len(parsed_batch), len(batch)):
                    all_evaluations.append({
                        'full_response': batch[missing_idx],
                        'error': 'Analysis Missing: No result returned by API for this item.'
                    })

            processed_count += len(batch) # Increment by number requested for progress calc

        except Exception as e: # Catch unexpected errors during batch processing
            st.error(f"Unexpected error processing Batch {i+1}: {e}")
            logging.exception(f"Exception occurred during batch {i+1} evaluation processing")
            # Add error placeholders for the entire failed batch
            for response_in_batch in batch:
                all_evaluations.append({'full_response': response_in_batch, 'error': f'Batch Processing Failed: {e}'})
            processed_count += len(batch)
            continue # Move to next batch

        # --- Update Progress Bar ---
        if progress_bar:
            progress = min(1.0, processed_count / total_responses) if total_responses > 0 else 0
            progress_bar.progress(progress, text=f"Analyzing... {processed_count}/{total_responses} responses attempted ({progress * 100:.0f}%)")
        # End of batch loop

    # --- Finalize ---
    if progress_bar: progress_bar.empty()
    total_time = time.time() - start_time
    logging.info(f"Evaluation complete. Attempted {processed_count}, successful results {responses_processed_successfully}, final records {len(all_evaluations)} in {total_time:.2f}s")

    if not all_evaluations:
        st.warning("Analysis returned no results. Check inputs and API key.")
        return pd.DataFrame()

    # --- Create DataFrame ---
    # Define expected columns based on the prompt structure
    expected_cols = [
        'full_response', 'relevance', 'completeness', 'specificity', 'language_quality',
        'sentiment_alignment', 'topic_status', 'sentiment', # New combined sentiment field
        'overall_score', 'explanation', 'bot_likelihood',
        'manual_tags', 'action_item', 'error'
    ]
    df = pd.DataFrame(all_evaluations)

    # Add any missing expected columns with default NA/None
    for col in expected_cols:
        if col not in df.columns:
            if col == 'sentiment': # Sentiment should hold dicts or None/NA
                df[col] = None
                df[col] = df[col].astype(object)
            elif col in ['manual_tags', 'action_item', 'error', 'explanation', 'topic_status', 'full_response']:
                df[col] = "" if col != 'topic_status' else "Unknown" # Default empty string or Unknown
            else: # Numeric columns default to NA
                df[col] = pd.NA

    # Ensure correct data types for numeric cols, coercing errors
    numeric_cols_to_convert = [
        'relevance', 'completeness', 'specificity', 'language_quality',
        'sentiment_alignment', 'overall_score', 'bot_likelihood'
    ]
    for col in numeric_cols_to_convert:
         if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce invalid numbers to NaT/NaN

    # Reorder columns for final DataFrame
    present_cols = [col for col in expected_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in present_cols] # Keep any unexpected extra cols at the end
    df = df[present_cols + other_cols]

    logging.info(f"Returning analysis DataFrame with shape: {df.shape}")
    return df


# --- AI Question Answering Function ---
# (This function is largely the same as in the Themer app, just uses the analyzer model helper)
def ask_ai_about_data(survey_question, responses_list, user_question, generation_config, api_key):
    """Ask the AI a question about the provided survey data."""
    if not responses_list:
        logging.warning("Analyzer Q&A: No responses provided.")
        return "Error: No responses provided to ask questions about."

    model = get_analyzer_model("gemini-1.5-flash-latest", api_key) # Use Flash for Q&A
    if not model:
        return "Error: Could not initialize AI model for Q&A."

    # Format context, limit length
    responses_context = "\n".join([f"- {r}" for r in responses_list if r and isinstance(r, str)])
    max_context_chars = 15000
    if len(responses_context) > max_context_chars:
        responses_context = responses_context[:max_context_chars] + "\n... (responses truncated)"
        logging.warning("Analyzer Q&A: Context truncated.")
        st.caption(f"Note: Input responses context truncated to {max_context_chars} chars for AI.")

    if not responses_context.strip():
        logging.warning("Analyzer Q&A: Context empty after processing.")
        return "Error: No valid response content for Q&A context."

    qa_prompt = f"""
Context:
You are analyzing feedback for the following survey question.

Survey Question:
"{survey_question}"

Provided Responses (potentially truncated):
{responses_context}

---
Task:
Based *only* on the provided Survey Question and Responses context above, answer the following user question as accurately and concisely as possible. Do not invent information. If asked for examples, retrieve them directly from the 'Provided Responses'. Use markdown formatting where helpful. Avoid introductions like "Based on the data...".

User Question:
"{user_question}"

Answer:
"""

    try:
        logging.info(f"Analyzer Q&A: Sending query: {user_question[:50]}...")
        # Use slightly different config for Q&A? Optional.
        qa_config = GenerationConfig(
            temperature=0.5, # Allow a bit more flexibility for summarization/explanation
            top_k=generation_config.top_k, top_p=generation_config.top_p,
            max_output_tokens=max(1024, generation_config.max_output_tokens) # Ensure decent length
        )
        result = model.generate_content(
            qa_prompt,
            generation_config=qa_config, # Use specific QA config
            safety_settings=SAFETY_SETTINGS # Use settings from config
        )

        # Robust Response Handling (copied from evaluate_responses)
        block_reason = None
        response_text = None
        try:
            if hasattr(result, 'prompt_feedback') and result.prompt_feedback and result.prompt_feedback.block_reason:
                 block_reason = result.prompt_feedback.block_reason.name
            elif result.candidates and result.candidates[0].finish_reason.name != 'STOP':
                  block_reason = f"Finish Reason: {result.candidates[0].finish_reason.name}"
            if not block_reason or block_reason == 'Finish Reason: MAX_TOKENS':
                response_text = result.text
        except ValueError as e:
            logging.warning(f"ValueError accessing Q&A result.text: {e}")
            if not block_reason: block_reason = "CONTENT_BLOCKED"
        except Exception as e:
            logging.error(f"Unexpected error accessing Q&A result parts: {e}")
            if not block_reason: block_reason = f"ACCESS_ERROR: {e}"

        if block_reason and block_reason != 'Finish Reason: MAX_TOKENS':
             st.error(f"AI Q&A response blocked. Reason: {block_reason}")
             logging.error(f"AI Q&A blocked: {block_reason}")
             return f"Error: AI response was blocked (Reason: {block_reason})."

        if not response_text:
             st.error("AI Q&A returned no response text.")
             logging.error("AI Q&A returned empty response text.")
             return "Error: AI returned an empty response."

        logging.info("Analyzer Q&A: Response received successfully.")
        return response_text.strip()

    except Exception as e:
        st.error(f"Error during AI Q&A: {e}")
        logging.exception("Analyzer: Exception occurred during AI Q&A")
        return f"Error: An exception occurred generating the answer: {e}"
