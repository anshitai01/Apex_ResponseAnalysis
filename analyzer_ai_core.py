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
# (Keep the existing evaluate_responses function here)
def evaluate_responses(survey_question, responses, is_batch, batch_size, generation_config, api_key):
    """
    Evaluate survey responses using Gemini API based on predefined criteria.
    """
    model = get_analyzer_model("gemini-1.5-flash-latest", api_key)
    if not model: return pd.DataFrame()

    if not isinstance(responses, list): responses = []
    responses_list = [str(r).strip() for r in responses if r and isinstance(r, str) and str(r).strip()]
    if not responses_list:
        st.error("No valid, non-empty responses found to analyze."); logging.error("evaluate_responses: No valid responses after cleaning."); return pd.DataFrame()

    batches = batch_responses(responses_list, batch_size) if is_batch else [[responses_list[0]]]
    all_evaluations = []; total_responses = len(responses_list)
    processed_count = 0; responses_processed_successfully = 0

    evaluation_prompt_template = """
    Analyze EACH response in the provided list based on the survey question and the criteria below. Return a JSON list where EACH element corresponds to ONE response analysis.
    **Survey Question:** "{survey_question}"
    **Evaluation Criteria & Keys for EACH response object in the JSON list:**
    1.  **relevance:** (0-10) How directly does the response address the specific question?
    2.  **completeness:** (0-10) How thoroughly does the response answer all aspects implied?
    3.  **specificity:** (0-10) Level of detail and concrete information.
    4.  **language_quality:** (0-10) Coherence, clarity, grammar, vocabulary.
    5.  **sentiment_alignment:** (0-10) Appropriateness of tone/sentiment for the question's context.
    6.  **topic_status:** ("On-topic" or "Off-topic") Does it address the question's core subject? Be critical using the examples provided. * Off-topic examples: Wrong brand category (luxury perfume vs Axe), wrong time period (80s vs 2010s movies), wrong geography (Italian vs French cuisine), wrong price tier (budget vs premium phone), wrong industry (commercial vs military aircraft), wrong field (cardiologist vs dermatologist), wrong product type (gaming laptop vs tablet), answers completely different question.
    7.  **sentiment:** (Dictionary with "label" and "score") *   "label": Classify sentiment as "Positive", "Negative", or "Neutral". *   "score": Provide a confidence score (0.0 to 1.0) for the assigned label.
    8.  **overall_score:** (0-100) Calculate this as EXACTLY (Sum of relevance, completeness, specificity, language_quality, sentiment_alignment) * 2. Integer output required.
    9.  **explanation:** (1 concise sentence, max 15 words) Justify the overall evaluation, highlighting key strengths/weaknesses.
    10. **bot_likelihood:** (0-10) Analyze the response for signs of AI authorship (formality, lack of personal experience/specifics, generic tone). Assign a score from 1 (human) to 10 (bot).
    ---
    **Input Responses for this Batch:** {json_batch}
    **Output Format:** Provide the analysis as a **single JSON list** containing multiple JSON objects, one for each response analyzed in this batch. Each object must contain exactly the keys specified above. Do not use markdown formatting (like ```json). Ensure all numeric scores are integers/floats within their specified ranges. Respond ONLY with the JSON list, no introductory text.
    """

    progress_bar = None
    if is_batch and total_responses > 1: progress_bar = st.progress(0, text="Initializing Analysis...")
    start_time = time.time()

    for i, batch in enumerate(batches):
        if not batch: continue
        prompt = evaluation_prompt_template.format(survey_question=survey_question, json_batch=json.dumps(batch))
        try:
            logging.info(f"Sending Batch {i+1}/{len(batches)} ({len(batch)} responses) to Gemini for evaluation.")
            result = model.generate_content(prompt, generation_config=generation_config, safety_settings=SAFETY_SETTINGS)

            block_reason = None; response_text = None
            try: # Safe access
                if hasattr(result, 'prompt_feedback') and result.prompt_feedback and result.prompt_feedback.block_reason: block_reason = result.prompt_feedback.block_reason.name
                elif result.candidates and result.candidates[0].finish_reason.name != 'STOP':
                    block_reason = f"Finish Reason: {result.candidates[0].finish_reason.name}"
                    if result.candidates[0].finish_reason.name == 'MAX_TOKENS': logging.warning(f"Batch {i+1} hit MAX_TOKENS limit."); st.warning(f"Batch {i+1} response may truncated.")
                if not block_reason or block_reason == 'Finish Reason: MAX_TOKENS': response_text = result.text
            except ValueError as e: logging.warning(f"ValueError access text B{i+1}: {e}. Blocked?"); block_reason = block_reason or "CONTENT_BLOCKED"
            except Exception as e: logging.error(f"Unexpected error access result parts B{i+1}: {e}"); block_reason = block_reason or f"ACCESS_ERROR: {e}"

            if block_reason and block_reason != 'Finish Reason: MAX_TOKENS':
                 st.error(f"API err B{i+1}: {block_reason}. Skip."); logging.error(f"API err B{i+1}: {block_reason}")
                 for r in batch: all_evaluations.append({'full_response': r, 'error': f'API Error: {block_reason}'})
                 processed_count += len(batch); continue

            if not response_text:
                st.error(f"API empty B{i+1}. Skip."); logging.error(f"API empty B{i+1}.")
                for r in batch: all_evaluations.append({'full_response': r, 'error': 'API Error: Empty Response'})
                processed_count += len(batch); continue

            parsed_batch = clean_and_parse_json_list(response_text)
            validated_batch = []; metrics_to_sum = ['relevance', 'completeness', 'specificity', 'language_quality', 'sentiment_alignment']
            if len(parsed_batch) != len(batch): logging.warning(f"B{i+1}: Mismatch req({len(batch)}) vs recv({len(parsed_batch)}).")

            for idx, item in enumerate(parsed_batch):
                if not isinstance(item, dict): logging.warning(f"Item {idx} B{i+1} not dict, skip."); continue
                calculated_score = 0; valid_metrics_count = 0; validation_issues = []
                for key in metrics_to_sum: # Recalc overall score
                    v = item.get(key); score=pd.NA
                    try: score = int(float(v)); ok = 0 <= score <= 10
                    except: ok = False
                    if ok: calculated_score += score; valid_metrics_count += 1
                    else: validation_issues.append(f"{key}({v}) issue"); item[key] = pd.NA
                if valid_metrics_count == len(metrics_to_sum): item['overall_score'] = min(calculated_score * 2, 100)
                else: item['overall_score'] = pd.NA; validation_issues.append("Overall score fail")

                sentiment_v = item.get('sentiment') # Validate sentiment
                if not isinstance(sentiment_v, dict) or 'label' not in sentiment_v or 'score' not in sentiment_v:
                    validation_issues.append("Sentiment invalid"); item['sentiment'] = None
                else:
                    try: score = float(sentiment_v['score']); ok = 0.0 <= score <= 1.0
                    except: ok = False
                    if not ok: validation_issues.append(f"Sent score ({sentiment_v.get('score')}) issue"); item['sentiment']['score'] = max(0.0, min(1.0, score)) if ok is False else pd.NA

                try: # Validate bot score
                    bot_lik_v = item.get('bot_likelihood'); ok = False
                    if bot_lik_v is not None: bot_lik_s = int(float(bot_lik_v)); ok = 0 <= bot_lik_s <= 10
                    if not ok: validation_issues.append(f"Bot score ({bot_lik_v}) issue"); item['bot_likelihood'] = pd.NA
                except: validation_issues.append(f"Bot score ({item.get('bot_likelihood')}) issue"); item['bot_likelihood'] = pd.NA

                if idx < len(batch): item['full_response'] = batch[idx] # Add response
                else: item['full_response'] = f"Assoc Error {idx}"; validation_issues.append("Response assoc failed")
                item['error'] = ", ".join(validation_issues) if validation_issues else ""
                item.setdefault('topic_status', 'Unknown'); item.setdefault('explanation', '')
                item.setdefault('manual_tags', ""); item.setdefault('action_item', "")
                validated_batch.append(item); responses_processed_successfully += 1

            all_evaluations.extend(validated_batch)
            if len(parsed_batch) < len(batch): # Add placeholders if needed
                miss_count = len(batch) - len(parsed_batch); st.warning(f"B{i+1}: API miss {miss_count}. Add placeholders.")
                logging.warning(f"B{i+1}: Add {miss_count} placeholders.")
                for miss_idx in range(len(parsed_batch), len(batch)): all_evaluations.append({'full_response': batch[miss_idx], 'error': 'Analysis Missing: No API result.'})
            processed_count += len(batch)

        except Exception as e:
            st.error(f"Unexpected err B{i+1}: {e}"); logging.exception(f"Exception B{i+1} processing")
            for r in batch: all_evaluations.append({'full_response': r, 'error': f'Batch Fail: {e}'})
            processed_count += len(batch); continue

        if progress_bar: progress = min(1.0, processed_count / total_responses) if total_responses > 0 else 0; progress_bar.progress(progress, text=f"Analyzing... {processed_count}/{total_responses} ({progress*100:.0f}%)")

    if progress_bar: progress_bar.empty()
    total_time = time.time() - start_time
    st.session_state['execution_time'] = total_time # Store execution time
    logging.info(f"Evaluation done. Attempt {processed_count}, Success {responses_processed_successfully}, Final {len(all_evaluations)} in {total_time:.2f}s")
    if not all_evaluations: st.warning("No results."); return pd.DataFrame()

    expected_cols = ['full_response', 'relevance', 'completeness', 'specificity', 'language_quality', 'sentiment_alignment', 'topic_status', 'sentiment', 'overall_score', 'explanation', 'bot_likelihood', 'manual_tags', 'action_item', 'error']
    df = pd.DataFrame(all_evaluations)
    for col in expected_cols: # Add missing cols
        if col not in df.columns:
            if col == 'sentiment': df[col] = None; df[col] = df[col].astype(object)
            elif col in ['manual_tags', 'action_item', 'error', 'explanation', 'topic_status', 'full_response']: df[col] = "" if col != 'topic_status' else "Unknown"
            else: df[col] = pd.NA
    numeric_cols = ['relevance', 'completeness', 'specificity', 'language_quality', 'sentiment_alignment', 'overall_score', 'bot_likelihood']
    for col in numeric_cols: # Coerce numeric types
         if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    present_cols = [col for col in expected_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in present_cols]
    df = df[present_cols + other_cols] # Reorder
    logging.info(f"Return analysis DF shape: {df.shape}")
    return df


# --- AI Question Answering Function ---
# (This is the version from the original Analyzer code you provided)
def ask_ai_about_data(survey_question, responses_list, user_question, generation_config, api_key):
    """Ask the AI a question about the provided survey data."""
    logging.info(f"Analyzer Q&A: Processing request. Q: '{user_question[:50]}...'")
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
Based *only* on the provided Survey Question and Responses context above, answer the following user question as accurately and concisely as possible. Do not invent information not present in the responses. If the question asks for specific examples (like verbatims), provide them directly from the 'Provided Responses'. Use markdown formatting for lists or emphasis where appropriate.

User Question:
"{user_question}"

Answer:
"""

    # Safety Settings (Keep relaxed or adjust - using imported SAFETY_SETTINGS)
    # safety_settings = { ... } # Not needed here, imported

    try:
        logging.info(f"Analyzer Q&A: Sending query to Gemini: {user_question[:50]}...")
        # Use slightly different config for Q&A? Optional.
        qa_config = GenerationConfig(
            temperature=0.5, # Allow some flexibility
            top_k=generation_config.top_k, top_p=generation_config.top_p,
            max_output_tokens=max(1024, generation_config.max_output_tokens)
        )
        result = model.generate_content(
            qa_prompt,
            generation_config=qa_config, # Use specific QA config
            safety_settings=SAFETY_SETTINGS # Use imported settings
        )

        # Robust Response Handling
        block_reason = None; response_text = None
        try:
            if hasattr(result, 'prompt_feedback') and result.prompt_feedback and result.prompt_feedback.block_reason: block_reason = result.prompt_feedback.block_reason.name
            elif result.candidates and result.candidates[0].finish_reason.name != 'STOP': block_reason = f"Finish Reason: {result.candidates[0].finish_reason.name}"
            if not block_reason or block_reason == 'Finish Reason: MAX_TOKENS': response_text = result.text
        except ValueError as e: logging.warning(f"ValueError accessing Q&A text: {e}"); block_reason = block_reason or "CONTENT_BLOCKED"
        except Exception as e: logging.error(f"Unexpected error accessing Q&A result: {e}"); block_reason = block_reason or f"ACCESS_ERROR: {e}"

        if block_reason and block_reason != 'Finish Reason: MAX_TOKENS':
             st.error(f"AI Q&A response blocked. Reason: {block_reason}")
             logging.error(f"Analyzer Q&A blocked: {block_reason}")
             return f"Error: The AI's response was blocked (Reason: {block_reason})."

        if not response_text:
             st.error("AI Q&A returned no response text.")
             logging.error("Analyzer Q&A returned empty response text.")
             return "Error: AI returned an empty response."

        logging.info("Analyzer Q&A: Response received successfully.")
        return response_text.strip() # Return the AI's answer

    except Exception as e:
        st.error(f"An unexpected error occurred during AI Q&A: {e}")
        logging.exception("Analyzer: Exception occurred during AI Q&A")
        return f"Error: An exception occurred while generating the answer: {e}"
