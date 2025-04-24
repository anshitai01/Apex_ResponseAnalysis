# analyzer_main_app.py
"""
Main Streamlit application file for the PPL APEX AI Response Analyzer.
Orchestrates the UI, state management, and interactions between modules.
Run this file with `streamlit run analyzer_main_app.py`.
"""

# --- Core Imports ---
import streamlit as st
import pandas as pd
import numpy as np # Often needed with pandas
import logging
import time
from pathlib import Path # Needed for logo path
from google.generativeai.types import GenerationConfig # For creating config object

# --- Application Modules Imports ---
import analyzer_config as cfg
from analyzer_styling import apply_styling
import analyzer_utils as utils
import analyzer_ai_core as ai_core
import analyzer_processing as processing
import analyzer_ui_components as ui

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Response Analyzer - Apex", # Updated Title
    page_icon="📊", # Analyzer icon
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, # Adjust level as needed (DEBUG, INFO, WARNING)
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("AI Response Analyzer Application Started (Apex Theme).")

# --- Apply Custom Styling ---
apply_styling() # Applies the Phronesis Apex theme CSS

# --- Session State Initialization ---
# Ensure essential keys exist using defaults from config
default_state_values = {
    cfg.INITIALIZED_KEY: False, cfg.API_KEY: None, cfg.API_KEY_SOURCE_KEY: None,
    cfg.INPUT_METHOD_KEY: 'Paste Text', cfg.SURVEY_QUESTION_INPUT_KEY: '',
    cfg.RESPONSES_KEY: [], cfg.UPLOADED_DF_KEY: None, cfg.SELECTED_COLUMN_KEY: None,
    cfg.CURRENT_FILE_NAME_KEY: None, cfg.RESPONSES_INPUT_AREA_KEY: '',
    cfg.SELECTED_COLUMN_INDEX_KEY: 0, cfg.ANALYSIS_DF_KEY: None,
    cfg.CLUSTERING_DF_KEY: None, cfg.AI_QA_HISTORY_KEY: [],
    cfg.SELECTED_ROW_INDEX_KEY: 0, cfg.ANALYSIS_MODE_KEY: "Single",
    cfg.RESPONSES_USED_KEY: [], cfg.SURVEY_QUESTION_USED_KEY: "",
    cfg.EXECUTION_TIME_KEY: None, cfg.SIMILARITY_THRESHOLD_KEY: 70,
    # Default AI parameters
    cfg.BATCH_SIZE_KEY: 25, cfg.GEN_TEMP_KEY: 0.3, cfg.GEN_TOP_K_KEY: 40,
    cfg.GEN_TOP_P_KEY: 0.95, cfg.GEN_MAX_TOKENS_KEY: 8000,
}
for key, default_value in default_state_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
        logging.debug(f"Initialized session state key '{key}' with default.")

# --- Helper function to clear app-specific state ---
def clear_analyzer_app_state(clear_api_keys=False):
    """Clears specific session state keys related to the analyzer app."""
    keys_to_clear = [
        cfg.SURVEY_QUESTION_INPUT_KEY, cfg.RESPONSES_INPUT_AREA_KEY,
        cfg.UPLOADED_DF_KEY, cfg.SELECTED_COLUMN_KEY, cfg.SELECTED_COLUMN_INDEX_KEY,
        cfg.CURRENT_FILE_NAME_KEY, cfg.INPUT_METHOD_KEY, cfg.RESPONSES_KEY,
        cfg.ANALYSIS_DF_KEY, cfg.CLUSTERING_DF_KEY, cfg.AI_QA_HISTORY_KEY,
        cfg.SELECTED_ROW_INDEX_KEY, cfg.ANALYSIS_MODE_KEY, cfg.RESPONSES_USED_KEY,
        cfg.SURVEY_QUESTION_USED_KEY, cfg.EXECUTION_TIME_KEY,
        # Don't clear settings unless clear_api_keys is True
        # cfg.BATCH_SIZE_KEY, cfg.SIMILARITY_THRESHOLD_KEY, ...
        # Widget state keys if defined and need clearing
        cfg.MANUAL_EDITOR_WIDGET_KEY,
    ]
    if clear_api_keys:
        keys_to_clear.extend([
            cfg.INITIALIZED_KEY, cfg.API_KEY, cfg.API_KEY_SOURCE_KEY,
            # Also reset settings if clearing API key
            cfg.BATCH_SIZE_KEY, cfg.GEN_TEMP_KEY, cfg.GEN_TOP_K_KEY,
            cfg.GEN_TOP_P_KEY, cfg.GEN_MAX_TOKENS_KEY, cfg.SIMILARITY_THRESHOLD_KEY,
        ])

    cleared_count = 0
    for key in keys_to_clear:
        if key in st.session_state:
            try: del st.session_state[key]
            except KeyError: pass # Handle potential race conditions if key deleted elsewhere
            cleared_count += 1
            logging.debug(f"Cleared session state key: {key}")

    # Re-initialize essential keys after clearing if necessary
    if clear_api_keys:
        st.session_state[cfg.INITIALIZED_KEY] = False
        st.session_state[cfg.API_KEY] = None
        st.session_state[cfg.API_KEY_SOURCE_KEY] = None
        # Re-apply default settings
        st.session_state[cfg.BATCH_SIZE_KEY] = 25
        st.session_state[cfg.GEN_TEMP_KEY] = 0.3
        # ... (re-apply other default settings if needed) ...
        st.session_state[cfg.SIMILARITY_THRESHOLD_KEY] = 70
        logging.info(f"Cleared {cleared_count} keys and reset API/init/settings state.")
    else:
         # Ensure core data structures are reset to defaults after clearing data
         st.session_state[cfg.RESPONSES_KEY] = []
         st.session_state[cfg.AI_QA_HISTORY_KEY] = []
         # Reset editor state implicitly by clearing analysis_df
         st.session_state[cfg.ANALYSIS_DF_KEY] = None
         st.session_state[cfg.CLUSTERING_DF_KEY] = None
         logging.info(f"Cleared {cleared_count} data/state keys.")


# ==========================================================================
# --- Sidebar ---
# ==========================================================================
with st.sidebar:
    # --- Logo Handling ---
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    LOGO_PATH = current_dir / cfg.LOGO_FILENAME
    logo_base64 = utils.get_base64_of_bin_file(LOGO_PATH) # Use util function

    logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Phronesis Apex Logo" class="sidebar-logo">' if logo_base64 else '<div class="sidebar-logo-placeholder">Logo</div>'
    # Add specific CSS for sidebar logo (can be moved to analyzer_styling.py if preferred)
    st.markdown("""
    <style>
    .sidebar-logo { display: block; margin: 0 auto 1.5rem auto; height: 80px; width: auto; filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));}
    .sidebar-logo-placeholder { height: 80px; width: 80px; background-color: #333; border: 1px dashed #555; display: flex; align-items: center; justify-content: center; color: #888; font-size: 0.9em; text-align: center; border-radius: 5px; margin: 0 auto 1.5rem auto; }
    </style>""", unsafe_allow_html=True)
    st.markdown(logo_html, unsafe_allow_html=True)

    #st.header("⚙️ Settings")

    # --- API Key Handling Logic ---
    # Uses analyzer_config state keys and analyzer_utils.validate_api_key
    api_key_source = None
    if st.session_state.get(cfg.INITIALIZED_KEY, False) and st.session_state.get(cfg.API_KEY):
        st.success("✅ Initialized")
        api_key_source = st.session_state.get(cfg.API_KEY_SOURCE_KEY, "manual")
        st.caption(f"Key Source: {api_key_source.capitalize()}")
    else:
        secrets_api_key = None; secret_key_name = "GEMINI_API_KEY"
        try: # Try loading from secrets
            if hasattr(st, 'secrets') and secret_key_name in st.secrets:
                secrets_api_key = st.secrets[secret_key_name]
                logging.info(f"Analyzer: Found API key in Streamlit Secrets ('{secret_key_name}').")
                api_key_source = "secrets"
            else: api_key_source = "manual"
        except Exception as e: logging.warning(f"Analyzer: Could not access Streamlit secrets: {e}"); api_key_source = "manual"

        if api_key_source == "secrets" and secrets_api_key: # Validate secrets key
            is_valid, message = utils.validate_api_key(secrets_api_key)
            if is_valid:
                st.session_state[cfg.API_KEY] = secrets_api_key
                st.session_state[cfg.INITIALIZED_KEY] = True
                st.session_state[cfg.API_KEY_SOURCE_KEY] = "secrets"
                st.success("✅ Initialized via Secrets!")
                logging.info("Analyzer: Initialized using key from Secrets.")
                time.sleep(0.5); st.rerun()
            else:
                st.error(f"Secrets key invalid: {message}"); logging.error(f"Analyzer: Secrets key validation failed: {message}")
                api_key_source = "manual" # Fallback
                st.session_state.pop(cfg.API_KEY, None); st.session_state.pop(cfg.INITIALIZED_KEY, None); st.session_state.pop(cfg.API_KEY_SOURCE_KEY, None)

        if not st.session_state.get(cfg.INITIALIZED_KEY, False): # Manual input fallback
            if api_key_source == "manual":
                api_key_input = st.text_input("Enter Gemini API Key:", type="password", help="Get key from Google AI Studio.", key="analyzer_api_input_sidebar")
                if st.button("Initialize Analyzer", key=cfg.INIT_BUTTON_KEY, type="primary"):
                    if api_key_input:
                        is_valid, message = utils.validate_api_key(api_key_input)
                        if is_valid:
                            st.session_state[cfg.API_KEY] = api_key_input
                            st.session_state[cfg.INITIALIZED_KEY] = True
                            st.session_state[cfg.API_KEY_SOURCE_KEY] = "manual"
                            st.success("✅ Initialized (Manual Key)!")
                            logging.info("Analyzer: Initialized using manual key.")
                            time.sleep(0.5); st.rerun()
                        else:
                            st.error(f"Init failed: {message}")
                            st.session_state.pop(cfg.API_KEY, None); st.session_state.pop(cfg.INITIALIZED_KEY, None); st.session_state.pop(cfg.API_KEY_SOURCE_KEY, None)
                    else: st.warning("Please enter API Key.")

    # --- Stop Execution if Not Initialized ---
    if not st.session_state.get(cfg.INITIALIZED_KEY, False):
        st.warning("Please provide a valid Gemini API Key and initialize.")
        st.info("Enter key above or configure in Streamlit Secrets as `GEMINI_API_KEY`.")
        st.stop() # Stop script execution

    # --- Post-Initialization Settings ---
    st.markdown("---")
    st.subheader("Analysis Parameters")
    # Use state keys from config for widgets
    st.session_state[cfg.BATCH_SIZE_KEY] = st.slider(
        "Batch Size", 1, 50, value=st.session_state.get(cfg.BATCH_SIZE_KEY, 25),
        key="analyzer_batch_size_slider", help="Responses per API call. Adjust based on speed/errors."
    )

    with st.expander("🤖 Advanced Generation Config", expanded=False):
        st.session_state[cfg.GEN_TEMP_KEY] = st.slider("Temperature", 0.0, 1.0, value=st.session_state.get(cfg.GEN_TEMP_KEY, 0.3), step=0.05, key="analyzer_temp_slider", help="Lower=more factual, Higher=more creative.")
        st.session_state[cfg.GEN_TOP_K_KEY] = st.number_input("Top K", 1, 100, value=st.session_state.get(cfg.GEN_TOP_K_KEY, 40), step=1, key="analyzer_topk_input")
        st.session_state[cfg.GEN_TOP_P_KEY] = st.slider("Top P", 0.0, 1.0, value=st.session_state.get(cfg.GEN_TOP_P_KEY, 0.95), step=0.05, key="analyzer_topp_slider")
        st.session_state[cfg.GEN_MAX_TOKENS_KEY] = st.number_input("Max Output Tokens", 256, 8192, value=st.session_state.get(cfg.GEN_MAX_TOKENS_KEY, 8000), step=128, key="analyzer_maxtokens_input", help="Max length of generated analysis per API call.")

    # --- Reset Button ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset API Key & Clear All Data", key=cfg.RESET_BUTTON_KEY):
        logging.warning("Analyzer: Reset button clicked.")
        clear_analyzer_app_state(clear_api_keys=True)
        st.success("API Key and session data cleared. Re-initializing...")
        time.sleep(1); st.rerun()

# ==========================================================================
# --- Main Application Area ---
# ==========================================================================
st.markdown("<h1 style='margin-bottom: 0.5rem;'>Response Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem; color: #a9b4d2;'>Evaluate open-ended survey responses using AI.</p>", unsafe_allow_html=True)


# --- Define Tabs ---
tab_list = [
    "📝 Input & Run",
    "📈 Results & Details",
    "✍️ Tagging & Actions",
    "💡 Aggregate Insights",
    "🔗 Clustering",
    "❓ Ask AI",
    "💾 Session Overview"
]
input_tab, results_tab, edit_tab, insights_tab, clustering_tab, ai_qa_tab, history_tab = st.tabs(tab_list)

# --- Create GenerationConfig object from state ---
# Done once here, used by evaluate_responses and ask_ai_about_data
analyzer_gen_config = GenerationConfig(
    temperature=st.session_state[cfg.GEN_TEMP_KEY],
    top_k=st.session_state[cfg.GEN_TOP_K_KEY],
    top_p=st.session_state[cfg.GEN_TOP_P_KEY],
    max_output_tokens=st.session_state[cfg.GEN_MAX_TOKENS_KEY]
)

# ======================= Tab 1: Input & Run ==========================
with input_tab:
    st.header("1. Input Data & Run Analysis") # Changed header slightly

    # --- Survey Question Input ---
    st.session_state[cfg.SURVEY_QUESTION_INPUT_KEY] = st.text_area(
        "❓ **Survey Question:**",
        value=st.session_state.get(cfg.SURVEY_QUESTION_INPUT_KEY, ''),
        height=100, key="analyzer_q_input_main", # Use prefixed key
        placeholder="e.g., What did you like most about our new feature?"
    )

    st.markdown("---")
    st.subheader("Provide Responses")

    # --- Input Method Selection ---
    st.radio(
        "Choose input method:", ("Paste Text", "Upload File (.csv, .xlsx)"),
        key=cfg.INPUT_METHOD_KEY, # Controls state directly
        horizontal=True, label_visibility="collapsed"
    )
    input_method = st.session_state[cfg.INPUT_METHOD_KEY]
    responses_list_input = [] # Local list for this tab's processing

    # --- Conditional Input Areas ---
    if input_method == "Paste Text":
        # Clear file state if switching
        if st.session_state.get(cfg.CURRENT_FILE_NAME_KEY):
             keys_to_clear = [cfg.UPLOADED_DF_KEY, cfg.SELECTED_COLUMN_KEY, cfg.SELECTED_COLUMN_INDEX_KEY, cfg.CURRENT_FILE_NAME_KEY, cfg.RESPONSES_KEY]
             for key in keys_to_clear: st.session_state.pop(key, None)
             logging.info("Analyzer: Switched to text input, cleared file state.")

        st.session_state[cfg.RESPONSES_INPUT_AREA_KEY] = st.text_area(
            "📋 Paste Responses (One per line):",
            value=st.session_state.get(cfg.RESPONSES_INPUT_AREA_KEY, ''),
            height=200, key="analyzer_r_input_main",
            placeholder="Response 1...\nResponse 2...\nResponse 3..."
        )
        # Process pasted text
        raw_lines = st.session_state[cfg.RESPONSES_INPUT_AREA_KEY].splitlines()
        responses_list_input = [r.strip() for r in raw_lines if r and r.strip()]
        st.caption(f"{len(responses_list_input)} response(s) entered.")
        # Store processed list in the main state key for analysis
        st.session_state[cfg.RESPONSES_KEY] = responses_list_input

    elif input_method == "Upload File":
        # Clear text state if switching
        if st.session_state.get(cfg.RESPONSES_INPUT_AREA_KEY):
             st.session_state.pop(cfg.RESPONSES_INPUT_AREA_KEY, None)
             st.session_state.pop(cfg.RESPONSES_KEY, None)
             logging.info("Analyzer: Switched to file upload, cleared text input state.")

        uploaded_file = st.file_uploader(
            "📁 Upload Data File:", type=['csv', 'xlsx', 'xls'],
            key="analyzer_file_uploader_main", accept_multiple_files=False
        )
        if uploaded_file is not None:
            current_df = st.session_state.get(cfg.UPLOADED_DF_KEY)
            reload_file = (st.session_state.get(cfg.CURRENT_FILE_NAME_KEY) != uploaded_file.name) or \
                          (current_df is None or not isinstance(current_df, pd.DataFrame))

            if reload_file: # Load or reload file
                logging.info(f"Analyzer: New file ('{uploaded_file.name}') or DF missing. Loading.")
                with st.spinner(f"Reading and processing {uploaded_file.name}..."):
                    st.session_state.pop(cfg.SELECTED_COLUMN_KEY, None)
                    st.session_state.pop(cfg.SELECTED_COLUMN_INDEX_KEY, None)
                    st.session_state.pop(cfg.RESPONSES_KEY, None)
                    st.session_state[cfg.UPLOADED_DF_KEY] = None
                    loaded_data = utils.load_data_from_file(uploaded_file) # Use util function
                    if loaded_data is not None and not loaded_data.empty:
                        st.session_state[cfg.UPLOADED_DF_KEY] = loaded_data
                        st.session_state[cfg.CURRENT_FILE_NAME_KEY] = uploaded_file.name
                        st.success(f"Loaded `{uploaded_file.name}` ({len(loaded_data)} rows). Select column below.")
                        st.rerun() # Rerun needed after successful load
                    else: # load_data handles error messages
                        st.session_state.pop(cfg.UPLOADED_DF_KEY, None)
                        st.session_state.pop(cfg.CURRENT_FILE_NAME_KEY, None)

            # --- Column Selection (Only if DF is loaded) ---
            df_for_selection = st.session_state.get(cfg.UPLOADED_DF_KEY)
            if df_for_selection is not None and isinstance(df_for_selection, pd.DataFrame):
                st.markdown("---")
                st.write(f"**File:** `{st.session_state.get(cfg.CURRENT_FILE_NAME_KEY, 'N/A')}` ({len(df_for_selection)} rows)")
                with st.expander("Preview Data"):
                    try: st.dataframe(df_for_selection.head(), use_container_width=True, height=200)
                    except Exception as e: st.warning(f"Could not display preview: {e}")

                available_columns = df_for_selection.columns.tolist()
                if available_columns:
                    current_index = st.session_state.get(cfg.SELECTED_COLUMN_INDEX_KEY, 0)
                    if current_index >= len(available_columns): current_index = 0
                    # Auto-detect plausible column only if none selected yet
                    if st.session_state.get(cfg.SELECTED_COLUMN_KEY) is None:
                        plausible_kw = ['response', 'feedback', 'text', 'comment', 'verbatim', 'open end', 'answer']
                        plausible_cols = [c for c in available_columns if any(kw in str(c).lower() for kw in plausible_kw)]
                        if plausible_cols:
                            try: current_index = available_columns.index(plausible_cols[0]); logging.info(f"Analyzer: Auto-selected column '{plausible_cols[0]}'")
                            except ValueError: pass # Keep 0 if error

                    selected_col_name = st.selectbox(
                        "⬇️ **Select column with responses:**", options=available_columns, index=current_index,
                        key="analyzer_column_selector", help="Choose the column with text answers."
                    )
                    # Update state if selection changed
                    if selected_col_name != st.session_state.get(cfg.SELECTED_COLUMN_KEY):
                        st.session_state[cfg.SELECTED_COLUMN_KEY] = selected_col_name
                        try: st.session_state[cfg.SELECTED_COLUMN_INDEX_KEY] = available_columns.index(selected_col_name)
                        except ValueError: st.session_state[cfg.SELECTED_COLUMN_INDEX_KEY] = 0
                        st.session_state[cfg.RESPONSES_KEY] = [] # Clear old responses

                    # Extract responses if column selected and responses not yet extracted
                    if selected_col_name and not st.session_state.get(cfg.RESPONSES_KEY):
                        try:
                            if selected_col_name in df_for_selection.columns:
                                responses_series = df_for_selection[selected_col_name].astype(str).fillna('')
                                responses_list_input = [r.strip() for r in responses_series if r and r.strip()]
                                st.caption(f"{len(responses_list_input)} valid response(s) extracted.")
                                st.session_state[cfg.RESPONSES_KEY] = responses_list_input # Store for analysis
                            else: st.error(f"Column '{selected_col_name}' not found."); st.session_state[cfg.RESPONSES_KEY] = []
                        except Exception as e: st.error(f"Error extracting data: {e}"); logging.exception("Analyzer: Error extracting column data."); st.session_state[cfg.RESPONSES_KEY] = []
                else: st.error("Uploaded file has no columns."); st.session_state.pop(cfg.UPLOADED_DF_KEY, None)

        else: # No file uploaded or removed
            if st.session_state.get(cfg.CURRENT_FILE_NAME_KEY):
                keys_to_clear = [cfg.UPLOADED_DF_KEY, cfg.SELECTED_COLUMN_KEY, cfg.SELECTED_COLUMN_INDEX_KEY, cfg.CURRENT_FILE_NAME_KEY, cfg.RESPONSES_KEY]
                for key in keys_to_clear: st.session_state.pop(key, None)
                logging.info("Analyzer: File removed, clearing related state.")


    # --- Analysis Button ---
    st.markdown("---")
    st.subheader("Run Analysis")

    # Read final values from state for button logic
    question = st.session_state.get(cfg.SURVEY_QUESTION_INPUT_KEY, '').strip()
    final_responses_list = st.session_state.get(cfg.RESPONSES_KEY, [])
    final_analysis_mode = "Batch" if len(final_responses_list) > 1 else "Single"

    disable_analyze = not question or not final_responses_list
    if disable_analyze:
        if not question: st.warning("⚠️ Please enter the Survey Question.")
        if not final_responses_list: st.warning("⚠️ Please provide Responses (paste or upload+select column).")

    if st.button("🚀 Start Analysis", type="primary", use_container_width=True, key="analyzer_start_button", disabled=disable_analyze):
        st.info(f"Starting {final_analysis_mode} analysis for {len(final_responses_list)} response(s)...")
        with st.spinner(f"🧠 AI is analyzing ({final_analysis_mode} Mode)... Please wait."):
            # Clear previous results
            keys_to_clear = [cfg.ANALYSIS_DF_KEY, cfg.CLUSTERING_DF_KEY, cfg.AI_QA_HISTORY_KEY, cfg.SELECTED_ROW_INDEX_KEY, cfg.EXECUTION_TIME_KEY]
            for key in keys_to_clear: st.session_state.pop(key, None)

            # Call the core AI evaluation function
            analysis_df_result = ai_core.evaluate_responses(
                survey_question=question,
                responses=final_responses_list,
                is_batch=(final_analysis_mode == "Batch"),
                batch_size=st.session_state[cfg.BATCH_SIZE_KEY],
                generation_config=analyzer_gen_config, # Pass the config object
                api_key=st.session_state[cfg.API_KEY]
            )

        # Store results in session state
        if analysis_df_result is not None and not analysis_df_result.empty:
            st.session_state[cfg.ANALYSIS_DF_KEY] = analysis_df_result
            st.session_state[cfg.RESPONSES_USED_KEY] = final_responses_list # Store actual list used
            st.session_state[cfg.SURVEY_QUESTION_USED_KEY] = question # Store question used
            st.session_state[cfg.ANALYSIS_MODE_KEY] = final_analysis_mode
            st.session_state[cfg.SELECTED_ROW_INDEX_KEY] = 0 # Reset selected row
            st.session_state[cfg.EXECUTION_TIME_KEY] = st.session_state.get('execution_time', 0) # Get time if set by func
            st.success(f"✅ Analysis Complete! View results in other tabs.")
            logging.info(f"Analyzer: Analysis successful ({len(final_responses_list)} responses).")
            st.toast("Analysis complete!", icon="🎉")
        else:
            st.error("❌ Analysis failed or returned no results. Check inputs/API key/logs.")
            logging.error("Analyzer: Analysis failed or returned empty/None dataframe.")
            st.session_state.pop(cfg.ANALYSIS_DF_KEY, None)


# ======================= Tab 2: Results & Details =====================
with results_tab:
    st.header("📈 Analysis Results & Details")
    analysis_df = st.session_state.get(cfg.ANALYSIS_DF_KEY)
    if analysis_df is None or analysis_df.empty:
        st.info("Run an analysis from the '📝 Input & Run' tab to see results here.")
    else:
        mode = st.session_state.get(cfg.ANALYSIS_MODE_KEY, 'Unknown')
        question = st.session_state.get(cfg.SURVEY_QUESTION_USED_KEY, '(Question not recorded)')
        exec_time = st.session_state.get(cfg.EXECUTION_TIME_KEY)

        st.subheader(f"Results for: \"{question}\"")
        if exec_time: st.caption(f"Analysis completed in {exec_time:.2f} seconds.")

        if mode == "Single":
            st.markdown("#### Single Response Analysis")
            ui.render_single_response_dashboard(analysis_df.iloc[0]) # Use UI component
        else: # Batch Mode
            st.markdown("---")
            st.subheader("Batch Analysis Summary")
            summary_cols = st.columns(4)
            with summary_cols[0]: ui.render_colored_metric("Avg. Overall", analysis_df['overall_score'].mean(), '/ 100', cfg.OVERALL_THRESHOLDS)
            with summary_cols[1]:
                on_topic_pct = (analysis_df['topic_status'].str.lower() == 'on-topic').mean() * 100 if analysis_df['topic_status'].notna().any() else 0
                ui.render_colored_metric("On-Topic %", on_topic_pct, '%', thresholds={}) # No specific threshold needed
            with summary_cols[2]: ui.render_colored_metric("Avg. Relevance", analysis_df['relevance'].mean(), '/ 10', cfg.METRIC_THRESHOLDS)
            with summary_cols[3]: ui.render_colored_metric("Avg. Bot Score", analysis_df['bot_likelihood'].mean(), '/ 10', thresholds={})

            st.markdown("---")
            st.subheader("Overall Score Distribution")
            dist_fig = ui.create_distribution_plot(analysis_df, 'overall_score') # Use UI component
            if dist_fig: st.pyplot(dist_fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Analyzed Responses Table")
            st.info("📋 Select a Row Number below for detailed analysis.")
            display_cols = [
                'full_response', 'manual_tags', 'action_item', 'overall_score', 'topic_status',
                'sentiment', 'relevance', 'completeness', 'specificity', 'language_quality',
                'sentiment_alignment', 'bot_likelihood', 'explanation', 'error'
            ]
            display_cols = [col for col in display_cols if col in analysis_df.columns]
            df_display = analysis_df[display_cols].copy()
            df_display.rename(columns={'full_response': 'Response', 'manual_tags': 'Tags', 'action_item': 'Action', 'overall_score': 'Score', 'topic_status': 'Topic', 'sentiment': 'Sentiment', 'language_quality': 'Language', 'sentiment_alignment': 'Sent. Align', 'bot_likelihood': 'Bot Score', 'error': 'Analysis Error'}, inplace=True)
            st.dataframe(df_display, use_container_width=True)

            st.markdown("---")
            st.subheader("🔍 View Detailed Analysis")
            max_row = len(analysis_df) - 1
            default_idx = st.session_state.get(cfg.SELECTED_ROW_INDEX_KEY, 0)
            sel_idx = st.number_input(f"Enter Row Number (0-{max_row}):", min_value=0, max_value=max_row, value=min(default_idx, max_row), step=1, key="analyzer_row_selector")
            if sel_idx is not None and 0 <= sel_idx <= max_row:
                st.session_state[cfg.SELECTED_ROW_INDEX_KEY] = sel_idx
                ui.render_single_response_dashboard(analysis_df.iloc[sel_idx]) # Use UI component
            else: st.warning("Invalid row number.")

            st.markdown("---")
            try: # Download button
                csv_download = analysis_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="💾 Download Full Results (CSV)", data=csv_download, file_name=f"analyzer_results_{time.strftime('%Y%m%d_%H%M')}.csv", mime='text/csv', use_container_width=True, key="analyzer_download_results")
            except Exception as e: st.error(f"Failed to create download file: {e}")

# ======================= Tab 3: Tagging & Actions =====================
with edit_tab:
    st.header("✍️ Manual Tagging & Action Items")
    analysis_df = st.session_state.get(cfg.ANALYSIS_DF_KEY)
    if analysis_df is None or analysis_df.empty:
        st.info("Run an analysis from the '📝 Input & Run' tab first.")
    else:
        st.info("📝 Edit manual tags (comma-separated) and action items. Click 'Save Changes' below the editor.")

        # Define editor configuration
        editor_column_config = {
            "manual_tags": st.column_config.TextColumn("Manual Tags (CSV)", width="medium", help="Enter tags separated by commas."),
            "action_item": st.column_config.TextColumn("Action Item", width="medium", help="Enter required action/follow-up."),
            "full_response": st.column_config.TextColumn("Response", width="large", disabled=True),
            "overall_score": st.column_config.NumberColumn("Score", disabled=True, format="%d"),
            "topic_status": st.column_config.TextColumn("Topic", disabled=True),
            "sentiment": st.column_config.TextColumn("Sentiment", disabled=True, help="Displays dict: {'label': '...', 'score': ...}"),
            "explanation": st.column_config.TextColumn("AI Explanation", disabled=True),
            "error": st.column_config.TextColumn("Analysis Error", disabled=True),
        }
        # Add other metrics as disabled columns
        for col in ['relevance', 'completeness', 'specificity', 'language_quality', 'sentiment_alignment', 'bot_likelihood']:
            if col in analysis_df.columns:
                editor_column_config[col] = st.column_config.NumberColumn(col.replace('_', ' ').title(), disabled=True, format="%.1f")

        # Define column order and disabled columns
        editor_column_order = ["manual_tags", "action_item", "full_response", "overall_score", "topic_status", "sentiment"] + \
                              [col for col in ['relevance', 'completeness', 'specificity', 'language_quality', 'sentiment_alignment', 'bot_likelihood'] if col in analysis_df.columns] + \
                              ["explanation", "error"]
        final_editor_column_order = [col for col in editor_column_order if col in analysis_df.columns]
        disabled_cols_editor = [col for col in final_editor_column_order if col not in ['manual_tags', 'action_item']]

        # Display the editor
        edited_df = st.data_editor(
            analysis_df, # Pass the current state DF
            key=cfg.MANUAL_EDITOR_WIDGET_KEY, use_container_width=True, hide_index=False,
            column_config=editor_column_config, column_order=final_editor_column_order,
            num_rows="fixed", disabled=disabled_cols_editor
        )

        # --- Save Changes Button Logic ---
        save_changes = False
        try: # Compare only editable columns for changes
            orig_editable = st.session_state[cfg.ANALYSIS_DF_KEY][['manual_tags', 'action_item']].fillna('')
            edited_editable = edited_df[['manual_tags', 'action_item']].fillna('')
            save_changes = not orig_editable.equals(edited_editable)
        except Exception as e: save_changes = True; logging.warning(f"Comparison failed: {e}")

        if save_changes: st.warning("You have unsaved changes.")
        if st.button("💾 Save Changes to Session", key="analyzer_save_edits", type="primary", disabled=not save_changes):
            st.session_state[cfg.ANALYSIS_DF_KEY] = edited_df.copy() # Update state
            st.toast("Changes saved to session!", icon="💾")
            logging.info("Analyzer: Manual edits saved.")
            time.sleep(0.5); st.rerun()
        elif not save_changes: st.success("✅ No unsaved changes.")


        # --- Cluster Tagging ---
        st.markdown("---"); st.subheader("🏷️ Tag Responses by Cluster")
        clustering_df = st.session_state.get(cfg.CLUSTERING_DF_KEY)
        if clustering_df is None or clustering_df.empty:
            st.info("Run Clustering in the '🔗 Clustering' tab to enable tagging by cluster.")
        else:
            try: # Merge cluster info with potentially edited analysis data
                df_for_cluster_tag = st.session_state[cfg.ANALYSIS_DF_KEY].copy()
                cluster_info = clustering_df[['response', 'Group']].rename(columns={'response': 'full_response'})
                if 'Group' in df_for_cluster_tag.columns: df_for_cluster_tag = df_for_cluster_tag.drop(columns=['Group']) # Avoid duplicate column
                df_with_clusters = pd.merge(df_for_cluster_tag.reset_index(), cluster_info, on='full_response', how='left').set_index('index')
                df_with_clusters.index.name = None
                df_with_clusters['Group'] = df_with_clusters['Group'].fillna('N/A')

                valid_groups = sorted([g for g in df_with_clusters['Group'].unique() if str(g).lower() not in ['unique', 'n/a', 'error', 'nan', '']])
                if not valid_groups: st.info("No cluster groups found for tagging.")
                else:
                    tc1, tc2, tc3 = st.columns([2, 2, 1])
                    with tc1: sel_cluster = st.selectbox("Select Cluster Group:", options=valid_groups, key="analyzer_cluster_select")
                    with tc2: tags_to_add_cluster = st.text_input("Tag(s) to Apply (CSV):", key="analyzer_cluster_tags", placeholder="e.g., UI Issue")
                    with tc3:
                        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        apply_btn = st.button("Apply to Cluster", key="analyzer_apply_cluster_tag")

                    if apply_btn and sel_cluster and tags_to_add_cluster:
                        indices_to_tag = df_with_clusters.index[df_with_clusters['Group'] == sel_cluster].tolist()
                        if indices_to_tag:
                            current_df_state = st.session_state[cfg.ANALYSIS_DF_KEY] # Get current DF from state
                            count = 0
                            for index in indices_to_tag:
                                if index in current_df_state.index:
                                    new_tags = utils.add_tags(current_df_state.loc[index, 'manual_tags'], tags_to_add_cluster)
                                    current_df_state.loc[index, 'manual_tags'] = new_tags
                                    count += 1
                            st.session_state[cfg.ANALYSIS_DF_KEY] = current_df_state # Save back to state
                            st.success(f"Applied tag(s) to {count} items in Cluster '{sel_cluster}'. Save changes if desired.")
                            st.toast(f"Cluster {sel_cluster} tagged!", icon="🏷️")
                            logging.info(f"Analyzer: Applied tags '{tags_to_add_cluster}' to cluster '{sel_cluster}' ({count} items).")
                            time.sleep(1); st.rerun()
                        else: st.warning(f"No items found for Cluster '{sel_cluster}'.")
                    elif apply_btn: st.warning("Select cluster AND enter tags.")
            except Exception as e: st.error(f"Error during cluster tagging setup: {e}"); logging.exception("Analyzer: Cluster tagging error.")

        # --- Download Edited Data ---
        st.markdown("---")
        try:
            csv_edited = st.session_state[cfg.ANALYSIS_DF_KEY].to_csv(index=False).encode('utf-8')
            st.download_button(label="💾 Download Current Data with Edits (CSV)", data=csv_edited, file_name=f"analyzer_results_edited_{time.strftime('%Y%m%d_%H%M')}.csv", mime='text/csv', use_container_width=True, key="analyzer_download_edited")
        except Exception as e: st.error(f"Failed to generate edited download: {e}")

# ======================= Tab 4: Aggregate Insights =====================
with insights_tab:
    st.header("💡 Aggregate Insights")
    analysis_df = st.session_state.get(cfg.ANALYSIS_DF_KEY)
    if analysis_df is None or analysis_df.empty:
        st.info("Run an analysis from the '📝 Input & Run' tab first.")
    else:
        responses_for_viz = st.session_state.get(cfg.RESPONSES_USED_KEY, analysis_df['full_response'].dropna().tolist())

        st.subheader("☁️ Word Cloud")
        if responses_for_viz:
             with st.spinner("Generating Word Cloud..."):
                 wc_fig = ui.create_word_cloud(responses_for_viz) # Use UI component
                 if wc_fig: st.pyplot(wc_fig, use_container_width=True)
                 # else: create_word_cloud handles messages
        else: st.info("No response text available.")

        st.markdown("---")
        st.subheader("📊 Metric Distributions")
        numeric_cols = analysis_df.select_dtypes(include=np.number).columns.tolist()
        potential_metrics = ['relevance', 'completeness', 'specificity', 'language_quality', 'sentiment_alignment', 'overall_score', 'bot_likelihood']
        metrics_to_plot = sorted([col for col in potential_metrics if col in numeric_cols and analysis_df[col].notna().any()])

        if not metrics_to_plot: st.warning("No numeric metrics found to plot.")
        else:
             default_idx = metrics_to_plot.index('overall_score') if 'overall_score' in metrics_to_plot else 0
             sel_metric = st.selectbox("Select Metric:", options=metrics_to_plot, index=default_idx, format_func=lambda x: x.replace('_', ' ').title(), key="analyzer_metric_selector")
             if sel_metric:
                 dist_fig = ui.create_distribution_plot(analysis_df, sel_metric) # Use UI component
                 if dist_fig: st.pyplot(dist_fig, use_container_width=True)
                 # else: create_distribution_plot handles messages

# ======================= Tab 5: Clustering =============================
with clustering_tab:
    st.header("🔗 Response Clustering")
    analysis_df = st.session_state.get(cfg.ANALYSIS_DF_KEY)
    is_batch = st.session_state.get(cfg.ANALYSIS_MODE_KEY, 'Single') == 'Batch'

    if analysis_df is None or analysis_df.empty: st.info("Run an analysis first.")
    elif not is_batch: st.info("Clustering requires batch analysis results (2+ responses).")
    else:
        st.markdown("Group similar responses based on text content using TF-IDF and hierarchical clustering.")
        responses_for_cluster = st.session_state.get(cfg.RESPONSES_USED_KEY, analysis_df['full_response'].tolist())

        if len(responses_for_cluster) < 2: st.warning("Need at least 2 responses to perform clustering.")
        else:
            st.session_state[cfg.SIMILARITY_THRESHOLD_KEY] = st.slider(
                "Similarity Threshold (%)", min_value=10, max_value=95,
                value=st.session_state.get(cfg.SIMILARITY_THRESHOLD_KEY, 70), step=5,
                key="analyzer_similarity_slider", help="Higher % = groups more similar (more groups)."
            )
            similarity_thresh = st.session_state[cfg.SIMILARITY_THRESHOLD_KEY]

            if st.button("Cluster Responses", key="analyzer_cluster_button", use_container_width=True, type="primary"):
                 with st.spinner("Clustering responses..."):
                     st.session_state.pop(cfg.CLUSTERING_DF_KEY, None) # Clear previous
                     logging.info(f"Analyzer: Starting clustering with threshold {similarity_thresh}%.")
                     cluster_df_result = processing.cluster_responses(responses_for_cluster, similarity_thresh) # Use processing module

                     if cluster_df_result is not None and not cluster_df_result.empty:
                         st.session_state[cfg.CLUSTERING_DF_KEY] = cluster_df_result
                         st.success("Clustering complete!"); st.toast("Clustering complete!", icon="🔗")
                         logging.info(f"Analyzer: Clustering successful ({len(cluster_df_result)} rows).")
                         time.sleep(0.5); st.rerun()
                     else: st.error("Clustering failed."); logging.error("Analyzer: Clustering returned None/empty."); st.session_state.pop(cfg.CLUSTERING_DF_KEY, None)

            # Display clustering results if available
            clustering_df = st.session_state.get(cfg.CLUSTERING_DF_KEY)
            if clustering_df is not None and not clustering_df.empty:
                st.markdown("---"); st.subheader("Clustering Results")
                try: # Calculate summary stats
                    valid_groups = clustering_df[~clustering_df['Group'].str.lower().isin(['unique', 'n/a', 'error', 'nan']) & (clustering_df['Group'].str.strip() != '')]['Group']
                    num_groups = valid_groups.nunique(); num_unique = (clustering_df['Group'].str.lower() == 'unique').sum()
                    num_errors = (clustering_df['Group'].str.lower() == 'error').sum(); num_na = clustering_df[clustering_df['Group'].str.lower().isin(['n/a', 'nan']) | (clustering_df['Group'].str.strip() == '')].shape[0]
                    num_clustered = len(clustering_df) - num_unique - num_errors - num_na
                except Exception as e: num_groups, num_unique, num_errors, num_na, num_clustered = 'Err', 'Err', 'Err', 'Err', 'Err'; logging.error(f"Cluster summary error: {e}")

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Clustered", num_clustered)
                with c2: st.metric("Clusters", num_groups, help="Groups with 2+ responses.")
                with c3: st.metric("Unique", num_unique, help="Not similar to others.")
                with c4: st.metric("Not Clustered", num_na + num_errors, help="N/A, errors, etc.")

                st.dataframe(clustering_df, use_container_width=True, hide_index=True,
                             column_config={ "response": st.column_config.TextColumn("Response", width="large"), "Group": st.column_config.TextColumn("Cluster Group", width="small"), "Similarity Score": st.column_config.NumberColumn("Similarity", format="%.3f", help="Max similarity within cluster.") })
                try: # Download button
                    csv_cluster = clustering_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="💾 Download Clustering Results (CSV)", data=csv_cluster, file_name=f"analyzer_clustering_{similarity_thresh}pct_{time.strftime('%Y%m%d_%H%M')}.csv", mime='text/csv', use_container_width=True, key="analyzer_download_cluster")
                except Exception as e: st.error(f"Failed cluster download: {e}")

# ======================= Tab 6: Ask AI =================================
# ======================= Tab 6: Ask AI =================================
with ai_qa_tab:
    st.header("❓ Ask AI About the Data")

    # --- MODIFIED CHECK for Prerequisites ---
    # Get the analysis dataframe safely
    analysis_df_for_qa = st.session_state.get(cfg.ANALYSIS_DF_KEY)
    # Check if it's a DataFrame and not empty
    analysis_data_exists = isinstance(analysis_df_for_qa, pd.DataFrame) and not analysis_df_for_qa.empty
    # Check if the responses used for that analysis are available
    responses_avail_for_qa = cfg.RESPONSES_USED_KEY in st.session_state and st.session_state.get(cfg.RESPONSES_USED_KEY)
    # --- END OF MODIFIED CHECK ---

    # Check if prerequisites are met
    if not analysis_data_exists or not responses_avail_for_qa:
        st.info("Run an analysis first (📝 Input & Run tab) to provide context for Q&A.")
    else:
        # Prerequisites met, proceed with the rest of the tab's logic
        question = st.session_state.get(cfg.SURVEY_QUESTION_USED_KEY, '(Question not recorded)')
        responses_context = st.session_state[cfg.RESPONSES_USED_KEY] # Get the list used for analysis

        # Ensure we actually have responses in the list retrieved from state
        if not responses_context:
             st.warning("Responses used for analysis were empty. Cannot provide context for Q&A.")
        else:
            st.markdown("Ask a question about the survey responses. AI uses the **original** responses and question provided during the analysis run.")
            st.info(f"Context: Question \"_{question}_\" and {len(responses_context)} original responses.")

            # Q&A Input Area
            # Check if the key exists from a previous run; if not, initialize to empty string
            if "analyzer_qa_input" not in st.session_state:
                st.session_state.analyzer_qa_input = ""

            user_q = st.text_area(
                "Your Question:",
                key="analyzer_qa_input", # Use the state key directly
                height=100,
                placeholder="e.g., What are the main recurring themes?\nList responses mentioning 'pricing'.\nSummarize negative feedback."
            )

            # Ask Button
            if st.button("Ask AI", key="analyzer_ask_ai_button", type="primary"):
                user_question_stripped = user_q.strip() # Read from widget state
                if not user_question_stripped:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("🧠 AI is thinking..."):
                        # API key should be valid if initialized check passed
                        api_key_qa = st.session_state.get(cfg.API_KEY)
                        if not api_key_qa:
                            st.error("API Key error. Please re-initialize.")
                            ai_answer = "Error: API Key missing."
                        else:
                            # Call the Q&A function from ai_core
                            ai_answer = ai_core.ask_ai_about_data(
                                survey_question=question,
                                responses_list=responses_context,
                                user_question=user_question_stripped,
                                generation_config=analyzer_gen_config, # Use the globally defined config
                                api_key=api_key_qa
                            )

                    # Add to history (most recent first) if valid answer received
                    if ai_answer is not None: # Check if function returned something
                         # Ensure history list exists
                         if not isinstance(st.session_state.get(cfg.AI_QA_HISTORY_KEY), list):
                              st.session_state[cfg.AI_QA_HISTORY_KEY] = []
                         # Insert new Q&A pair
                         st.session_state[cfg.AI_QA_HISTORY_KEY].insert(0, {"question": user_question_stripped, "answer": ai_answer})
                         logging.info(f"Analyzer Q&A: Q='{user_question_stripped[:50]}...', A_len={len(str(ai_answer))}")
                         # Clear the input box after successful submission by resetting its state variable
                         st.session_state.analyzer_qa_input = ""
                         st.rerun() # Rerun to display the new answer immediately and clear input box
                    else:
                         # Handle case where ask_ai_about_data might theoretically return None (though it aims to return error strings)
                         logging.error("Analyzer Q&A: ask_ai_about_data returned None unexpectedly.")


            # Display Q&A History
            st.markdown("---")
            st.subheader("Q&A History")
            qa_history_list = st.session_state.get(cfg.AI_QA_HISTORY_KEY, [])
            if qa_history_list:
                 # Show last N Q&As expanded by default
                 history_limit = 5
                 for i, qa_pair in enumerate(qa_history_list):
                     # Ensure keys exist and values are strings for display
                     q_text = str(qa_pair.get('question', '(Question missing)'))
                     a_text = str(qa_pair.get('answer', '(Answer missing)'))
                     with st.expander(f"Q: {q_text}", expanded=(i < history_limit)):
                         st.markdown("**AI Answer:**")
                         # Allow markdown formatting in the AI's answer
                         # Use unsafe_allow_html cautiously if AI might generate harmful HTML/scripts
                         st.markdown(a_text, unsafe_allow_html=True)
            else:
                 st.caption("No questions asked yet in this session.")
# ======================= Tab 7: Session Overview =========================
with history_tab:
    st.header("💾 Session Data Overview")
    st.markdown("Review current session data. Edits from 'Tagging & Actions' are reflected below.")

    st.subheader("Latest Analysis Results (incl. Manual Edits)")
    analysis_df = st.session_state.get(cfg.ANALYSIS_DF_KEY)
    if analysis_df is not None and not analysis_df.empty:
        st.dataframe(analysis_df.head(), use_container_width=True)
        st.caption(f"Showing first 5 of {len(analysis_df)} analyzed responses. Download full data from 'Results' or 'Tagging' tabs.")
    else: st.info("No analysis data.")

    st.markdown("---"); st.subheader("Latest Clustering Results")
    clustering_df = st.session_state.get(cfg.CLUSTERING_DF_KEY)
    if clustering_df is not None and not clustering_df.empty:
        st.dataframe(clustering_df.head(), use_container_width=True)
        st.caption(f"Showing first 5 of {len(clustering_df)} responses. Download full data from 'Clustering' tab.")
    else: st.info("No clustering data.")

    st.markdown("---"); st.subheader("Latest AI Q&A")
    qa_history = st.session_state.get(cfg.AI_QA_HISTORY_KEY)
    if qa_history:
         latest = qa_history[0]
         st.markdown(f"**Last Question:**\n> _{latest.get('question', '...')}_")
         st.markdown("**Last AI Answer:**")
         st.markdown(latest.get('answer', '...'), unsafe_allow_html=True)
    else: st.info("No questions asked via 'Ask AI' yet.")

# --- Footer ---
st.markdown(f"""<div class="footer"><p>© {time.strftime('%Y')} Phronesis Partners. All rights reserved.</p></div>""", unsafe_allow_html=True)

# --- End of App ---
logging.info("Reached end of analyzer_main_app.py execution.")