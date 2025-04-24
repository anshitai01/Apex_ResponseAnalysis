# analyzer_config.py
"""
Stores configuration constants for the AI Response Analyzer application.
Uses the Phronesis Apex theme colors and fonts.
Defines specific state keys prefixed with 'analyzer_'.
"""

from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Theme Configuration (Phronesis Apex Theme) ---
# Copied from the previous (Themer) app's final config
PRIMARY_ACCENT_COLOR = "#cd669b"
PRIMARY_ACCENT_COLOR_RGB = "205, 102, 155"
CARD_TEXT_COLOR = "#a9b4d2"
CARD_TITLE_TEXT_COLOR = PRIMARY_ACCENT_COLOR
MAIN_TITLE_COLOR = "#ffff"
BODY_TEXT_COLOR = "#ffff"
SUBTITLE_COLOR = "#8b98b8"
MAIN_BACKGROUND_COLOR = "#0b132b"
CARD_BACKGROUND_COLOR = "#1c2541"
SIDEBAR_BACKGROUND_COLOR = "#121a35"
HOVER_GLOW_COLOR = f"rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.4)"
CONTAINER_BG_COLOR = "rgba(11, 19, 43, 0.0)"
CONTAINER_BORDER_RADIUS = "15px"
INPUT_BG_COLOR = "#1c2541"
INPUT_BORDER_COLOR = "#3a506b"
INPUT_TEXT_COLOR = BODY_TEXT_COLOR
BUTTON_PRIMARY_BG = PRIMARY_ACCENT_COLOR
BUTTON_PRIMARY_TEXT = "#FFFFFF"
BUTTON_SECONDARY_BG = "transparent"
BUTTON_SECONDARY_TEXT = PRIMARY_ACCENT_COLOR
BUTTON_SECONDARY_BORDER = PRIMARY_ACCENT_COLOR
DATAFRAME_HEADER_BG = "#1c2541"
DATAFRAME_HEADER_TEXT = MAIN_TITLE_COLOR
DATAFRAME_CELL_BG = MAIN_BACKGROUND_COLOR
DATAFRAME_CELL_TEXT = BODY_TEXT_COLOR
CHART_SUCCESS_COLOR = "#2ecc71"
CHART_WARNING_COLOR = "#f39c12"
CHART_ERROR_COLOR = "#e74c3c"

# --- Font Families ---
TITLE_FONT = "'Montserrat', sans-serif"
BODY_FONT = "'Roboto', sans-serif"
CARD_TITLE_FONT = "'Montserrat', sans-serif"

# --- App-Specific Thresholds ---
# Thresholds for 0-100 scale (e.g., overall score)
OVERALL_THRESHOLDS = {'error': (0, 40), 'orange': (40, 60), 'warning': (60, 80), 'success': (80, 101)}
# Thresholds for 0-10 scale (e.g., individual metrics)
METRIC_THRESHOLDS = {'error': (0, 4), 'orange': (4, 6), 'warning': (6, 8), 'success': (8, 11)}
# Thresholds for Sentiment Score (0-1 scale) - Adjust as needed
SENTIMENT_THRESHOLDS = {'error': (0, 0.4), 'warning': (0.4, 0.6), 'success': (0.6, 1.1)}

# Safety Settings (Copied from previous app)
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, # Keep relaxed for analysis?
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- App State Keys (Prefixed with 'analyzer_') ---
APP_PREFIX = "analyzer_"

# Initialization & API
INITIALIZED_KEY = f"{APP_PREFIX}initialized"
API_KEY = f"{APP_PREFIX}api_key"
API_KEY_SOURCE_KEY = f"{APP_PREFIX}api_key_source"

# Input Data
SURVEY_QUESTION_INPUT_KEY = f"{APP_PREFIX}survey_question_input"
RESPONSES_INPUT_AREA_KEY = f"{APP_PREFIX}responses_input_area"
UPLOADED_DF_KEY = f"{APP_PREFIX}uploaded_df"
SELECTED_COLUMN_KEY = f"{APP_PREFIX}selected_column"
SELECTED_COLUMN_INDEX_KEY = f"{APP_PREFIX}selected_column_index"
CURRENT_FILE_NAME_KEY = f"{APP_PREFIX}current_file_name"
INPUT_METHOD_KEY = f"{APP_PREFIX}input_method"
RESPONSES_KEY = f"{APP_PREFIX}responses" # Holds the final list of responses to analyze

# Analysis Results & State
ANALYSIS_DF_KEY = f"{APP_PREFIX}analysis_df"
CLUSTERING_DF_KEY = f"{APP_PREFIX}clustering_df"
AI_QA_HISTORY_KEY = f"{APP_PREFIX}ai_qa_history"
SELECTED_ROW_INDEX_KEY = f"{APP_PREFIX}selected_row_index"
ANALYSIS_MODE_KEY = f"{APP_PREFIX}analysis_mode" # Stores 'Single' or 'Batch'
RESPONSES_USED_KEY = f"{APP_PREFIX}responses_used_in_analysis" # Store actual list used
SURVEY_QUESTION_USED_KEY = f"{APP_PREFIX}survey_question_used_in_analysis" # Store question used
EXECUTION_TIME_KEY = f"{APP_PREFIX}execution_time"
SIMILARITY_THRESHOLD_KEY = f"{APP_PREFIX}similarity_threshold_pct"

# Settings
BATCH_SIZE_KEY = f"{APP_PREFIX}batch_size"
GEN_TEMP_KEY = f"{APP_PREFIX}gen_temp"
GEN_TOP_K_KEY = f"{APP_PREFIX}gen_top_k"
GEN_TOP_P_KEY = f"{APP_PREFIX}gen_top_p"
GEN_MAX_TOKENS_KEY = f"{APP_PREFIX}gen_max_tokens"

# Widget Keys (If needed for specific state control)
RESET_BUTTON_KEY = f"{APP_PREFIX}reset_all_button"
INIT_BUTTON_KEY = f"{APP_PREFIX}init_button_manual"
MANUAL_EDITOR_WIDGET_KEY = f"{APP_PREFIX}manual_editor_widget"
# Add other specific widget keys if required

# --- Logo Path (Relative to main_app.py location) ---
# Assuming the same logo file is used
LOGO_FILENAME = "apexlogo.png"