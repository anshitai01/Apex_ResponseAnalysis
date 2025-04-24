# analyzer_styling.py
"""
Contains the CSS styles for the AI Response Analyzer application,
using the Phronesis Apex theme. Includes sidebar logo styles.
"""
import streamlit as st # Required for st.markdown

# Import necessary config variables FROM analyzer_config
from analyzer_config import (
    PRIMARY_ACCENT_COLOR, PRIMARY_ACCENT_COLOR_RGB, CARD_TEXT_COLOR,
    CARD_TITLE_TEXT_COLOR, MAIN_TITLE_COLOR, BODY_TEXT_COLOR,
    SUBTITLE_COLOR, MAIN_BACKGROUND_COLOR, CARD_BACKGROUND_COLOR,
    HOVER_GLOW_COLOR, CONTAINER_BG_COLOR, CONTAINER_BORDER_RADIUS,
    TITLE_FONT, BODY_FONT, CARD_TITLE_FONT, SIDEBAR_BACKGROUND_COLOR,
    INPUT_BG_COLOR, INPUT_BORDER_COLOR, INPUT_TEXT_COLOR,
    BUTTON_PRIMARY_BG, BUTTON_PRIMARY_TEXT, BUTTON_SECONDARY_BG,
    BUTTON_SECONDARY_TEXT, BUTTON_SECONDARY_BORDER,
    DATAFRAME_HEADER_BG, DATAFRAME_HEADER_TEXT, DATAFRAME_CELL_BG,
    DATAFRAME_CELL_TEXT,
    # Chart/Alert colors from config
    CHART_SUCCESS_COLOR, CHART_WARNING_COLOR, CHART_ERROR_COLOR
)

# --- Custom CSS Injection ---
APP_STYLE = f"""
<style>
    /* --- Import Fonts --- */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400&display=swap');

    /* --- Global Body & Streamlit App Styling --- */
    body {{
        background-color: {MAIN_BACKGROUND_COLOR};
        color: {BODY_TEXT_COLOR};
        font-family: {BODY_FONT};
    }}
    .stApp {{
        background-color: {MAIN_BACKGROUND_COLOR};
        color: {BODY_TEXT_COLOR};
    }}
    .stApp > header {{
        background-color: transparent;
        border-bottom: none;
    }}

    /* --- Main Content Area Container --- */
    .main .block-container {{
        max-width: 1200px;
        padding: 1.5rem 2rem 4rem 2rem;
        background-color: {CONTAINER_BG_COLOR};
        border-radius: {CONTAINER_BORDER_RADIUS};
        color: {BODY_TEXT_COLOR};
        margin: auto;
        font-family: {BODY_FONT};
    }}

    /* --- Headings --- */
     h1 {{
        font-family: {TITLE_FONT}; font-size: 2.2rem; font-weight: 700;
        color: {MAIN_TITLE_COLOR}; letter-spacing: 1px; text-align: center;
        margin-bottom: 1.5rem; text-shadow: 0 0 8px rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.3);
    }}
    h2, h3 {{
        font-family: {TITLE_FONT}; color: {PRIMARY_ACCENT_COLOR}; margin-top: 2.5rem;
        margin-bottom: 1.5rem; border-bottom: 1px solid rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.3);
        padding-bottom: 0.6rem; font-weight: 600; font-size: 1.7rem;
    }}
    h4 {{
        font-family: {CARD_TITLE_FONT}; color: {MAIN_TITLE_COLOR}; font-weight: 600;
        margin-top: 2rem; margin-bottom: 1rem; font-size: 1.3rem;
    }}
    h6 {{
        color: {SUBTITLE_COLOR}; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem;
        font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px;
        border-bottom: 1px solid {INPUT_BORDER_COLOR}; padding-bottom: 0.2rem;
    }}

    /* --- Sidebar Styling --- */
    [data-testid=stSidebar] {{
        background-color: {SIDEBAR_BACKGROUND_COLOR} !important;
        padding: 1.5rem 1.5rem;
        border-right: 1px solid rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.2);
    }}

    /* --- Sidebar Logo Styling --- */
    .sidebar-logo {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 1.5rem; /* Space below logo */
        height: 120px; /* <-- ADJUST SIZE HERE */
        width: auto; /* Keeps aspect ratio */
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
    }}
    .sidebar-logo-placeholder {{
        height: 120px; /* <-- Match logo height */
        width: 120px; /* <-- Match logo height (for square placeholder) */
        background-color: #333;
        border: 1px dashed #555;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #888;
        font-size: 0.9em;
        text-align: center;
        border-radius: 5px;
        margin: 0 auto 1.5rem auto; /* Centering and space below */
    }}
    /* --- End Sidebar Logo Styling --- */


    /* Keep the styling for elements INSIDE the sidebar */
    [data-testid=stSidebar] h2, [data-testid=stSidebar] h3 {{
        font-size: 1.3em; color: {PRIMARY_ACCENT_COLOR}; text-align: left;
        margin-top: 1rem; margin-bottom: 1rem; font-weight: 600; border-bottom: none;
    }}
    [data-testid=stSidebar] p, [data-testid=stSidebar] label, [data-testid=stSidebar] .caption {{
        color: {BODY_TEXT_COLOR} !important;
    }}
    [data-testid=stSidebar] .stMarkdown {{
        color: {BODY_TEXT_COLOR} !important;
    }}
    [data-testid=stSidebar] .stExpander {{
        border: 1px solid {INPUT_BORDER_COLOR}; border-radius: 8px;
        background-color: rgba(0,0,0, 0.1);
    }}
    [data-testid=stSidebar] .stExpander header {{
        color: {MAIN_TITLE_COLOR}; font-weight: 600;
    }}
    [data-testid=stSidebar] div[data-testid="stSlider"] label {{
         color: {BODY_TEXT_COLOR} !important; font-size: 0.95em;
    }}
     [data-testid=stSidebar] div[data-testid="stSlider"] div[data-baseweb="slider"] {{
         color: {BODY_TEXT_COLOR};
    }}

    /* --- Tab Styling --- */
    div[data-testid="stTabs"] {{
        border-bottom: 1px solid {INPUT_BORDER_COLOR}; margin-bottom: 1.5rem;
    }}
    div[data-testid="stTabs"] button[role="tab"] {{
        font-family: {CARD_TITLE_FONT}; border-radius: 8px 8px 0 0; padding: 0.7rem 1.4rem;
        font-weight: 600; font-size: 0.95rem; color: {BODY_TEXT_COLOR}; background-color: transparent;
        border: none; border-bottom: 3px solid transparent; transition: all 0.3s ease; opacity: 0.7;
    }}
    div[data-testid="stTabs"] button[role="tab"]:hover {{
        background-color: rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.1); color: {PRIMARY_ACCENT_COLOR}; opacity: 1;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
         background-color: transparent; color: {PRIMARY_ACCENT_COLOR};
         border-bottom: 3px solid {PRIMARY_ACCENT_COLOR}; opacity: 1;
    }}

    /* --- Button Styling --- */
    div[data-testid="stButton"] > button {{
        border-radius: 20px; padding: 0.6rem 1.6rem; font-weight: 600; font-family: {BODY_FONT};
        transition: all 0.3s ease; border: 1px solid {BUTTON_SECONDARY_BORDER};
        background-color: {BUTTON_SECONDARY_BG}; color: {BUTTON_SECONDARY_TEXT};
    }}
    div[data-testid="stButton"] > button:hover {{
        transform: translateY(-2px); box-shadow: 0 4px 10px rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.2);
        border-color: {PRIMARY_ACCENT_COLOR}; color: {PRIMARY_ACCENT_COLOR};
        background-color: rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.05);
    }}
    div[data-testid="stButton"] > button:active {{ transform: translateY(0px); box-shadow: none; }}
    div[data-testid="stButton"] > button[kind="primary"] {{
         background-color: {BUTTON_PRIMARY_BG}; color: {BUTTON_PRIMARY_TEXT}; border-color: {BUTTON_PRIMARY_BG};
    }}
    div[data-testid="stButton"] > button[kind="primary"]:hover {{
         background-color: {PRIMARY_ACCENT_COLOR}; color: {BUTTON_PRIMARY_TEXT}; border-color: {PRIMARY_ACCENT_COLOR};
         box-shadow: 0 6px 15px rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.3); opacity: 0.9;
    }}
     div[data-testid="stButton"] > button:disabled {{
        background-color: rgba(255, 255, 255, 0.1) !important; color: rgba(255, 255, 255, 0.4) !important;
        border-color: rgba(255, 255, 255, 0.2) !important; cursor: not-allowed;
    }}
    div[data-testid="stButton"] > button:disabled:hover {{
         box-shadow: none; transform: none; background-color: rgba(255, 255, 255, 0.1) !important;
    }}

    /* --- Input Element Styling --- */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background-color: {INPUT_BG_COLOR} !important; color: {INPUT_TEXT_COLOR} !important;
        border: 1px solid {INPUT_BORDER_COLOR} !important; border-radius: 8px !important;
        box-shadow: none !important;
    }}
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stSelectbox"] label {{
        color: {BODY_TEXT_COLOR} !important; font-weight: 600; font-family: {BODY_FONT}; margin-bottom: 0.5rem;
    }}
    div[data-baseweb="popover"] ul[role="listbox"] {{
         background-color: {CARD_BACKGROUND_COLOR}; border: 1px solid {INPUT_BORDER_COLOR};
    }}
     div[data-baseweb="popover"] ul[role="listbox"] li {{ color: {INPUT_TEXT_COLOR}; }}
    div[data-baseweb="popover"] ul[role="listbox"] li:hover {{ background-color: rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.2); }}

    /* --- Radio Button Label Text Color --- */
    div[data-testid="stRadio"] label p {{ color: #FFFFFF !important; }}
    div[data-testid="stRadio"] label span {{ border-color: {INPUT_BORDER_COLOR} !important; }}
     div[data-testid="stRadio"] label input:checked + div div {{ background-color: {PRIMARY_ACCENT_COLOR} !important; }}


    /* --- Data Editor / DataFrame Styling --- */
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{
        border: 1px solid {INPUT_BORDER_COLOR}; border-radius: 8px; background-color: {DATAFRAME_CELL_BG};
    }}
    .stDataFrame th, .stDataEditor th {{
        background-color: {DATAFRAME_HEADER_BG} !important; color: {DATAFRAME_HEADER_TEXT} !important;
        font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.5px;
        border-radius: 0 !important; border-bottom: 2px solid {PRIMARY_ACCENT_COLOR} !important;
    }}
    .stDataFrame td, .stDataEditor td {{
        font-size: 0.9rem; vertical-align: middle; padding: 0.6rem 0.7rem; color: {DATAFRAME_CELL_TEXT};
        border-bottom: 1px solid {INPUT_BORDER_COLOR}; border-right: 1px solid {INPUT_BORDER_COLOR};
    }}
    .stDataEditor td input, .stDataEditor td div[data-baseweb="select"] > div {{
         background-color: {INPUT_BG_COLOR} !important; color: {INPUT_TEXT_COLOR} !important;
         border: 1px solid {PRIMARY_ACCENT_COLOR} !important;
    }}

    /* --- Markdown & Misc Elements --- */
    .stMarkdown p {{ color: {BODY_TEXT_COLOR}; line-height: 1.6; }}
    .stMarkdown a {{ color: {PRIMARY_ACCENT_COLOR}; text-decoration: none; }}
    .stMarkdown a:hover {{ text-decoration: underline; }}
    .stMarkdown blockquote {{
        margin-left: 0; padding: 0.8rem 1.2rem; border-left: 4px solid {PRIMARY_ACCENT_COLOR};
        font-style: italic; color: {CARD_TEXT_COLOR}; background-color: {CARD_BACKGROUND_COLOR};
        border-radius: 0 8px 8px 0;
    }}
    .stCaption {{ color: {SUBTITLE_COLOR}; font-size: 0.85rem; }}

    /* --- Alert Styling --- */
    div[data-testid="stAlert"] {{
        border-radius: 8px !important; border: 1px solid {INPUT_BORDER_COLOR} !important;
        border-left-width: 5px !important; padding: 1rem 1.2rem !important;
    }}
    div[data-testid="stAlert"] div[role="alert"] {{ font-family: {BODY_FONT}; font-size: 0.95rem; }}
    div[data-testid="stAlert"][data-baseweb="notification-info"] {{
        border-left-color: {PRIMARY_ACCENT_COLOR} !important; background-color: rgba({PRIMARY_ACCENT_COLOR_RGB}, 0.1) !important;
    }}
    div[data-testid="stAlert"][data-baseweb="notification-info"] div[role="alert"] {{ color: {PRIMARY_ACCENT_COLOR} !important; font-weight: 500; }}
    div[data-testid="stAlert"][data-baseweb="notification-info"] svg {{ fill: {PRIMARY_ACCENT_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-warning"] {{
        border-left-color: {CHART_WARNING_COLOR} !important; background-color: rgba(243, 156, 18, 0.1) !important;
    }}
     div[data-testid="stAlert"][data-baseweb="notification-warning"] div[role="alert"] {{ color: {CHART_WARNING_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-warning"] svg {{ fill: {CHART_WARNING_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-error"] {{
        border-left-color: {CHART_ERROR_COLOR} !important; background-color: rgba(231, 76, 60, 0.1) !important;
    }}
     div[data-testid="stAlert"][data-baseweb="notification-error"] div[role="alert"] {{ color: {CHART_ERROR_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-error"] svg {{ fill: {CHART_ERROR_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-success"] {{
        border-left-color: {CHART_SUCCESS_COLOR} !important; background-color: rgba(46, 204, 113, 0.1) !important;
    }}
     div[data-testid="stAlert"][data-baseweb="notification-success"] div[role="alert"] {{ color: {CHART_SUCCESS_COLOR} !important; }}
    div[data-testid="stAlert"][data-baseweb="notification-success"] svg {{ fill: {CHART_SUCCESS_COLOR} !important; }}

    /* --- Response Detail Container Specific Styling --- */
    .response-detail-container {{
        background-color: {CARD_BACKGROUND_COLOR}; border: 1px solid {INPUT_BORDER_COLOR};
        border-radius: 10px; padding: 1.5rem 2rem; margin-top: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    .response-detail-container h6 {{
        color: {PRIMARY_ACCENT_COLOR}; border-bottom: none; margin-top: 1.2rem;
    }}

    /* Custom Metric Card Styling */
    .custom-metric {{
        background-color: {INPUT_BG_COLOR}; border: 1px solid {INPUT_BORDER_COLOR}; border-radius: 6px;
        padding: 0.6rem 0.8rem; text-align: center; margin-bottom: 0.5rem; box-shadow: none;
        height: 85px; display: flex; flex-direction: column; justify-content: center;
    }}
    .custom-metric-label {{
        font-size: 0.8rem; color: {SUBTITLE_COLOR}; display: block; margin-bottom: 0.3rem;
        font-weight: 500; line-height: 1.2;
    }}
    .custom-metric-value {{
        font-size: 1.4rem; font-weight: 600; line-height: 1; display: flex; justify-content: center; align-items: baseline;
    }}
    .custom-metric-value .unit {{
         font-size: 0.8rem; font-weight: 400; margin-left: 0.2rem; color: {SUBTITLE_COLOR};
         align-self: flex-end; padding-bottom: 0.1rem;
    }}

    /* Overall Score Card Styling */
    .overall-score-container {{
        border-radius: 8px; padding: 0.8rem 1rem; text-align: center; color: {BUTTON_PRIMARY_TEXT};
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 85px; display: flex; flex-direction: column; justify-content: center;
    }}
    .overall-score-title {{
        font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.9; margin-bottom: 0.1rem;
    }}
    .overall-score-value {{ font-size: 2.2rem; font-weight: 700; line-height: 1; }}
    .overall-score-label {{ font-size: 0.9rem; font-weight: 400; opacity: 0.8; margin-left: 0.2rem; }}

    /* --- Footer Styling --- */
    .footer {{
        text-align: center; color: {SUBTITLE_COLOR}; opacity: 0.7; margin: 4rem auto 1rem auto;
        font-size: 0.9rem; max-width: 1100px; padding-bottom: 1rem;
    }}

    /* --- Streamlit Cleanup --- */
    header[data-testid="stHeader"], footer {{ display: none !important; }}
    div[data-testid="stDecoration"] {{ display: none !important; }}

    /* --- Responsive Adjustments --- */
    @media (max-width: 768px) {{
        .main .block-container {{ padding: 1.5rem 1rem 3rem 1rem; }}
         h1 {{ font-size: 2rem; }} h2, h3 {{ font-size: 1.5rem; }} h4 {{ font-size: 1.1rem; }}
         div[data-testid="stTabs"] button[role="tab"] {{ padding: 0.6rem 1rem; font-size: 0.9rem; }}
         div[data-testid="stButton"] > button {{ padding: 0.5rem 1.2rem; }}
         .custom-metric {{ height: 75px; }}
         .overall-score-container {{ height: 75px; }}
         .overall-score-value {{ font-size: 1.8rem; }}
    }}

</style>
"""

def apply_styling():
    """Injects the custom CSS into the Streamlit app."""
    st.markdown(APP_STYLE, unsafe_allow_html=True)