# analyzer_ui_components.py
"""
Functions for creating and displaying UI components in the
AI Response Analyzer application, adapted for the Phronesis Apex theme.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import textwrap # For text wrapping in plots
import logging

# --- CORRECTED IMPORT SECTION ---
# Import config and utils from analyzer modules
from analyzer_config import (
    OVERALL_THRESHOLDS, METRIC_THRESHOLDS, SENTIMENT_THRESHOLDS,
    MAIN_BACKGROUND_COLOR, BODY_TEXT_COLOR, SUBTITLE_COLOR,
    PRIMARY_ACCENT_COLOR, CARD_BACKGROUND_COLOR, INPUT_BORDER_COLOR,
    MAIN_TITLE_COLOR, # <--- Added
    CHART_SUCCESS_COLOR, # <--- Added
    CHART_WARNING_COLOR, # <--- Added
    CHART_ERROR_COLOR # <--- Added
)
from analyzer_utils import get_color_for_score # For metric colors
# --- END OF CORRECTED IMPORT SECTION ---


# --- Visualization Functions ---

def create_word_cloud(responses):
    """
    Generates a matplotlib figure containing a word cloud from text responses.
    Adapted for dark theme (transparent background).
    """
    # ... (rest of function code remains the same) ...
    if not responses:
        logging.info("Word cloud generation skipped: No responses provided.")
        return None
    try:
        text_list = [str(r).strip() for r in responses if r and isinstance(r, (str, int, float)) and str(r).strip()]
        if not text_list:
            logging.info("Word cloud generation skipped: No valid text content after cleaning.")
            return None
        text = ' '.join(text_list)
        if not text.strip():
            logging.info("Word cloud generation skipped: Joined text is empty.")
            return None

        wordcloud = WordCloud(
            width=800, height=350,
            background_color=None, # Use None for transparent background
            mode="RGBA",           # Ensure RGBA mode for transparency
            colormap='plasma',     # Choose a colormap suitable for dark backgrounds
            max_words=100,
            random_state=42
            ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0.0) # Make figure background transparent
        ax.patch.set_alpha(0.0)  # Make axes background transparent
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        logging.info("Word cloud figure generated successfully (dark theme).")
        return fig

    except ValueError as ve:
         if "empty vocabulary" in str(ve).lower():
             logging.warning(f"Word cloud generation failed: Empty vocabulary. {ve}")
             st.caption("Could not generate word cloud: No significant words found.")
         else:
             st.error(f"Error generating word cloud (ValueError): {ve}")
             logging.error(f"Word cloud ValueError: {ve}")
         return None
    except ImportError as ie:
        st.error(f"Error generating word cloud: Missing library ({ie}).")
        logging.error(f"Word cloud ImportError: {ie}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred generating the word cloud: {e}")
        logging.exception("Word cloud generation failed.")
        return None


def create_distribution_plot(df, metric):
    """
    Create a histogram/KDE plot for a specified metric using Seaborn/Matplotlib.
    Adapted for dark theme.
    """
    # ... (rest of function code remains the same) ...
    if metric not in df.columns or pd.api.types.is_numeric_dtype(df[metric]) == False:
        logging.warning(f"Metric '{metric}' not found or not numeric for distribution plot.")
        return None
    metric_data = df[metric].dropna()
    if metric_data.empty:
        st.info(f"Metric '{metric.replace('_', ' ').title()}' has no valid data for distribution plot.")
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        sns.histplot(metric_data, kde=True, ax=ax, color=PRIMARY_ACCENT_COLOR, bins=10,
                     edgecolor=BODY_TEXT_COLOR, line_kws={'linewidth': 1.5})

        title = f"{metric.replace('_', ' ').title()} Distribution"
        # This line now has MAIN_TITLE_COLOR available
        ax.set_title(title, fontsize=14, color=MAIN_TITLE_COLOR, weight='bold')
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11, color=BODY_TEXT_COLOR)
        ax.set_ylabel("Frequency", fontsize=11, color=BODY_TEXT_COLOR)

        ax.tick_params(axis='both', which='major', labelsize=9, colors=BODY_TEXT_COLOR)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color=INPUT_BORDER_COLOR, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(BODY_TEXT_COLOR)
        ax.spines['bottom'].set_color(BODY_TEXT_COLOR)

        plt.tight_layout()
        logging.info(f"Distribution plot created for metric '{metric}'.")
        return fig

    except Exception as e:
         st.error(f"Error generating distribution plot for '{metric}': {e}")
         logging.exception(f"Failed to generate distribution plot for {metric}.")
         return None


def create_radar_chart(values, categories):
    """
    Generate a radar chart for given values and categories using Matplotlib.
    Adapted for dark theme.
    """
    # ... (rest of function code remains the same) ...
    N = len(categories)
    if N == 0:
        logging.warning("Cannot create radar chart: No categories provided.")
        return None
    if len(values) != N:
        logging.warning(f"Radar chart mismatch: {len(values)} values, {N} categories.")
        return None

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    values_num = []
    for v in values:
        try:
            val = float(v)
            if pd.isna(val): val = 0
            else: val = max(0, min(10, val))
        except (ValueError, TypeError): val = 0
        values_num.append(val)
    values_num += values_num[:1]

    try:
        fig = plt.figure(figsize=(5, 5))
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111, polar=True, facecolor=CARD_BACKGROUND_COLOR)
        ax.patch.set_alpha(0.5)

        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)

        ax.plot(angles, values_num, linewidth=2, linestyle='solid', color=PRIMARY_ACCENT_COLOR, marker='o', markersize=5)
        ax.fill(angles, values_num, color=PRIMARY_ACCENT_COLOR, alpha=0.35)

        ax.set_xticks(angles[:-1])
        wrapped_labels = [textwrap.fill(cat.replace('_', ' ').title(), width=12) for cat in categories]
        ax.set_xticklabels(wrapped_labels, size=9, color=BODY_TEXT_COLOR, weight='medium')

        ax.set_yticks(np.arange(0, 11, 2))
        ax.set_yticklabels([str(i) for i in np.arange(0, 11, 2)], color=SUBTITLE_COLOR, size=8)
        ax.set_ylim(0, 10)

        ax.grid(color=INPUT_BORDER_COLOR, linestyle='--', linewidth=0.6, alpha=0.7)
        ax.spines['polar'].set_visible(False)

        plt.tight_layout()
        logging.info("Radar chart generated successfully.")
        return fig

    except Exception as e:
         st.error(f"Error generating radar chart: {e}")
         logging.exception("Failed to generate radar chart.")
         return None


# --- Render Custom Colored Metric Function ---
def render_colored_metric(label, value, unit='/ 10', thresholds=METRIC_THRESHOLDS):
    """Renders a metric display with color-coded text based on thresholds."""
    # ... (rest of function code remains the same) ...
    try:
        score_value = float(value)
        if pd.isna(score_value):
            display_value = "?"
            unit = ""
        elif score_value == int(score_value):
            display_value = f"{score_value:.0f}"
        else:
            display_value = f"{score_value:.1f}"
    except (ValueError, TypeError):
        display_value = str(value) if pd.notna(value) and str(value).strip() != "" else "?"
        unit = ""
        score_value = -1

    color = get_color_for_score(score_value, thresholds)

    st.markdown(f"""
    <div class="custom-metric">
        <span class="custom-metric-label">{label}</span>
        <span class="custom-metric-value" style="color: {color};">
            {display_value}<span class="unit">{unit}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


# --- Single Response Dashboard Rendering Function ---
# analyzer_ui_components.py
# ... (keep imports and other functions like create_word_cloud, create_distribution_plot, etc.) ...

# --- Single Response Dashboard Rendering Function ---
# UPDATED for new layout
def render_single_response_dashboard(response_data: pd.Series):
    """
    Render a detailed dashboard for a single response using Streamlit columns
    and custom metric rendering.
    Layout: Metrics & Sentiment on left, Radar on right.

    Args:
        response_data (pd.Series): A row from the analysis DataFrame.
    """
    if response_data is None or not isinstance(response_data, pd.Series) or response_data.empty:
        st.warning("No response data selected or available to display details.")
        return

    # Use the CSS class for the main container
    st.markdown('<div class="response-detail-container">', unsafe_allow_html=True)

    full_response = response_data.get('full_response', 'N/A')
    st.markdown(f"#### Detailed Analysis for Response (Row {response_data.name})")
    st.markdown(f"> {full_response}")
    st.markdown("---")

    # --- Row 1: Score, Status, Bot, Error/Explanation (No Change) ---
    col_score, col_status, col_bot, col_error_exp = st.columns([1.2, 1, 1, 2])

    with col_score:
        overall_score = response_data.get('overall_score', pd.NA)
        score_color = get_color_for_score(overall_score, OVERALL_THRESHOLDS)
        try:
            score_num = float(overall_score)
            score_display = f"{score_num:.0f}" if pd.notna(score_num) else "?"
        except (ValueError, TypeError): score_display = "?"
        st.markdown(f"""
            <div class="overall-score-container" style="background-color: {score_color};">
                <span class="overall-score-title">Overall Score</span>
                <div class='overall-score-value'>{score_display}<span class='overall-score-label'> / 100</span></div>
            </div>""", unsafe_allow_html=True)

    with col_status:
        topic_status = response_data.get('topic_status', 'Unknown')
        if pd.isna(topic_status) or topic_status == '': topic_status = 'Unknown'
        status_lower = topic_status.lower()
        if status_lower == "on-topic": st.markdown(f"<p style='color:{CHART_SUCCESS_COLOR}; font-weight:bold; margin-top:1.5rem; text-align:center;'>✅ On-Topic</p>", unsafe_allow_html=True)
        elif status_lower == "off-topic": st.markdown(f"<p style='color:{CHART_ERROR_COLOR}; font-weight:bold; margin-top:1.5rem; text-align:center;'>❌ Off-Topic</p>", unsafe_allow_html=True)
        else: st.markdown(f"<p style='color:{CHART_WARNING_COLOR}; margin-top:1.5rem; text-align:center;'>⚠️ Unknown Status</p>", unsafe_allow_html=True)

    with col_bot:
         bot_lik = response_data.get('bot_likelihood', pd.NA)
         render_colored_metric("Bot Likelihood", bot_lik, '/ 10', thresholds={}) # Use metric renderer

    with col_error_exp:
        error_msg = response_data.get('error', ''); explanation = response_data.get('explanation', '')
        st.markdown("**AI Rationale / Issues**")
        if error_msg and pd.notna(error_msg) and error_msg.strip(): st.warning(f"_{error_msg}_")
        elif explanation and pd.notna(explanation) and explanation.strip(): st.caption(f"_{explanation}_")
        else: st.caption("_No specific explanation or issues noted._")

    st.markdown("---")

    # --- Row 2: Metrics & Sentiment (Left) / Radar Chart (Right) --- ## <<-- LAYOUT CHANGE HERE -- ##
    col_metrics_sentiment, col_radar = st.columns([3, 2]) # Adjust ratio if needed (e.g., [2, 1])

    # --- Left Column: Quality Metrics + Sentiment Analysis ---
    with col_metrics_sentiment:
        # Quality Metrics
        st.markdown("<h6>Quality Metrics</h6>", unsafe_allow_html=True)
        m_cols1 = st.columns(3)
        with m_cols1[0]: render_colored_metric("Relevance", response_data.get('relevance', pd.NA))
        with m_cols1[1]: render_colored_metric("Completeness", response_data.get('completeness', pd.NA))
        with m_cols1[2]: render_colored_metric("Specificity", response_data.get('specificity', pd.NA))
        # Use 2 columns for the remaining 2 metrics for better spacing
        m_cols2 = st.columns(2)
        with m_cols2[0]: render_colored_metric("Language", response_data.get('language_quality', pd.NA))
        with m_cols2[1]: render_colored_metric("Sent. Align.", response_data.get('sentiment_alignment', pd.NA))

        st.markdown("<br>", unsafe_allow_html=True) # Add space before sentiment

        # Sentiment Analysis (Moved Here)
        st.markdown("<h6>Sentiment Analysis</h6>", unsafe_allow_html=True)
        sentiment_data = response_data.get('sentiment')
        sentiment_label = "N/A"; sentiment_score = pd.NA; score_display = "?"; sent_color = BODY_TEXT_COLOR
        if isinstance(sentiment_data, dict):
            sentiment_label = sentiment_data.get('label', 'Error')
            try:
                sentiment_score = float(sentiment_data.get('score'))
                if pd.notna(sentiment_score):
                    score_display = f"{sentiment_score:.0%}" # Format as percentage
                    sent_color = get_color_for_score(sentiment_score, SENTIMENT_THRESHOLDS)
                else: sentiment_label = "Score N/A"
            except (ValueError, TypeError): sentiment_score = pd.NA; sentiment_label = "Invalid Score"
        elif pd.notna(sentiment_data): sentiment_label = f"Invalid ({type(sentiment_data).__name__})"

        # Display sentiment label and confidence side-by-side maybe?
        sent_col1, sent_col2 = st.columns(2)
        with sent_col1:
             st.markdown(f"**Label:** `{sentiment_label}`")
        with sent_col2:
             st.markdown(f"**Confidence:** <span style='color:{sent_color}; font-weight:bold;'>{score_display}</span>", unsafe_allow_html=True)


    # --- Right Column: Radar Chart ---
    with col_radar:
        st.markdown("<h6>Metric Radar</h6>", unsafe_allow_html=True)
        radar_categories = ['relevance', 'completeness', 'specificity', 'language_quality', 'sentiment_alignment']
        radar_values = [response_data.get(cat, 0) for cat in radar_categories] # Default to 0

        if any(pd.notna(v) and v > 0 for v in radar_values):
            try:
                fig_radar = create_radar_chart(radar_values, radar_categories) # Use UI component function
                if fig_radar: st.pyplot(fig_radar, use_container_width=True)
                else: st.caption("Could not generate radar chart.")
            except Exception as e:
                st.warning(f"Could not generate Radar Chart: {e}")
                logging.warning(f"Radar chart generation error for row {response_data.name}: {e}")
        else:
             st.caption("No metric data > 0 for radar chart.")


    st.markdown("---")

    # --- Row 3: Manual Tagging & Action Items (No Change) ---
    st.markdown("<h6>Manual Input</h6>", unsafe_allow_html=True)
    tag_col, action_col = st.columns(2)
    with tag_col:
        tags = response_data.get('manual_tags', ''); tags = '' if pd.isna(tags) else tags
        st.markdown("**Tags:**")
        st.code(f"{tags if tags else '(None)'}", language=None)
    with action_col:
        action = response_data.get('action_item', ''); action = '' if pd.isna(action) else action
        st.markdown("**Action Item:**")
        st.caption(f"{action if action else '(None)'}")

    # Close the main container div
    st.markdown("</div>", unsafe_allow_html=True)

# --- [ REST OF THE CODE IN analyzer_ui_components.py remains the same ] ---
# Make sure the functions render_colored_metric, create_radar_chart, etc. are present in the file.