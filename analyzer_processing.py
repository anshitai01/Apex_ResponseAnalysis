# analyzer_processing.py
"""
Contains data processing functions for the AI Response Analyzer,
specifically the text clustering functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# --- Logging Configuration (Setup primarily in main app) ---
# logging.basicConfig(...)

# --- Clustering Function ---
def cluster_responses(responses, threshold_percentage):
    """
    Cluster responses based on TF-IDF and cosine similarity using hierarchical clustering.

    Args:
        responses (list): A list of response strings.
        threshold_percentage (int): Similarity threshold (0-100) used to determine
                                     distance for clustering. Higher % means groups
                                     need to be more similar.

    Returns:
        pd.DataFrame: DataFrame with columns ['response', 'Group', 'Similarity Score'],
                      where 'Group' indicates the cluster number or 'Unique'/'N/A'/'Error',
                      and 'Similarity Score' is the max similarity to another item
                      in the same cluster (NaN if Unique/N/A/Error). Returns a basic
                      DataFrame with 'N/A' or 'Error' groups if clustering fails.
    """
    logging.info(f"Starting clustering for {len(responses)} responses with threshold {threshold_percentage}%.")

    # --- Input Validation and Filtering ---
    if not responses or not isinstance(responses, list):
         st.warning("No responses provided for clustering.")
         logging.warning("cluster_responses: Input 'responses' is empty or not a list.")
         # Return structure matching expected output even on failure
         return pd.DataFrame({'response': responses if responses else [], 'Group': 'N/A', 'Similarity Score': np.nan})

    # Filter out None/empty strings and keep track of original indices
    original_indices = [i for i, r in enumerate(responses) if r and isinstance(r, str) and str(r).strip()]
    valid_responses = [responses[i] for i in original_indices]

    if not valid_responses:
        st.warning("All provided responses were empty after filtering. Cannot perform clustering.")
        logging.warning("cluster_responses: No valid responses remain after filtering.")
        return pd.DataFrame({'response': responses, 'Group': 'N/A', 'Similarity Score': np.nan})

    if len(valid_responses) == 1:
        st.info("Only one valid response found. Assigning it to the 'Unique' group.")
        logging.info("cluster_responses: Only one valid response, assigned to 'Unique'.")
        # Create result DF matching the full input list structure
        df_result = pd.DataFrame({'response': responses, 'Group': 'N/A', 'Similarity Score': np.nan})
        # Assign 'Unique' only to the valid response's original index
        if original_indices: # Check if list is not empty
            df_result.loc[original_indices[0], 'Group'] = 'Unique'
        return df_result

    # Create DataFrame with only valid responses for processing
    df_valid = pd.DataFrame({'response': valid_responses}, index=original_indices)

    # --- TF-IDF Vectorization ---
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1) # min_df=1 ensures single words aren't ignored if they are the only content
    try:
        X = vectorizer.fit_transform(df_valid['response'])
        # Check if matrix is empty (e.g., only stopwords were present)
        if X.shape[0] == 0 or X.shape[1] == 0:
             st.warning("Text vectorization resulted in an empty matrix (perhaps only common words?). Cannot cluster.")
             logging.warning("cluster_responses: TF-IDF matrix is empty after fit_transform.")
             return pd.DataFrame({'response': responses, 'Group': 'N/A', 'Similarity Score': np.nan})
        logging.info(f"TF-IDF matrix created with shape: {X.shape}")

    except ValueError as e:
        # Handle specific error like "empty vocabulary"
        if "empty vocabulary" in str(e).lower():
            st.warning(f"Text vectorization failed: Vocabulary might be empty after removing stop words. Cannot cluster.")
            logging.warning(f"cluster_responses: TF-IDF ValueError - empty vocabulary.")
        else:
            st.error(f"Error during text vectorization: {e}. Cannot cluster.")
            logging.error(f"cluster_responses: TF-IDF Vectorization error: {e}")
        # Return error state
        return pd.DataFrame({'response': responses, 'Group': 'Error', 'Similarity Score': np.nan})
    except Exception as e: # Catch other potential vectorizer errors
        st.error(f"Unexpected error during text vectorization: {e}. Cannot cluster.")
        logging.exception("cluster_responses: Unexpected TF-IDF Vectorization error.")
        return pd.DataFrame({'response': responses, 'Group': 'Error', 'Similarity Score': np.nan})


    # --- Similarity and Distance Calculation ---
    try:
        similarity_matrix = cosine_similarity(X)
        # Ensure distance is non-negative (cosine similarity can be slightly > 1 due to float precision)
        distance_matrix = np.maximum(0, 1 - similarity_matrix)
        # Ensure diagonal is exactly zero
        np.fill_diagonal(distance_matrix, 0)
    except Exception as e:
         st.error(f"Error calculating similarity/distance matrix: {e}")
         logging.exception("cluster_responses: Error calculating similarity/distance.")
         return pd.DataFrame({'response': responses, 'Group': 'Error', 'Similarity Score': np.nan})

    # --- Hierarchical Clustering ---
    clusters = None
    try:
        # Check for near-zero distances which can cause issues
        if np.allclose(distance_matrix, 0):
            st.warning("All valid responses appear identical based on text analysis. Assigning all to Group 1.")
            logging.warning("cluster_responses: All distances are close to zero. Assigning all to cluster 1.")
            clusters = np.ones(distance_matrix.shape[0], dtype=int) # Assign all to cluster 1
        else:
            # Ensure symmetry for squareform (correcting minor float inaccuracies)
            if not np.allclose(distance_matrix, distance_matrix.T):
                logging.debug("Distance matrix not perfectly symmetric, averaging.")
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0) # Re-ensure diagonal is zero

            # Convert to condensed form for linkage
            condensed_distance_matrix = squareform(distance_matrix, checks=True) # Enable checks

            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distance_matrix, method='average') # Average linkage is common

            # Determine clusters based on distance threshold derived from similarity percentage
            # Distance = 1 - Similarity. Threshold is similarity, so use 1-similarity for distance.
            distance_threshold = max(0, 1.0 - (threshold_percentage / 100.0)) # Ensure threshold >= 0
            logging.info(f"Clustering with distance threshold: {distance_threshold:.4f}")

            clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    except ValueError as ve: # Catch errors from squareform or fcluster
         st.error(f"Error during clustering calculation: {ve}")
         logging.exception(f"Clustering linkage/fcluster ValueError: {ve}")
         return pd.DataFrame({'response': responses, 'Group': 'Error', 'Similarity Score': np.nan})
    except Exception as e: # Catch other potential clustering errors
         st.error(f"Unexpected error during clustering: {e}")
         logging.exception("cluster_responses: Unexpected clustering linkage/fcluster error")
         return pd.DataFrame({'response': responses, 'Group': 'Error', 'Similarity Score': np.nan})

    if clusters is None: # Should not happen if logic above is correct, but safeguard
         logging.error("cluster_responses: Clusters array is None after processing.")
         return pd.DataFrame({'response': responses, 'Group': 'Error', 'Similarity Score': np.nan})

    # Assign clusters to the valid DataFrame
    df_valid['Group'] = clusters

    # --- Calculate Max Similarity Score within each cluster ---
    similarity_scores = []
    for i in range(X.shape[0]): # Iterate through the *valid* responses used for clustering
        current_cluster = df_valid['Group'].iloc[i]
        # Find indices of other items *in the same cluster* within the valid subset
        same_cluster_mask = (df_valid['Group'] == current_cluster)
        # Get the row indices in the similarity matrix corresponding to this cluster
        indices_in_X_for_cluster = np.where(same_cluster_mask)[0]
        # Exclude the item itself
        other_indices_in_X = [idx for idx in indices_in_X_for_cluster if idx != i]

        if other_indices_in_X:
            # Find max similarity between item 'i' and others in its cluster
            max_sim = np.max(similarity_matrix[i, other_indices_in_X])
            similarity_scores.append(max_sim)
        else:
            # If item is alone in its cluster (will be marked 'Unique' later)
            similarity_scores.append(np.nan)

    df_valid['Similarity Score'] = similarity_scores

    # --- Merge results back to the original response list structure ---
    df_result = pd.DataFrame({'response': responses}) # Start with all original responses
    # Merge cluster/score info using the original indices stored in df_valid's index
    df_result = df_result.merge(df_valid[['Group', 'Similarity Score']],
                                left_index=True, right_index=True, how='left')

    # --- Identify and Label 'Unique' groups ---
    # Count occurrences of each group number *among clustered items*
    group_counts = df_valid['Group'].value_counts()
    # Find group numbers that only appear once
    unique_group_numbers = group_counts[group_counts == 1].index
    # Update the main result DataFrame
    df_result.loc[df_result['Group'].isin(unique_group_numbers), 'Group'] = "Unique"
    # Set similarity score to NaN for unique items
    df_result.loc[df_result['Group'] == "Unique", 'Similarity Score'] = np.nan

    # --- Final Formatting and Sorting ---
    # Fill any remaining NaNs in 'Group' (e.g., for original empty responses)
    df_result['Group'] = df_result['Group'].fillna('N/A')
    # Ensure Similarity Score is float or NaN
    df_result['Similarity Score'] = pd.to_numeric(df_result['Similarity Score'], errors='coerce')
    # Ensure Group column is string type
    df_result['Group'] = df_result['Group'].astype(str)

    # Create a sort key: Treat 'Unique', 'N/A', 'Error' specially
    def group_sort_key(group_val):
        if group_val.isdigit():
            return int(group_val)
        elif group_val == 'Unique':
            return np.inf - 2 # Sort Unique after numbered groups
        elif group_val == 'N/A':
            return np.inf -1 # Sort N/A after Unique
        elif group_val == 'Error':
             return np.inf # Sort Error last
        else: # Should not happen, but handle other strings
             return np.inf - 3

    df_result['Group_Sort'] = df_result['Group'].apply(group_sort_key)
    # Sort by Group number (ascending), then Similarity Score (descending for highest sim first)
    df_result = df_result.sort_values(
        by=['Group_Sort', 'Similarity Score'],
        ascending=[True, False],
        na_position='last' # Put NaNs in similarity score last within a group
    )
    # Remove the temporary sort column
    df_result.drop(columns=['Group_Sort'], inplace=True)

    logging.info("Clustering finished successfully.")
    return df_result[['response', 'Group', 'Similarity Score']] # Return selected columns