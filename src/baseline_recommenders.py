import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. LOAD DATA
# =========================

train = pd.read_csv("data/processed/train_interactions.csv")
items = pd.read_csv("data/raw/items.csv")

matrix = load_npz(
    "data/processed/user_item_matrix.npz"
)


# =========================
# 2. ITEM-ITEM COLLABORATIVE FILTERING
# =========================

def item_item_recommender(
    user_id,
    train_data,
    item_matrix,
    items_data,
    top_n=10
):

    # Find user index
    user_ids = train_data["user_id"].unique()

    if user_id not in user_ids:
        return pd.DataFrame()

    user_index = list(user_ids).index(user_id)

    # Items interacted with by the user
    user_vector = item_matrix[user_index]

    interacted_items = user_vector.indices

    if len(interacted_items) == 0:
        return pd.DataFrame()

    # Item-item similarity
    similarity = cosine_similarity(
        item_matrix.T
    )

    # Calculate recommendation score
    scores = np.zeros(item_matrix.shape[1])

    for item_index in interacted_items:

        scores += similarity[item_index]

    # Do not recommend already interacted items
    scores[interacted_items] = -np.inf

    # Top N items
    recommended_indices = np.argsort(
        scores
    )[::-1][:top_n]

    # Get item IDs
    item_ids = items_data["item_id"].values

    recommended_item_ids = item_ids[
        recommended_indices
    ]

    result = items_data[
        items_data["item_id"].isin(
            recommended_item_ids
        )
    ].copy()

    return result


# =========================
# 3. TEST
# =========================

if __name__ == "__main__":

    # Select one user
    test_user = train["user_id"].iloc[0]

    recommendations = item_item_recommender(
        test_user,
        train,
        matrix,
        items,
        top_n=10
    )

    print("\n===== ITEM-ITEM RECOMMENDATIONS =====\n")

    print(
        recommendations[
            ["item_id"]
        ].to_string(index=False)
    )
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 4. CONTENT-BASED TF-IDF
# =========================

def content_based_recommender(
    item_id,
    items_data,
    top_n=10
):

    data = items_data.copy()

    # Combine product text fields
    text_columns = [
        "title",
        "category",
        "brand",
        "description"
    ]

    for column in text_columns:
        if column not in data.columns:
            data[column] = ""

    data["text"] = (
        data["title"].fillna("").astype(str)
        + " "
        + data["category"].fillna("").astype(str)
        + " "
        + data["brand"].fillna("").astype(str)
        + " "
        + data["description"].fillna("").astype(str)
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        data["text"]
    )

    # Find selected item
    item_indices = data.index[
        data["item_id"] == item_id
    ].tolist()

    if not item_indices:
        return pd.DataFrame()

    item_index = item_indices[0]

    # Similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[item_index],
        tfidf_matrix
    ).flatten()

    # Sort by similarity
    similar_indices = similarity_scores.argsort()[
        ::-1
    ]

    # Remove the selected item itself
    similar_indices = [
        index
        for index in similar_indices
        if index != item_index
    ]

    # Top N
    recommended_indices = similar_indices[:top_n]

    result = data.iloc[
        recommended_indices
    ].copy()

    result["similarity_score"] = similarity_scores[
        recommended_indices
    ]

    return result


# =========================
# 5. TEST CONTENT MODEL
# =========================

if __name__ == "__main__":

    test_item = items["item_id"].iloc[0]

    recommendations = content_based_recommender(
        test_item,
        items,
        top_n=10
    )

    print(
        "\n===== CONTENT-BASED RECOMMENDATIONS =====\n"
    )

    print(
        recommendations[
            ["item_id", "similarity_score"]
        ].to_string(index=False)
    )