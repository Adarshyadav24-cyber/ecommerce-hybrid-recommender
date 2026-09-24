import pandas as pd
import numpy as np

from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv(
    "data/processed/train_interactions.csv"
)

items = pd.read_csv(
    "data/raw/items.csv"
)

matrix = load_npz(
    "data/processed/user_item_matrix.npz"
)


# ============================================================
# 2. USER / ITEM MAPPING
# ============================================================

user_ids = sorted(
    train["user_id"].unique()
)

item_ids = items["item_id"].tolist()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

item_to_index = {
    item_id: index
    for index, item_id in enumerate(item_ids)
}


# ============================================================
# 3. ITEM-ITEM COLLABORATIVE SIMILARITY
# ============================================================

item_similarity = cosine_similarity(
    matrix.T
)


# ============================================================
# 4. CONTENT-BASED SIMILARITY
# ============================================================

text_columns = [
    "title",
    "category",
    "brand",
    "description"
]

for column in text_columns:

    if column not in items.columns:
        items[column] = ""


items["text"] = (
    items["title"].fillna("").astype(str)
    + " "
    + items["category"].fillna("").astype(str)
    + " "
    + items["brand"].fillna("").astype(str)
    + " "
    + items["description"].fillna("").astype(str)
)


vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    items["text"]
)

content_similarity = cosine_similarity(
    tfidf_matrix
)


# ============================================================
# 5. NORMALIZATION
# ============================================================

def normalize_scores(scores):

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# 6. HYBRID RECOMMENDER
# ============================================================

def hybrid_recommend(
    user_id,
    top_n=10,
    alpha=0.0
):

    if user_id not in user_to_index:

        print(
            f"\nUser '{user_id}' not found."
        )

        return None

    user_index = user_to_index[
        user_id
    ]

    interacted_indices = matrix[
        user_index
    ].indices

    if len(interacted_indices) == 0:

        print(
            "\nThis user has no previous interactions."
        )

        return None


    # --------------------------------------------------------
    # Collaborative score
    # --------------------------------------------------------

    collaborative_scores = np.zeros(
        len(item_ids)
    )

    for item_index in interacted_indices:

        collaborative_scores += (
            item_similarity[item_index]
        )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )


    # --------------------------------------------------------
    # Content score
    # --------------------------------------------------------

    content_scores = np.zeros(
        len(item_ids)
    )

    for item_index in interacted_indices:

        content_scores += (
            content_similarity[item_index]
        )

    content_scores = normalize_scores(
        content_scores
    )


    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

    hybrid_scores = (
        alpha * collaborative_scores
        +
        (1 - alpha) * content_scores
    )


    # Don't recommend already interacted products

    hybrid_scores[
        interacted_indices
    ] = -np.inf


    # --------------------------------------------------------
    # Top N
    # --------------------------------------------------------

    recommended_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_n]


    recommendations = []

    for index in recommended_indices:

        recommendations.append({

            "item_id":
                item_ids[index],

            "hybrid_score":
                hybrid_scores[index],

            "collaborative_score":
                collaborative_scores[index],

            "content_score":
                content_scores[index]
        })


    return pd.DataFrame(
        recommendations
    )


# ============================================================
# 7. DISPLAY PRODUCT DETAILS
# ============================================================

def show_recommendations(
    user_id,
    top_n=10
):

    recommendations = hybrid_recommend(
        user_id,
        top_n=top_n
    )

    if recommendations is None:
        return


    result = recommendations.merge(
        items,
        on="item_id",
        how="left"
    )


    print(
        "\n========================================"
    )

    print(
        "       PERSONALIZED RECOMMENDATIONS"
    )

    print(
        "========================================"
    )

    print(
        f"\nUser ID: {user_id}"
    )

    print(
        f"Top {top_n} Recommended Products:\n"
    )


    display_columns = [
        "item_id",
        "title",
        "category",
        "brand",
        "hybrid_score"
    ]


    available_columns = [
        column
        for column in display_columns
        if column in result.columns
    ]


    print(
        result[
            available_columns
        ].to_string(
            index=False
        )
    )


    return result


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "    E-COMMERCE RECOMMENDATION SYSTEM"
    )

    print(
        "========================================"
    )


    user_id = input(
        "\nEnter User ID: "
    ).strip()


    show_recommendations(
        user_id,
        top_n=10
    )