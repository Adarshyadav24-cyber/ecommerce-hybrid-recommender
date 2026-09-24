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

test = pd.read_csv(
    "data/processed/test_interactions.csv"
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
# 3. COLLABORATIVE SIMILARITY
# ============================================================

item_similarity = cosine_similarity(
    matrix.T
)


# ============================================================
# 4. CONTENT SIMILARITY
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

    scores = np.asarray(scores)

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ============================================================
# 6. HYBRID RECOMMENDER
# ============================================================

def hybrid_recommend(
    user_id,
    alpha,
    top_n=10
):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[
        user_id
    ]

    interacted_indices = matrix[
        user_index
    ].indices

    if len(interacted_indices) == 0:
        return []

    # -----------------------------
    # Collaborative score
    # -----------------------------

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

    # -----------------------------
    # Content score
    # -----------------------------

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

    # -----------------------------
    # Hybrid score
    # -----------------------------

    hybrid_scores = (
        alpha * collaborative_scores
        +
        (1 - alpha) * content_scores
    )

    # Remove already seen products

    hybrid_scores[
        interacted_indices
    ] = -np.inf

    recommended_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_n]

    return [
        item_ids[index]
        for index in recommended_indices
    ]


# ============================================================
# 7. METRICS
# ============================================================

def precision_at_k(
    recommended,
    actual,
    k=10
):

    recommended = recommended[:k]

    if not recommended:
        return 0.0

    hits = len(
        set(recommended)
        &
        set(actual)
    )

    return hits / k


def recall_at_k(
    recommended,
    actual,
    k=10
):

    if not actual:
        return 0.0

    hits = len(
        set(recommended[:k])
        &
        set(actual)
    )

    return hits / len(actual)


def average_precision_at_k(
    recommended,
    actual,
    k=10
):

    actual = set(actual)

    if not actual:
        return 0.0

    recommended = recommended[:k]

    score = 0.0
    hits = 0

    for position, item in enumerate(
        recommended,
        start=1
    ):

        if item in actual:

            hits += 1

            score += (
                hits / position
            )

    return score / min(
        len(actual),
        k
    )


def ndcg_at_k(
    recommended,
    actual,
    k=10
):

    actual = set(actual)

    recommended = recommended[:k]

    dcg = 0.0

    for position, item in enumerate(
        recommended,
        start=1
    ):

        if item in actual:

            dcg += (
                1
                /
                np.log2(position + 1)
            )

    ideal_hits = min(
        len(actual),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / np.log2(position + 1)
        for position in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


# ============================================================
# 8. EVALUATE ONE ALPHA
# ============================================================

def evaluate_alpha(
    alpha,
    k=10
):

    users = test[
        "user_id"
    ].unique()

    precision_scores = []
    recall_scores = []
    map_scores = []
    ndcg_scores = []

    for user_id in users:

        actual_items = test[
            test["user_id"] == user_id
        ]["item_id"].tolist()

        recommended_items = hybrid_recommend(
            user_id,
            alpha,
            top_n=k
        )

        if not recommended_items:
            continue

        precision_scores.append(
            precision_at_k(
                recommended_items,
                actual_items,
                k
            )
        )

        recall_scores.append(
            recall_at_k(
                recommended_items,
                actual_items,
                k
            )
        )

        map_scores.append(
            average_precision_at_k(
                recommended_items,
                actual_items,
                k
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                recommended_items,
                actual_items,
                k
            )
        )

    return {
        "Alpha": alpha,
        "Collaborative Weight": alpha,
        "Content Weight": 1 - alpha,
        "Precision@10": np.mean(
            precision_scores
        ),
        "Recall@10": np.mean(
            recall_scores
        ),
        "MAP@10": np.mean(
            map_scores
        ),
        "NDCG@10": np.mean(
            ndcg_scores
        ),
        "Users": len(
            precision_scores
        )
    }


# ============================================================
# 9. TEST DIFFERENT ALPHA VALUES
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "        HYBRID ALPHA TUNING"
    )

    print(
        "========================================\n"
    )

    alpha_values = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0
    ]

    results = []

    for alpha in alpha_values:

        print(
            f"Evaluating alpha = {alpha}"
        )

        result = evaluate_alpha(
            alpha
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n========================================"
    )

    print(
        "             ALPHA RESULTS"
    )

    print(
        "========================================\n"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # Save results

    results_df.to_csv(
        "reports/alpha_tuning.csv",
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        "reports/alpha_tuning.csv"
    )