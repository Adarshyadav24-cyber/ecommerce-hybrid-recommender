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
# 2. CORRECT USER / ITEM INDEX MAPPING
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
# 4. CONTENT-BASED TF-IDF
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
# 5. NORMALIZATION FUNCTION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(scores)

    min_value = scores.min()
    max_value = scores.max()

    if max_value == min_value:
        return np.zeros_like(scores)

    return (
        (scores - min_value)
        / (max_value - min_value)
    )


# ============================================================
# 6. POPULARITY RECOMMENDER
# ============================================================

popularity_counts = (
    train.groupby("item_id")
    .size()
)

popularity_scores = np.zeros(
    len(item_ids)
)

for item_id, count in popularity_counts.items():

    if item_id in item_to_index:

        popularity_scores[
            item_to_index[item_id]
        ] = count


popularity_scores = normalize_scores(
    popularity_scores
)


def popularity_recommend(
    user_id,
    top_n=10
):

    user_index = user_to_index.get(
        user_id
    )

    if user_index is None:
        return []

    seen_items = set(
        train[
            train["user_id"] == user_id
        ]["item_id"]
    )

    scores = popularity_scores.copy()

    for item_id in seen_items:

        if item_id in item_to_index:

            scores[
                item_to_index[item_id]
            ] = -np.inf

    indices = np.argsort(
        scores
    )[::-1][:top_n]

    return [
        item_ids[index]
        for index in indices
        if scores[index] != -np.inf
    ]


# ============================================================
# 7. COLLABORATIVE RECOMMENDER
# ============================================================

def collaborative_recommend(
    user_id,
    top_n=10
):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[
        user_id
    ]

    user_vector = matrix[
        user_index
    ]

    interacted_indices = (
        user_vector.indices
    )

    if len(interacted_indices) == 0:
        return []

    scores = np.zeros(
        matrix.shape[1]
    )

    for item_index in interacted_indices:

        scores += item_similarity[
            item_index
        ]

    scores = normalize_scores(
        scores
    )

    scores[
        interacted_indices
    ] = -np.inf

    indices = np.argsort(
        scores
    )[::-1][:top_n]

    return [
        item_ids[index]
        for index in indices
        if scores[index] != -np.inf
    ]


# ============================================================
# 8. CONTENT-BASED RECOMMENDER
# ============================================================

def content_recommend(
    user_id,
    top_n=10
):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[
        user_id
    ]

    interacted_indices = (
        matrix[
            user_index
        ].indices
    )

    if len(interacted_indices) == 0:
        return []

    scores = np.zeros(
        len(item_ids)
    )

    for item_index in interacted_indices:

        scores += content_similarity[
            item_index
        ]

    scores = normalize_scores(
        scores
    )

    scores[
        interacted_indices
    ] = -np.inf

    indices = np.argsort(
        scores
    )[::-1][:top_n]

    return [
        item_ids[index]
        for index in indices
        if scores[index] != -np.inf
    ]


# ============================================================
# 9. HYBRID RECOMMENDER
# ============================================================

def hybrid_recommend(
    user_id,
    top_n=10,
    alpha=0.7
):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[
        user_id
    ]

    interacted_indices = (
        matrix[
            user_index
        ].indices
    )

    if len(interacted_indices) == 0:
        return []

    # Collaborative score

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

    # Content score

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

    # Hybrid

    hybrid_scores = (
        alpha * collaborative_scores
        +
        (1 - alpha) * content_scores
    )

    hybrid_scores[
        interacted_indices
    ] = -np.inf

    indices = np.argsort(
        hybrid_scores
    )[::-1][:top_n]

    return [
        item_ids[index]
        for index in indices
        if hybrid_scores[index] != -np.inf
    ]


# ============================================================
# 10. PRECISION@K
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


# ============================================================
# 11. RECALL@K
# ============================================================

def recall_at_k(
    recommended,
    actual,
    k=10
):

    if not actual:
        return 0.0

    recommended = recommended[:k]

    hits = len(
        set(recommended)
        &
        set(actual)
    )

    return hits / len(actual)


# ============================================================
# 12. AP@K
# ============================================================

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


# ============================================================
# 13. NDCG@K
# ============================================================

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
# 14. EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    recommender,
    k=10,
    max_users=200
):

    test_users = test[
        "user_id"
].unique()

    if max_users is not None:
        test_users = test_users[:max_users]

    precision_scores = []
    recall_scores = []
    map_scores = []
    ndcg_scores = []

    for user_id in test_users:

        actual_items = test[
            test["user_id"] == user_id
        ]["item_id"].tolist()

        recommended_items = recommender(
            user_id,
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
# 15. COMPARE ALL MODELS
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "   RECOMMENDER MODEL COMPARISON"
    )

    print(
        "========================================\n"
    )

    models = {

        "Popularity":
            popularity_recommend,

        "Collaborative":
            collaborative_recommend,

        "Content-Based":
            content_recommend,

        "Hybrid":
            hybrid_recommend
    }

    results = []

    for model_name, recommender in models.items():

        print(
            f"Evaluating {model_name}..."
        )

        metrics = evaluate_model(
            recommender,
            k=10,
            max_users=None
)

        metrics["Model"] = model_name

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "Model",
            "Precision@10",
            "Recall@10",
            "MAP@10",
            "NDCG@10",
            "Users"
        ]
    ]

    print(
        "\n========================================"
    )

    print(
        "           FINAL RESULTS"
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
        "reports/model_comparison.csv",
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        "reports/model_comparison.csv"
    )