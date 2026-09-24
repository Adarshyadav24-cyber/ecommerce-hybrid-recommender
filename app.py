import streamlit as st
import pandas as pd
import numpy as np

from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 30px 35px;
    border-radius: 20px;
    margin-bottom: 25px;
    background: linear-gradient(
        135deg,
        #111827 0%,
        #1f2937 50%,
        #374151 100%
    );
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.85;
}

.metric-card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.05);
    min-height: 115px;
}

.metric-title {
    font-size: 13px;
    opacity: 0.7;
}

.metric-value {
    font-size: 29px;
    font-weight: 750;
    margin-top: 6px;
}

.product-card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 12px;
    background: rgba(128,128,128,0.035);
}

.product-title {
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 5px;
}

.product-id {
    font-size: 13px;
    opacity: 0.6;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: rgba(128,128,128,0.12);
    font-size: 12px;
    margin-right: 5px;
}

.score {
    font-size: 25px;
    font-weight: 800;
    text-align: center;
}

.score-label {
    font-size: 11px;
    opacity: 0.65;
    text-align: center;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-box {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.20);
    background: rgba(128,128,128,0.04);
}

.reason-box {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.20);
    background: rgba(128,128,128,0.035);
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    train = pd.read_csv(
        "data/processed/train_interactions.csv"
    )

    items = pd.read_csv(
        "data/raw/items.csv"
    )

    matrix = load_npz(
        "data/processed/user_item_matrix.npz"
    )

    return train, items, matrix


train, items, matrix = load_data()


# ============================================================
# USER / ITEM MAPPING
# ============================================================

user_ids = sorted(
    train["user_id"].astype(str).unique()
)

items["item_id"] = items["item_id"].astype(str)

item_ids = items["item_id"].tolist()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}


# ============================================================
# COLLABORATIVE SIMILARITY
# IMPORTANT:
# _matrix prevents Streamlit cache hashing error
# ============================================================

@st.cache_data
def calculate_collaborative_similarity(_matrix):

    return cosine_similarity(
        _matrix.T
    )


item_similarity = calculate_collaborative_similarity(
    matrix
)


# ============================================================
# CONTENT DATA
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


@st.cache_data
def calculate_content_similarity(text):

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        text
    )

    return cosine_similarity(
        tfidf_matrix
    )


content_similarity = calculate_content_similarity(
    items["text"]
)


# ============================================================
# SCORE NORMALIZATION
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
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_products(
    user_id,
    alpha,
    top_n
):

    user_id = str(user_id)

    if user_id not in user_to_index:

        return None

    user_index = user_to_index[user_id]

    interacted_indices = matrix[
        user_index
    ].indices

    if len(interacted_indices) == 0:

        return None


    # --------------------------------------------------------
    # COLLABORATIVE SCORE
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
    # CONTENT SCORE
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
    # HYBRID SCORE
    # --------------------------------------------------------

    hybrid_scores = (
        alpha * collaborative_scores
        +
        (1 - alpha) * content_scores
    )


    # Remove already interacted products

    hybrid_scores[
        interacted_indices
    ] = -np.inf


    # Top recommendations

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


    result = pd.DataFrame(
        recommendations
    )


    result = result.merge(
        items,
        on="item_id",
        how="left"
    )


    return result


# ============================================================
# USER HISTORY
# ============================================================

def get_user_history(
    user_id,
    limit=20
):

    user_id = str(user_id)

    if user_id not in user_to_index:

        return pd.DataFrame()

    user_index = user_to_index[user_id]

    interacted_indices = matrix[
        user_index
    ].indices

    if len(interacted_indices) == 0:

        return pd.DataFrame()

    history_item_ids = [
        item_ids[index]
        for index in interacted_indices
    ]

    history = items[
        items["item_id"].isin(
            history_item_ids
        )
    ].copy()

    return history.head(limit)


# ============================================================
# RECOMMENDATION EXPLANATION
# ============================================================

def get_recommendation_reason(
    row,
    alpha
):

    collaborative = float(
        row["collaborative_score"]
    )

    content = float(
        row["content_score"]
    )

    reasons = []


    # Content explanation

    if content >= 0.70:

        reasons.append(
            "Very high product-content similarity"
        )

    elif content >= 0.50:

        reasons.append(
            "High product-content similarity"
        )

    elif content >= 0.30:

        reasons.append(
            "Moderate product-content similarity"
        )

    else:

        reasons.append(
            "Lower content similarity contribution"
        )


    # Collaborative explanation

    if collaborative >= 0.70:

        reasons.append(
            "Very strong similarity with user interactions"
        )

    elif collaborative >= 0.50:

        reasons.append(
            "Strong similarity with user interactions"
        )

    elif collaborative >= 0.30:

        reasons.append(
            "Moderate collaborative similarity"
        )


    # Model mode

    if alpha == 0:

        reasons.append(
            "Recommendation generated using Content-Based Filtering"
        )

    elif alpha == 1:

        reasons.append(
            "Recommendation generated using Collaborative Filtering"
        )

    else:

        reasons.append(
            "Recommendation generated using Hybrid Filtering"
        )


    return reasons


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🛍️ ShopSense AI"
    )

    st.caption(
        "Intelligent E-Commerce Recommendation Engine"
    )

    st.divider()

    st.markdown(
        "### Recommendation Settings"
    )


    user_id = st.selectbox(
        "👤 Select User",
        user_ids
    )


    alpha = st.slider(
        "⚖️ Collaborative Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.25
    )


    top_n = st.selectbox(
        "🔢 Number of Recommendations",
        [5, 10, 15, 20],
        index=1
    )


    st.divider()

    st.markdown(
        "### Model Configuration"
    )


    if alpha == 0:

        st.info(
            "Content-Based Mode\n\n"
            "Recommendations are driven entirely "
            "by product similarity."
        )

    elif alpha == 1:

        st.info(
            "Collaborative Mode\n\n"
            "Recommendations are driven entirely "
            "by user-item interactions."
        )

    else:

        st.info(
            f"Hybrid Mode\n\n"
            f"Collaborative: {alpha:.0%}\n\n"
            f"Content: {(1-alpha):.0%}"
        )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🛍️ ShopSense AI
</div>

<div class="hero-subtitle">
Personalized E-Commerce Recommendation Dashboard
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">
        TOTAL USERS
        </div>

        <div class="metric-value">
        {len(user_ids):,}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">
        PRODUCTS
        </div>

        <div class="metric-value">
        {len(item_ids):,}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">
        TRAINING INTERACTIONS
        </div>

        <div class="metric-value">
        {len(train):,}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">
        RECOMMENDER
        </div>

        <div class="metric-value">
        HYBRID
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# USER HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">🛒 User Interaction History</div>',
    unsafe_allow_html=True
)


history = get_user_history(
    user_id,
    limit=10
)


if history.empty:

    st.info(
        "No interaction history found for this user."
    )

else:

    st.caption(
        f"Previously interacted products for {user_id}"
    )

    history_display = history[
        [
            "item_id",
            "title",
            "category",
            "brand"
        ]
    ].copy()

    st.dataframe(
        history_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RECOMMENDATION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Personalized Recommendations</div>',
    unsafe_allow_html=True
)


generate = st.button(
    "✨ Generate Recommendations",
    use_container_width=True,
    type="primary"
)


if generate:

    with st.spinner(
        "Analyzing user preferences and generating recommendations..."
    ):

        result = recommend_products(
            user_id,
            alpha,
            top_n
        )


    if result is None:

        st.error(
            "No recommendations available for this user."
        )


    else:

        st.success(
            f"Successfully generated "
            f"{len(result)} recommendations for {user_id}."
        )


        # ====================================================
        # PRODUCT CARDS
        # ====================================================

        for rank, (_, row) in enumerate(
            result.iterrows(),
            start=1
        ):


            col1, col2, col3 = st.columns(
                [0.7, 6, 1.5]
            )


            # ------------------------------------------------
            # RANK
            # ------------------------------------------------

            with col1:

                st.markdown(
                    f"""
                    <div style="
                    font-size:26px;
                    font-weight:800;
                    text-align:center;
                    padding-top:15px;">
                    #{rank}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # PRODUCT
            # ------------------------------------------------

            with col2:

                title = row.get(
                    "title",
                    "Unknown Product"
                )

                category = row.get(
                    "category",
                    "Unknown"
                )

                brand = row.get(
                    "brand",
                    "Unknown"
                )

                product_id = row.get(
                    "item_id",
                    "N/A"
                )


                st.markdown(
                    f"""
                    <div class="product-card">

                    <div class="product-title">
                    {title}
                    </div>

                    <div class="product-id">
                    Product ID: {product_id}
                    </div>

                    <br>

                    <span class="badge">
                    📦 {category}
                    </span>

                    <span class="badge">
                    🏷️ {brand}
                    </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # SCORE
            # ------------------------------------------------

            with col3:

                score = float(
                    row["hybrid_score"]
                )

                st.markdown(
                    f"""
                    <div style="
                    padding-top:25px;">

                    <div class="score">
                    {score:.3f}
                    </div>

                    <div class="score-label">
                    RECOMMENDATION SCORE
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # RECOMMENDATION EXPLANATION
            # =================================================

            reasons = get_recommendation_reason(
                row,
                alpha
            )


            with st.expander(
                "💡 Why is this product recommended?"
            ):


                st.markdown(
                    "### 🔎 Recommendation Explanation"
                )


                for reason in reasons:

                    st.write(
                        f"✓ {reason}"
                    )


                st.markdown(
                    "<br>",
                    unsafe_allow_html=True
                )


                score_col1, score_col2, score_col3 = st.columns(3)


                with score_col1:

                    st.metric(
                        "Content Score",
                        f"{float(row['content_score']):.3f}"
                    )


                with score_col2:

                    st.metric(
                        "Collaborative Score",
                        f"{float(row['collaborative_score']):.3f}"
                    )


                with score_col3:

                    st.metric(
                        "Hybrid Score",
                        f"{float(row['hybrid_score']):.3f}"
                    )


                # Detailed interpretation

                if float(row["content_score"]) >= 0.60:

                    st.success(
                        "The product has strong similarity "
                        "with products in the user's interaction history."
                    )

                elif float(row["content_score"]) >= 0.40:

                    st.info(
                        "The product has moderate content similarity "
                        "with the user's previous interests."
                    )


                if float(row["collaborative_score"]) >= 0.60:

                    st.success(
                        "The collaborative model found strong "
                        "similarity with the user's interaction pattern."
                    )

                elif float(row["collaborative_score"]) >= 0.40:

                    st.info(
                        "The collaborative model found moderate "
                        "similarity with the user's interaction pattern."
                    )


        # ====================================================
        # SCORE BREAKDOWN
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Recommendation Score Breakdown</div>',
            unsafe_allow_html=True
        )


        chart_data = result[
            [
                "item_id",
                "collaborative_score",
                "content_score"
            ]
        ].copy()


        chart_data = chart_data.set_index(
            "item_id"
        )


        st.bar_chart(
            chart_data
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">📊 Model Performance Comparison</div>',
    unsafe_allow_html=True
)


try:

    model_results = pd.read_csv(
        "reports/model_comparison.csv"
    )


    model_results.columns = (
        model_results.columns
        .str.strip()
    )


    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "### 📈 Precision@10"
    )


    precision_chart = model_results[
        ["Model", "Precision@10"]
    ].set_index("Model")


    st.bar_chart(
        precision_chart
    )


    st.markdown(
        "### 📈 Recall@10"
    )


    recall_chart = model_results[
        ["Model", "Recall@10"]
    ].set_index("Model")


    st.bar_chart(
        recall_chart
    )


    st.markdown(
        "### 📈 MAP@10"
    )


    map_chart = model_results[
        ["Model", "MAP@10"]
    ].set_index("Model")


    st.bar_chart(
        map_chart
    )


    st.markdown(
        "### 📈 NDCG@10"
    )


    ndcg_chart = model_results[
        ["Model", "NDCG@10"]
    ].set_index("Model")


    st.bar_chart(
        ndcg_chart
    )


except FileNotFoundError:

    st.warning(
        "reports/model_comparison.csv not found."
    )


# ============================================================
# ALPHA ANALYSIS
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">⚖️ Hybrid Model Alpha Analysis</div>',
    unsafe_allow_html=True
)


st.caption(
    "Alpha controls the balance between "
    "Collaborative Filtering and Content-Based Filtering."
)


try:

    alpha_results = pd.read_csv(
        "reports/alpha_tuning.csv"
    )


    alpha_results.columns = (
        alpha_results.columns
        .str.strip()
    )


    st.dataframe(
        alpha_results,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "### 🎯 Precision vs Alpha"
    )


    alpha_precision = alpha_results[
        ["Alpha", "Precision@10"]
    ].set_index("Alpha")


    st.line_chart(
        alpha_precision
    )


    st.markdown(
        "### 🎯 Recall vs Alpha"
    )


    alpha_recall = alpha_results[
        ["Alpha", "Recall@10"]
    ].set_index("Alpha")


    st.line_chart(
        alpha_recall
    )


    st.markdown(
        "### 🎯 MAP vs Alpha"
    )


    alpha_map = alpha_results[
        ["Alpha", "MAP@10"]
    ].set_index("Alpha")


    st.line_chart(
        alpha_map
    )


    st.markdown(
        "### 🎯 NDCG vs Alpha"
    )


    alpha_ndcg = alpha_results[
        ["Alpha", "NDCG@10"]
    ].set_index("Alpha")


    st.line_chart(
        alpha_ndcg
    )


except FileNotFoundError:

    st.warning(
        "reports/alpha_tuning.csv not found."
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">📦 Dataset Summary</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Users",
        f"{len(user_ids):,}"
    )


with col2:

    st.metric(
        "Products",
        f"{len(item_ids):,}"
    )


with col3:

    st.metric(
        "Interactions",
        f"{len(train):,}"
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">🧠 Recommendation Engine</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        """
        <div class="info-box">

        <h4>🤝 Collaborative Filtering</h4>

        Uses historical user-item interactions
        to identify products related to the user's
        interaction pattern.

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="info-box">

        <h4>🔎 Content-Based Filtering</h4>

        Uses product attributes such as title,
        category, brand and description to identify
        similar products.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">🏗️ System Architecture</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="info-box">

    <b>1. Data Layer</b><br>
    Users + Products + Interaction History

    <br><br>

    ↓

    <br><br>

    <b>2. Processing Layer</b><br>
    Data Cleaning → Train/Test Split → User-Item Matrix

    <br><br>

    ↓

    <br><br>

    <b>3. Recommendation Layer</b><br>
    Collaborative Filtering + Content-Based Filtering

    <br><br>

    ↓

    <br><br>

    <b>4. Hybrid Layer</b><br>
    Weighted combination using Alpha

    <br><br>

    ↓

    <br><br>

    <b>5. Evaluation Layer</b><br>
    Precision@10 + Recall@10 + MAP@10 + NDCG@10

    <br><br>

    ↓

    <br><br>

    <b>6. Application Layer</b><br>
    Streamlit Personalized Recommendation Dashboard

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "ShopSense AI • Hybrid E-Commerce Recommendation System • "
    "Python + Pandas + SciPy + Scikit-learn + Streamlit"
)