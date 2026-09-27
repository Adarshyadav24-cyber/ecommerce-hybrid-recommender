import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SCANNER
# ============================================================

try:
    from src.scanner.barcode_scanner import decode_qr
    SCANNER_AVAILABLE = True
except Exception:
    SCANNER_AVAILABLE = False


# ============================================================
# MARKETPLACE INTEGRATION
# ============================================================
try:
    from src.marketplace.marketplace_api import compare_marketplaces
    MARKETPLACE_AVAILABLE = True
except Exception:
    MARKETPLACE_AVAILABLE = False
    compare_marketplaces = None


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

USERS_FILE = os.path.join(RAW_DIR, "users.csv")
ITEMS_FILE = os.path.join(RAW_DIR, "items.csv")
INTERACTIONS_FILE = os.path.join(RAW_DIR, "interactions.csv")

TRAIN_FILE = os.path.join(
    PROCESSED_DIR,
    "train_interactions.csv"
)

TEST_FILE = os.path.join(
    PROCESSED_DIR,
    "test_interactions.csv"
)

MODEL_COMPARISON_FILE = os.path.join(
    REPORTS_DIR,
    "model_comparison.csv"
)

ALPHA_FILE = os.path.join(
    REPORTS_DIR,
    "alpha_tuning.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #777;
        margin-bottom: 25px;
    }

    .product-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-bottom: 12px;
    }

    .product-card:hover {
        border-color: rgba(100,100,100,0.45);
    }

    .small-text {
        color: #777;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    users = pd.read_csv(USERS_FILE)
    items = pd.read_csv(ITEMS_FILE)
    interactions = pd.read_csv(INTERACTIONS_FILE)

    return users, items, interactions


@st.cache_data
def load_train_data():

    if os.path.exists(TRAIN_FILE):
        return pd.read_csv(TRAIN_FILE)

    return pd.DataFrame()


@st.cache_data
def load_test_data():

    if os.path.exists(TEST_FILE):
        return pd.read_csv(TEST_FILE)

    return pd.DataFrame()


@st.cache_data
def load_model_comparison():

    if os.path.exists(MODEL_COMPARISON_FILE):
        return pd.read_csv(MODEL_COMPARISON_FILE)

    return pd.DataFrame()


@st.cache_data
def load_alpha_results():

    if os.path.exists(ALPHA_FILE):
        return pd.read_csv(ALPHA_FILE)

    return pd.DataFrame()


# ============================================================
# LOAD DATA
# ============================================================

try:

    users, items, interactions = load_data()
    train_df = load_train_data()
    test_df = load_test_data()

except Exception as e:

    st.error("❌ Could not load project data.")
    st.code(str(e))
    st.stop()


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


USER_COL = find_column(
    interactions,
    ["user_id", "user", "userid"]
)

ITEM_COL = find_column(
    interactions,
    ["item_id", "product_id", "item"]
)

TIMESTAMP_COL = find_column(
    interactions,
    ["timestamp", "datetime", "date", "time"]
)

EVENT_COL = find_column(
    interactions,
    [
        "event_type",
        "event",
        "interaction_type",
        "action"
    ]
)

ITEM_ID_COL = find_column(
    items,
    [
        "item_id",
        "product_id",
        "id"
    ]
)

TITLE_COL = find_column(
    items,
    [
        "title",
        "product_title",
        "name",
        "product_name"
    ]
)

CATEGORY_COL = find_column(
    items,
    [
        "category",
        "product_category"
    ]
)

BRAND_COL = find_column(
    items,
    [
        "brand",
        "manufacturer"
    ]
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

if USER_COL is None:
    st.error("❌ User ID column not found in interactions.csv")
    st.stop()

if ITEM_COL is None:
    st.error("❌ Item ID column not found in interactions.csv")
    st.stop()

if ITEM_ID_COL is None:
    st.error("❌ Item ID column not found in items.csv")
    st.stop()


# ============================================================
# PREPARE ITEM TEXT
# ============================================================

def prepare_item_text(df):

    text_columns = []

    for col in [
        TITLE_COL,
        CATEGORY_COL,
        BRAND_COL,
        "description"
    ]:

        if col is not None and col in df.columns:
            text_columns.append(col)

    if not text_columns:

        return pd.Series(
            [""] * len(df),
            index=df.index
        )

    result = df[text_columns].fillna("").astype(str).agg(
        " ".join,
        axis=1
    )

    return result


items = items.copy()

items["_search_text"] = prepare_item_text(items)


# ============================================================
# CONTENT MODEL
# ============================================================

@st.cache_resource
def build_content_model(_items_df):

    df = _items_df.copy()

    text_columns = []

    for col in [
        "title",
        "category",
        "brand",
        "description"
    ]:

        if col in df.columns:
            text_columns.append(col)

    if not text_columns:

        raise ValueError(
            "No usable product text columns found."
        )

    combined_text = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(
        combined_text
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    return similarity_matrix


# IMPORTANT:
# Pass complete DataFrame, not tuple(_search_text)

content_similarity = build_content_model(items)


# ============================================================
# INTERACTION MATRIX
# ============================================================

@st.cache_data
def build_interaction_matrix(
    interaction_data,
    item_ids
):

    data = interaction_data.copy()

    if data.empty:

        return (
            csr_matrix(
                (0, len(item_ids))
            ),
            [],
            {}
        )

    users_list = sorted(
        data[USER_COL]
        .astype(str)
        .unique()
    )

    user_index = {
        user: i
        for i, user in enumerate(users_list)
    }

    item_index = {
        str(item): i
        for i, item in enumerate(item_ids)
    }

    rows = []
    cols = []
    values = []

    for _, row in data.iterrows():

        user = str(row[USER_COL])
        item = str(row[ITEM_COL])

        if (
            user in user_index
            and item in item_index
        ):

            rows.append(
                user_index[user]
            )

            cols.append(
                item_index[item]
            )

            values.append(1.0)

    matrix = csr_matrix(
        (
            values,
            (rows, cols)
        ),
        shape=(
            len(users_list),
            len(item_ids)
        )
    )

    return matrix, users_list, user_index


# ============================================================
# ITEM IDS
# ============================================================

item_ids = (
    items[ITEM_ID_COL]
    .astype(str)
    .tolist()
)


# ============================================================
# BUILD MATRIX
# ============================================================

interaction_matrix, matrix_users, user_index = (
    build_interaction_matrix(
        train_df
        if not train_df.empty
        else interactions,
        item_ids
    )
)


# ============================================================
# COLLABORATIVE MODEL
# ============================================================

@st.cache_resource
def build_collaborative_similarity(_matrix):

    if _matrix.shape[0] == 0:

        return np.zeros(
            (
                _matrix.shape[1],
                _matrix.shape[1]
            )
        )

    item_matrix = _matrix.T

    return cosine_similarity(
        item_matrix
    )


collaborative_similarity = (
    build_collaborative_similarity(
        interaction_matrix
    )
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_item_index(item_id):

    item_id = str(item_id)

    matches = np.where(
        np.array(item_ids) == item_id
    )[0]

    if len(matches) == 0:
        return None

    return int(matches[0])


def get_user_history(user_id):

    data = interactions[
        interactions[USER_COL].astype(str)
        == str(user_id)
    ].copy()

    if (
        TIMESTAMP_COL is not None
        and TIMESTAMP_COL in data.columns
    ):

        try:

            data = data.sort_values(
                TIMESTAMP_COL,
                ascending=False
            )

        except Exception:
            pass

    return data


def get_user_seen_items(user_id):

    history = get_user_history(
        user_id
    )

    if history.empty:
        return set()

    return set(
        history[ITEM_COL]
        .astype(str)
        .tolist()
    )


# ============================================================
# CONTENT SCORES
# ============================================================

def get_content_scores(user_id):

    scores = np.zeros(
        len(items)
    )

    history = get_user_history(
        user_id
    )

    if history.empty:
        return scores

    seen_items = set(
        history[ITEM_COL]
        .astype(str)
        .tolist()
    )

    valid_count = 0

    for item_id in seen_items:

        idx = get_item_index(
            item_id
        )

        if idx is not None:

            scores += (
                content_similarity[idx]
            )

            valid_count += 1

    if valid_count > 0:

        scores /= valid_count

    return scores


# ============================================================
# COLLABORATIVE SCORES
# ============================================================

def get_collaborative_scores(user_id):

    scores = np.zeros(
        len(items)
    )

    user_id = str(user_id)

    if user_id not in user_index:
        return scores

    user_row = interaction_matrix[
        user_index[user_id]
    ]

    user_items = (
        user_row.nonzero()[1]
    )

    if len(user_items) == 0:
        return scores

    for idx in user_items:

        scores += (
            collaborative_similarity[idx]
        )

    scores /= len(user_items)

    return scores


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(scores) == 0:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum - minimum == 0:

        return np.zeros_like(
            scores
        )

    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendations(
    user_id,
    alpha=0.50,
    top_k=10
):

    content_scores = (
        get_content_scores(
            user_id
        )
    )

    collaborative_scores = (
        get_collaborative_scores(
            user_id
        )
    )

    content_scores = normalize_scores(
        content_scores
    )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    hybrid_scores = (
        alpha * collaborative_scores
        +
        (1 - alpha) * content_scores
    )

    seen_items = (
        get_user_seen_items(
            user_id
        )
    )

    result = items.copy()

    result["content_score"] = (
        content_scores
    )

    result["collaborative_score"] = (
        collaborative_scores
    )

    result["hybrid_score"] = (
        hybrid_scores
    )

    # Remove products already interacted with
    result = result[
        ~result[ITEM_ID_COL]
        .astype(str)
        .isin(seen_items)
    ]

    result = result.sort_values(
        "hybrid_score",
        ascending=False
    )

    return result.head(
        top_k
    ).copy()


# ============================================================
# RECOMMENDATION EXPLANATION
# ============================================================

def explain_recommendation(
    product_row,
    alpha
):

    content_score = float(
        product_row["content_score"]
    )

    collaborative_score = float(
        product_row["collaborative_score"]
    )

    hybrid_score = float(
        product_row["hybrid_score"]
    )

    if content_score > collaborative_score:

        reason = (
            "The recommendation is mainly driven "
            "by similarity between this product's "
            "title/category/brand information and "
            "products previously interacted with."
        )

    elif collaborative_score > content_score:

        reason = (
            "The recommendation is mainly driven "
            "by collaborative interaction patterns "
            "from the user-item matrix."
        )

    else:

        reason = (
            "Both content similarity and collaborative "
            "signals contribute similarly to this result."
        )

    return {
        "reason": reason,
        "content": content_score,
        "collaborative": collaborative_score,
        "hybrid": hybrid_score,
        "alpha": alpha
    }


# ============================================================
# PRODUCT SEARCH
# ============================================================

def search_products(query):

    query = str(query).strip().lower()

    if not query:
        return pd.DataFrame()

    # Exact ID search first
    exact = items[
        items[ITEM_ID_COL]
        .astype(str)
        .str.lower()
        == query
    ]

    if not exact.empty:
        return exact.copy()

    # Safe substring search
    try:

        mask = (
            items["_search_text"]
            .str.lower()
            .str.contains(
                re.escape(query),
                na=False,
                regex=True
            )
        )

    except Exception:

        mask = pd.Series(
            False,
            index=items.index
        )

    return items[
        mask
    ].copy()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛍️ ShopSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'E-Commerce Hybrid Recommendation & Product Discovery System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧭 Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "🤖 Recommendations",
        "📷 Product Scanner",
        "🛒 Price Comparison",
        "👤 User History",
        "📊 Analytics",
        "🏗️ Architecture"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "ShopSense AI"
)

st.sidebar.caption(
    "Hybrid Recommendation System"
)

st.sidebar.caption(
    "Python • Machine Learning • Streamlit"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "📊 System Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Users",
            f"{len(users):,}"
        )

    with col2:

        st.metric(
            "🛍️ Products",
            f"{len(items):,}"
        )

    with col3:

        st.metric(
            "🔄 Interactions",
            f"{len(interactions):,}"
        )

    with col4:

        st.metric(
            "🧠 Model",
            "Hybrid"
        )

    st.divider()

    st.subheader(
        "How ShopSense AI Works"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info(
            """
            ### 🤝 Collaborative Filtering

            Learns from user-item interaction
            patterns and identifies products
            related to previous activity.
            """
        )

    with c2:

        st.info(
            """
            ### 📝 Content-Based Filtering

            Uses product title, category,
            brand and product information
            with TF-IDF similarity.
            """
        )

    with c3:

        st.info(
            """
            ### 🔀 Hybrid Model

            Combines collaborative and
            content-based signals using
            an adjustable Alpha parameter.
            """
        )

    st.divider()

    st.subheader(
        "📈 Dataset Summary"
    )

    summary = pd.DataFrame(
        {
            "Component": [
                "Users",
                "Products",
                "Interactions",
                "Training Interactions",
                "Test Interactions"
            ],
            "Count": [
                len(users),
                len(items),
                len(interactions),
                len(train_df),
                len(test_df)
            ]
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🤖 Recommendations":

    st.header(
        "🤖 Personalized Recommendations"
    )

    all_users = sorted(
        interactions[USER_COL]
        .astype(str)
        .unique()
    )

    if not all_users:

        st.warning(
            "No users found."
        )

    else:

        selected_user = st.selectbox(
            "Select User",
            all_users,
            key="recommendation_user_select"
        )

        alpha = st.slider(
            "Collaborative Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05
        )

        st.caption(
            f"Collaborative: {alpha:.0%}  |  "
            f"Content-Based: {(1-alpha):.0%}"
        )

        if st.button(
            "🚀 Generate Recommendations",
            use_container_width=True
        ):

            recommendations = (
                generate_recommendations(
                    selected_user,
                    alpha=alpha,
                    top_k=10
                )
            )

            st.session_state[
                "recommendations"
            ] = recommendations

            st.session_state[
                "recommendation_user"
            ] = selected_user

            st.session_state[
                "recommendation_alpha"
            ] = alpha

        if (
            "recommendations"
            in st.session_state
        ):

            recommendations = (
                st.session_state[
                    "recommendations"
                ]
            )

            selected_user = (
                st.session_state[
                    "recommendation_user"
                ]
            )

            alpha = (
                st.session_state[
                    "recommendation_alpha"
                ]
            )

            st.success(
                f"Recommendations generated for "
                f"**{selected_user}**"
            )

            st.subheader(
                "🏆 Top 10 Products"
            )

            for rank, (_, product) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                title = (
                    str(product[TITLE_COL])
                    if TITLE_COL
                    else str(product[ITEM_ID_COL])
                )

                category = (
                    str(product[CATEGORY_COL])
                    if CATEGORY_COL
                    else "Unknown"
                )

                brand = (
                    str(product[BRAND_COL])
                    if BRAND_COL
                    else "Unknown"
                )

                st.markdown(
                    f"""
                    <div class="product-card">

                    <h3>
                    #{rank} &nbsp; {title}
                    </h3>

                    <p>
                    <b>Product ID:</b>
                    {product[ITEM_ID_COL]}
                    &nbsp; | &nbsp;

                    <b>Category:</b>
                    {category}
                    &nbsp; | &nbsp;

                    <b>Brand:</b>
                    {brand}
                    </p>

                    <p>
                    <b>Hybrid Score:</b>
                    {product["hybrid_score"]:.4f}
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander(
                    "🔍 Why was this recommended?"
                ):

                    explanation = (
                        explain_recommendation(
                            product,
                            alpha
                        )
                    )

                    st.write(
                        explanation["reason"]
                    )

                    e1, e2, e3 = st.columns(3)

                    with e1:

                        st.metric(
                            "Content Score",
                            f'{explanation["content"]:.4f}'
                        )

                    with e2:

                        st.metric(
                            "Collaborative Score",
                            f'{explanation["collaborative"]:.4f}'
                        )

                    with e3:

                        st.metric(
                            "Hybrid Score",
                            f'{explanation["hybrid"]:.4f}'
                        )


# ============================================================
# PRODUCT SCANNER
# ============================================================

elif page == "📷 Product Scanner":

    st.header(
        "📷 Product Scanner"
    )

    st.write(
        "Scan a QR/product code using your "
        "device camera and search the catalog."
    )

    st.info(
        "📱 Camera permission is required."
    )

    if not SCANNER_AVAILABLE:

        st.error(
            "Scanner module is not available."
        )

        st.code(
            "src/scanner/barcode_scanner.py"
        )

    else:

        camera_image = st.camera_input(
            "📷 Scan Product"
        )

        if camera_image is not None:

            with st.spinner(
                "🔍 Scanning..."
            ):

                try:

                    detected_code, _ = (
                        decode_qr(
                            camera_image.getvalue()
                        )
                    )

                except Exception as e:

                    detected_code = None

                    st.error(
                        "Scanner error:"
                    )

                    st.code(str(e))

            if detected_code:

                st.success(
                    "✅ Code detected!"
                )

                st.subheader(
                    "Detected Code"
                )

                st.code(
                    detected_code
                )

                st.session_state[
                    "scanned_code"
                ] = detected_code

                search_result = (
                    search_products(
                        detected_code
                    )
                )

                if not search_result.empty:

                    st.success(
                        f"Found {len(search_result)} "
                        "matching product(s)."
                    )

                    display_columns = []

                    for col in [
                        ITEM_ID_COL,
                        TITLE_COL,
                        CATEGORY_COL,
                        BRAND_COL
                    ]:

                        if (
                            col
                            and col in search_result.columns
                        ):

                            display_columns.append(
                                col
                            )

                    st.dataframe(
                        search_result[
                            display_columns
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning(
                        "Code detected, but this code "
                        "does not exist in the current "
                        "local product catalog."
                    )

                    st.info(
                        "The detected code can later be "
                        "connected to external marketplace "
                        "or product APIs."
                    )

            else:

                st.warning(
                    "❌ No QR code detected."
                )

                st.caption(
                    "Keep the code inside the camera "
                    "frame, improve lighting and try again."
                )

    st.divider()

    st.subheader(
        "🔎 Manual Product Search"
    )

    manual_query = st.text_input(
        "Search by product ID, name, category or brand"
    )

    if manual_query:

        results = search_products(
            manual_query
        )

        if results.empty:

            st.warning(
                "No matching products found."
            )

        else:

            display_columns = []

            for col in [
                ITEM_ID_COL,
                TITLE_COL,
                CATEGORY_COL,
                BRAND_COL
            ]:

                if (
                    col
                    and col in results.columns
                ):

                    display_columns.append(
                        col
                    )

            st.dataframe(
                results[
                    display_columns
                ].head(20),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# MARKETPLACE PRICE COMPARISON
# ============================================================

elif page == "🛒 Price Comparison":
    st.header("🛒 Amazon + Flipkart Price Comparison")
    st.write(
        "Search a product by barcode, product name or model. "
        "Configured marketplace APIs can return live catalog, price, "
        "availability and affiliate/product links."
    )

    query = st.text_input(
        "🔎 Product / Barcode / Model",
        value=st.session_state.get("scanned_code", ""),
        placeholder="Example: 8900000000001 or Sony WH-1000XM5"
    )

    if not MARKETPLACE_AVAILABLE:
        st.error(
            "Marketplace integration files are missing. "
            "Expected: src/marketplace/marketplace_api.py"
        )
    elif st.button("🔍 Compare Amazon & Flipkart", use_container_width=True):
        if not query.strip():
            st.warning("Enter a barcode, product name or model first.")
        else:
            with st.spinner("Fetching marketplace data..."):
                try:
                    marketplace_results = compare_marketplaces(query.strip())
                    st.session_state["marketplace_results"] = marketplace_results
                    st.session_state["marketplace_query"] = query.strip()
                except Exception as e:
                    st.error("Marketplace lookup failed.")
                    st.code(str(e))

    if "marketplace_results" in st.session_state:
        marketplace_results = st.session_state["marketplace_results"]

        st.caption(f"Search: {st.session_state.get('marketplace_query', '')}")

        if not marketplace_results:
            st.warning(
                "No marketplace result was returned. Check API credentials/access "
                "or try a product name."
            )
        else:
            rows = []
            for result in marketplace_results:
                rows.append({
                    "Marketplace": result.get("marketplace", ""),
                    "Product": result.get("title", "Unknown"),
                    "Price": result.get("price_display", "Not available"),
                    "Availability": result.get("availability", "Unknown"),
                    "Offer": result.get("offer", "—"),
                    "Configured": "Yes" if result.get("configured") else "No",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.subheader("🛍️ Marketplace Offers")

            for result in marketplace_results:
                name = result.get("marketplace", "Marketplace")
                title = result.get("title", "Product")
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2.2, 1, 1])
                    with c1:
                        st.markdown(f"### {name}")
                        st.write(title)
                        st.caption(result.get("message", ""))
                    with c2:
                        st.metric("Price", result.get("price_display", "N/A"))
                        st.write(f"**Stock:** {result.get('availability', 'Unknown')}")
                    with c3:
                        offer = result.get("offer")
                        if offer:
                            st.write(f"🎁 **Offer:** {offer}")
                        url = result.get("url")
                        if url:
                            st.link_button(f"🛒 Open {name}", url, use_container_width=True)

            st.info(
                "Prices and availability can change. Displayed marketplace data "
                "should be treated as a live API response at lookup time."
            )

            st.divider()
            st.subheader("🤖 Existing ML Recommendations")
            local_matches = search_products(st.session_state.get("marketplace_query", ""))

            if not local_matches.empty:
                local_item_id = str(local_matches.iloc[0][ITEM_ID_COL])
                st.success(f"Local catalog match found: {local_item_id}")
                all_users_for_ml = sorted(interactions[USER_COL].astype(str).unique())
                if all_users_for_ml:
                    ml_user = st.selectbox("User for ML recommendations", all_users_for_ml, key="marketplace_ml_user")
                    alpha_ml = st.slider("Hybrid Alpha", 0.0, 1.0, 0.0, 0.05, key="marketplace_ml_alpha")
                    recs = generate_recommendations(ml_user, alpha=alpha_ml, top_k=5)
                    if not recs.empty:
                        display_rec_cols = [c for c in [ITEM_ID_COL, TITLE_COL, CATEGORY_COL, BRAND_COL, "content_score", "collaborative_score", "hybrid_score"] if c and c in recs.columns]
                        st.dataframe(recs[display_rec_cols], use_container_width=True, hide_index=True)
            else:
                st.caption(
                    "The scanned/searched product is not in the synthetic local catalog, "
                    "so only marketplace discovery is shown."
                )


# ============================================================
# USER HISTORY
# ============================================================

elif page == "👤 User History":

    st.header(
        "👤 User Interaction History"
    )

    all_users = sorted(
        interactions[USER_COL]
        .astype(str)
        .unique()
    )

    selected_user = st.selectbox(
        "Select User",
        all_users,
        key="history_user"
    )

    history = get_user_history(
        selected_user
    )

    st.metric(
        "Total Interactions",
        len(history)
    )

    if history.empty:

        st.info(
            "No interaction history found."
        )

    else:

        history_display = history.copy()

        if ITEM_COL and ITEM_ID_COL:

            history_display = (
                history_display.merge(
                    items,
                    left_on=ITEM_COL,
                    right_on=ITEM_ID_COL,
                    how="left",
                    suffixes=(
                        "",
                        "_product"
                    )
                )
            )

        columns_to_show = []

        for col in [
            ITEM_COL,
            TITLE_COL,
            CATEGORY_COL,
            BRAND_COL,
            EVENT_COL,
            TIMESTAMP_COL
        ]:

            if (
                col
                and col in history_display.columns
            ):

                if col not in columns_to_show:

                    columns_to_show.append(
                        col
                    )

        if columns_to_show:

            st.dataframe(
                history_display[
                    columns_to_show
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.dataframe(
                history_display,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.header(
        "📊 Model Analytics"
    )

    comparison = (
        load_model_comparison()
    )

    alpha_results = (
        load_alpha_results()
    )

    if comparison.empty:

        st.warning(
            "reports/model_comparison.csv "
            "not found."
        )

    else:

        st.subheader(
            "🏆 Model Comparison"
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        metric_options = [
            "Precision@10",
            "Recall@10",
            "MAP@10",
            "NDCG@10"
        ]

        available_metrics = [
            m
            for m in metric_options
            if m in comparison.columns
        ]

        if available_metrics:

            selected_metric = st.selectbox(
                "Select Metric",
                available_metrics,
                key="model_metric"
            )

            chart_data = (
                comparison[
                    [
                        "Model",
                        selected_metric
                    ]
                ]
                .set_index("Model")
            )

            st.bar_chart(
                chart_data
            )

    st.divider()

    if not alpha_results.empty:

        st.subheader(
            "🎛️ Alpha Tuning"
        )

        st.dataframe(
            alpha_results,
            use_container_width=True,
            hide_index=True
        )

        alpha_metric_options = [
            "Precision@10",
            "Recall@10",
            "MAP@10",
            "NDCG@10"
        ]

        available_alpha_metrics = [
            m
            for m in alpha_metric_options
            if m in alpha_results.columns
        ]

        if available_alpha_metrics:

            selected_alpha_metric = (
                st.selectbox(
                    "Alpha Performance Metric",
                    available_alpha_metrics,
                    key="alpha_metric"
                )
            )

            if "Alpha" in alpha_results.columns:

                alpha_chart = (
                    alpha_results[
                        [
                            "Alpha",
                            selected_alpha_metric
                        ]
                    ]
                    .set_index("Alpha")
                )

                st.line_chart(
                    alpha_chart
                )


# ============================================================
# ARCHITECTURE
# ============================================================

elif page == "🏗️ Architecture":

    st.header(
        "🏗️ System Architecture"
    )

    st.markdown(
        """
```text
                    ┌──────────────────────────┐
                    │       USER / CLIENT       │
                    │    Streamlit Dashboard    │
                    └────────────┬─────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
      ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
      │    User     │    │Recommendation│    │   Product   │
      │   History   │    │    Engine    │    │   Scanner   │
      └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
             │                  │                   │
             │          ┌───────┴────────┐          │
             │          │                │          │
             │          ▼                ▼          │
             │   ┌──────────────┐ ┌──────────────┐  │
             │   │ Collaborative│ │ Content-Based│  │
             │   │  Filtering   │ │    TF-IDF    │  │
             │   └──────┬───────┘ └──────┬───────┘  │
             │          │                │           │
             │          └───────┬────────┘           │
             │                  ▼                    │
             │         ┌─────────────────┐           │
             └────────►│ Hybrid Ranking  │◄──────────┘
                       │ Alpha Weighted  │
                       └────────┬────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Personalized Product│
                     │ Recommendations     │
                     └─────────────────────┘"""
    )
st.subheader(
    "🧠 Hybrid Recommendation Formula"
)

st.latex(
    r"""
    HybridScore =
    \alpha \times CollaborativeScore
    +
    (1-\alpha) \times ContentScore
    """
)

st.write(
    """
    **Alpha** controls the contribution of
    collaborative filtering.

    • Alpha = 0 → Content-Based only

    • Alpha = 0.5 → Balanced combination

    • Alpha = 1 → Collaborative only
    """
)

st.divider()

st.subheader(
    "🔧 Technology Stack"
)

tech = pd.DataFrame(
    {
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "SciPy",
            "Scikit-learn",
            "OpenCV / QR Scanner",
            "Streamlit",
            "CSV / JSON / NPZ"
        ],
        "Purpose": [
            "Application development",
            "Data processing",
            "Numerical computation",
            "Sparse matrices",
            "Machine Learning",
            "Product scanning",
            "Web dashboard",
            "Data storage"
        ]
    }
)

st.dataframe(
    tech,
    use_container_width=True,
    hide_index=True
)

#============================================================
#FOOTER
#============================================================

st.divider()

st.caption(
"🛍️ ShopSense AI • E-Commerce Hybrid Recommendation System"
)

st.caption(
"Built with Python • Machine Learning • Streamlit"
)