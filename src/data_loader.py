from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


LOGGER = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """
    Configuration for synthetic e-commerce data generation.
    """

    num_users: int = 1_000
    num_items: int = 500
    min_interactions_per_user: int = 8
    max_interactions_per_user: int = 40
    test_ratio: float = 0.2
    random_seed: int = 42
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"


CATEGORIES = {
    "electronics": [
        "wireless headphones",
        "smartphone",
        "mechanical keyboard",
        "computer monitor",
        "portable speaker",
        "fitness tracker",
        "webcam",
        "tablet",
    ],
    "home": [
        "desk lamp",
        "coffee maker",
        "air purifier",
        "storage organizer",
        "vacuum cleaner",
        "bedding set",
        "kitchen scale",
        "water bottle",
    ],
    "fashion": [
        "running shoes",
        "hoodie",
        "denim jacket",
        "cotton t-shirt",
        "winter coat",
        "backpack",
        "sunglasses",
        "wrist watch",
    ],
    "beauty": [
        "face moisturizer",
        "shampoo",
        "perfume",
        "skin cleanser",
        "hair dryer",
        "lip balm",
        "body lotion",
        "sunscreen",
    ],
    "sports": [
        "yoga mat",
        "dumbbell set",
        "cycling helmet",
        "tennis racket",
        "resistance bands",
        "hiking backpack",
        "football",
        "sports bottle",
    ],
    "books": [
        "data science book",
        "business strategy book",
        "science fiction novel",
        "history book",
        "personal finance book",
        "cooking book",
        "psychology book",
        "programming book",
    ],
}

BRANDS = [
    "Nova",
    "Vertex",
    "Apex",
    "Orion",
    "Zenith",
    "Pulse",
    "Lumina",
    "Atlas",
    "Nexa",
    "Summit",
]

COLORS = [
    "black",
    "white",
    "blue",
    "red",
    "green",
    "gray",
    "silver",
    "brown",
]

INTERACTION_TYPES = {
    "impression": 0.05,
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 4.0,
    "wishlist": 3.5,
    "purchase": 8.0,
}


def configure_logging() -> None:
    """
    Configure application logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_random_seeds(seed: int) -> None:
    """
    Set deterministic random seeds for reproducible data generation.
    """

    random.seed(seed)
    np.random.seed(seed)


def slugify(value: str) -> str:
    """
    Convert a string into a simple lowercase identifier.
    """

    return value.lower().replace(" ", "_").replace("-", "_")


def generate_item_catalog(
    config: DataConfig,
) -> pd.DataFrame:
    """
    Generate synthetic item metadata.

    Returns
    -------
    pandas.DataFrame
        Item catalog containing IDs, categories, titles, descriptions,
        brands, prices, colors, and image URLs.
    """

    rng = np.random.default_rng(config.random_seed)

    category_names = list(CATEGORIES.keys())
    item_rows = []

    for item_index in range(config.num_items):
        category = category_names[item_index % len(category_names)]
        product_type = rng.choice(CATEGORIES[category])
        brand = rng.choice(BRANDS)
        color = rng.choice(COLORS)

        title = f"{brand} {color.title()} {product_type.title()}"

        description = (
            f"{title} designed for {category} enthusiasts. "
            f"This reliable {product_type} features modern styling, "
            f"durable materials, practical performance, and comfortable everyday use. "
            f"Suitable for home, travel, work, and personal activities."
        )

        item_rows.append(
            {
                "item_id": f"item_{item_index:05d}",
                "category": category,
                "product_type": product_type,
                "brand": brand,
                "color": color,
                "title": title,
                "description": description,
                "price": round(float(rng.uniform(9.99, 999.99)), 2),
                "image_url": (
                    "https://images.unsplash.com/"
                    f"photo-{1000000000000 + item_index}"
                    "?auto=format&fit=crop&w=400&q=80"
                ),
            }
        )

    items = pd.DataFrame(item_rows)

    LOGGER.info(
        "Generated item catalog with %d items across %d categories.",
        len(items),
        items["category"].nunique(),
    )

    return items


def _sample_user_preferences(
    rng: np.random.Generator,
    categories: list[str],
) -> Dict[str, float]:
    """
    Generate category preference weights for one synthetic user.
    """

    raw_preferences = rng.gamma(shape=1.5, scale=1.0, size=len(categories))
    normalized_preferences = raw_preferences / raw_preferences.sum()

    return {
        category: float(weight)
        for category, weight in zip(categories, normalized_preferences)
    }


def _sample_interaction_type(
    rng: np.random.Generator,
) -> str:
    """
    Sample an implicit interaction event using realistic event frequencies.
    """

    interaction_types = [
        "impression",
        "view",
        "click",
        "add_to_cart",
        "wishlist",
        "purchase",
    ]

    probabilities = [
        0.34,
        0.30,
        0.16,
        0.08,
        0.05,
        0.07,
    ]

    return str(rng.choice(interaction_types, p=probabilities))


def generate_interactions(
    users: pd.DataFrame,
    items: pd.DataFrame,
    config: DataConfig,
) -> pd.DataFrame:
    """
    Generate synthetic user-item interaction events.

    The generator creates user-specific category preferences so that the
    resulting dataset contains learnable collaborative and content signals.

    Returns
    -------
    pandas.DataFrame
        Interaction-level dataset.
    """

    rng = np.random.default_rng(config.random_seed + 1)
    categories = sorted(items["category"].unique())

    item_lookup = items.set_index("item_id")
    item_ids = items["item_id"].tolist()

    start_timestamp = pd.Timestamp(config.start_date)
    end_timestamp = pd.Timestamp(config.end_date)

    total_seconds = int(
        (end_timestamp - start_timestamp).total_seconds()
    )

    interaction_rows = []

    for user_id in users["user_id"]:
        preference_weights = _sample_user_preferences(rng, categories)

        number_of_interactions = int(
            rng.integers(
                config.min_interactions_per_user,
                config.max_interactions_per_user + 1,
            )
        )

        for _ in range(number_of_interactions):
            preferred_category = rng.choice(
                categories,
                p=[
                    preference_weights[category]
                    for category in categories
                ],
            )

            category_item_ids = items.loc[
                items["category"] == preferred_category,
                "item_id",
            ].tolist()

            if rng.random() < 0.85 and category_item_ids:
                item_id = str(rng.choice(category_item_ids))
            else:
                item_id = str(rng.choice(item_ids))

            interaction_type = _sample_interaction_type(rng)

            event_offset_seconds = int(
                rng.integers(0, total_seconds + 1)
            )

            timestamp = start_timestamp + pd.to_timedelta(
                event_offset_seconds,
                unit="s",
            )

            item_row = item_lookup.loc[item_id]

            interaction_rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "category": item_row["category"],
                    "interaction_type": interaction_type,
                    "implicit_rating": INTERACTION_TYPES[
                        interaction_type
                    ],
                    "timestamp": timestamp,
                }
            )

    interactions = pd.DataFrame(interaction_rows)

    interactions = interactions.sort_values(
        by=["timestamp", "user_id", "item_id"]
    ).reset_index(drop=True)

    LOGGER.info(
        "Generated %d interaction events for %d users.",
        len(interactions),
        interactions["user_id"].nunique(),
    )

    return interactions


def generate_users(
    config: DataConfig,
) -> pd.DataFrame:
    """
    Generate synthetic user metadata.
    """

    rng = np.random.default_rng(config.random_seed + 2)

    user_rows = []

    for user_index in range(config.num_users):
        user_rows.append(
            {
                "user_id": f"user_{user_index:05d}",
                "age": int(rng.integers(18, 76)),
                "membership_tier": str(
                    rng.choice(
                        ["standard", "silver", "gold", "platinum"],
                        p=[0.55, 0.25, 0.15, 0.05],
                    )
                ),
                "region": str(
                    rng.choice(
                        [
                            "northeast",
                            "southeast",
                            "midwest",
                            "southwest",
                            "west",
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(user_rows)


def clean_interactions(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and clean raw interactions.

    Processing steps:
    - Cast identifiers to strings.
    - Parse timestamps.
    - Remove missing records.
    - Remove invalid ratings.
    - Remove exact duplicate events.
    """

    required_columns = {
        "user_id",
        "item_id",
        "category",
        "interaction_type",
        "implicit_rating",
        "timestamp",
    }

    missing_columns = required_columns.difference(interactions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required interaction columns: {sorted(missing_columns)}"
        )

    cleaned = interactions.copy()

    cleaned["user_id"] = cleaned["user_id"].astype(str)
    cleaned["item_id"] = cleaned["item_id"].astype(str)
    cleaned["category"] = cleaned["category"].astype(str)
    cleaned["interaction_type"] = cleaned[
        "interaction_type"
    ].astype(str)

    cleaned["implicit_rating"] = pd.to_numeric(
        cleaned["implicit_rating"],
        errors="coerce",
    )

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        errors="coerce",
        utc=True,
    )

    cleaned = cleaned.dropna(
        subset=[
            "user_id",
            "item_id",
            "interaction_type",
            "implicit_rating",
            "timestamp",
        ]
    )

    cleaned = cleaned.loc[
        cleaned["implicit_rating"] > 0
    ]

    cleaned = cleaned.drop_duplicates(
        subset=[
            "user_id",
            "item_id",
            "interaction_type",
            "timestamp",
        ]
    )

    cleaned = cleaned.sort_values(
        by=["timestamp", "user_id", "item_id"]
    ).reset_index(drop=True)

    LOGGER.info(
        "Cleaned interactions: %d rows remaining.",
        len(cleaned),
    )

    return cleaned


def time_based_train_test_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a per-user chronological train-test split.

    The latest interactions for every user are assigned to the test set.
    This prevents future interactions from leaking into model training.

    Parameters
    ----------
    interactions:
        Cleaned interaction dataframe.

    test_ratio:
        Fraction of each user's interactions reserved for testing.

    Returns
    -------
    train_interactions, test_interactions
    """

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    sorted_interactions = interactions.sort_values(
        by=["user_id", "timestamp"]
    ).reset_index(drop=True)

    train_parts = []
    test_parts = []

    for user_id, user_interactions in sorted_interactions.groupby(
        "user_id",
        sort=False,
    ):
        user_interactions = user_interactions.sort_values(
            "timestamp"
        )

        number_of_rows = len(user_interactions)

        if number_of_rows < 2:
            train_parts.append(user_interactions)
            continue

        number_of_test_rows = max(
            1,
            int(np.ceil(number_of_rows * test_ratio)),
        )

        number_of_test_rows = min(
            number_of_test_rows,
            number_of_rows - 1,
        )

        split_index = number_of_rows - number_of_test_rows

        train_parts.append(
            user_interactions.iloc[:split_index]
        )

        test_parts.append(
            user_interactions.iloc[split_index:]
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    if test_parts:
        test = pd.concat(
            test_parts,
            ignore_index=True,
        )
    else:
        test = pd.DataFrame(
            columns=interactions.columns
        )

    train = train.sort_values("timestamp").reset_index(drop=True)
    test = test.sort_values("timestamp").reset_index(drop=True)

    LOGGER.info(
        "Time-based split complete: train=%d, test=%d.",
        len(train),
        len(test),
    )

    return train, test


def build_user_item_matrix(
    interactions: pd.DataFrame,
    user_mapping: Dict[str, int] | None = None,
    item_mapping: Dict[str, int] | None = None,
) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
    """
    Convert interactions into a sparse user-item matrix.

    Multiple events for the same user-item pair are aggregated by summing
    their implicit ratings.

    Returns
    -------
    matrix:
        CSR sparse matrix with shape:
        (number_of_users, number_of_items)

    user_mapping:
        Mapping from user ID to matrix row index.

    item_mapping:
        Mapping from item ID to matrix column index.
    """

    if interactions.empty:
        raise ValueError(
            "Cannot create a matrix from an empty dataframe."
        )

    if user_mapping is None:
        user_ids = sorted(
            interactions["user_id"].astype(str).unique()
        )

        user_mapping = {
            user_id: index
            for index, user_id in enumerate(user_ids)
        }

    if item_mapping is None:
        item_ids = sorted(
            interactions["item_id"].astype(str).unique()
        )

        item_mapping = {
            item_id: index
            for index, item_id in enumerate(item_ids)
        }

    matrix_data = interactions.copy()

    matrix_data["user_index"] = matrix_data["user_id"].map(
        user_mapping
    )

    matrix_data["item_index"] = matrix_data["item_id"].map(
        item_mapping
    )

    matrix_data = matrix_data.dropna(
        subset=["user_index", "item_index"]
    )

    matrix = csr_matrix(
        (
            matrix_data["implicit_rating"].astype(float),
            (
                matrix_data["user_index"].astype(int),
                matrix_data["item_index"].astype(int),
            ),
        ),
        shape=(
            len(user_mapping),
            len(item_mapping),
        ),
    )

    matrix.sum_duplicates()

    LOGGER.info(
        "Created sparse user-item matrix with shape=%s and %d non-zero values.",
        matrix.shape,
        matrix.nnz,
    )

    return matrix, user_mapping, item_mapping


def save_json(
    payload: Dict[str, int],
    output_path: Path,
) -> None:
    """
    Save a mapping dictionary as JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )


def save_pipeline_outputs(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    matrix: csr_matrix,
    user_mapping: Dict[str, int],
    item_mapping: Dict[str, int],
    output_dir: Path,
) -> None:
    """
    Save all generated datasets and model-ready artifacts.
    """

    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    users.to_csv(
        raw_dir / "users.csv",
        index=False,
    )

    items.to_csv(
        raw_dir / "items.csv",
        index=False,
    )

    interactions.to_csv(
        raw_dir / "interactions.csv",
        index=False,
    )

    train.to_csv(
        processed_dir / "train_interactions.csv",
        index=False,
    )

    test.to_csv(
        processed_dir / "test_interactions.csv",
        index=False,
    )

    save_npz(
        processed_dir / "user_item_matrix.npz",
        matrix,
    )

    save_json(
        user_mapping,
        processed_dir / "user_mapping.json",
    )

    save_json(
        item_mapping,
        processed_dir / "item_mapping.json",
    )

    metadata = {
        "num_users": len(user_mapping),
        "num_items": len(item_mapping),
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "matrix_non_zero_values": int(matrix.nnz),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }

    with (processed_dir / "dataset_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    LOGGER.info(
        "Saved all pipeline outputs to %s.",
        output_dir,
    )


def run_pipeline(
    config: DataConfig,
    output_dir: Path,
) -> None:
    """
    Execute the complete synthetic data pipeline.
    """

    set_random_seeds(config.random_seed)

    users = generate_users(config)
    items = generate_item_catalog(config)

    interactions = generate_interactions(
        users=users,
        items=items,
        config=config,
    )

    interactions = clean_interactions(interactions)

    train, test = time_based_train_test_split(
        interactions=interactions,
        test_ratio=config.test_ratio,
    )

    matrix, user_mapping, item_mapping = build_user_item_matrix(
        interactions=train,
    )

    save_pipeline_outputs(
        users=users,
        items=items,
        interactions=interactions,
        train=train,
        test=test,
        matrix=matrix,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate and preprocess synthetic e-commerce "
            "recommendation data."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where generated files will be saved.",
    )

    parser.add_argument(
        "--num-users",
        type=int,
        default=1_000,
        help="Number of synthetic users.",
    )

    parser.add_argument(
        "--num-items",
        type=int,
        default=500,
        help="Number of synthetic products.",
    )

    parser.add_argument(
        "--min-interactions",
        type=int,
        default=8,
        help="Minimum interactions per user.",
    )

    parser.add_argument(
        "--max-interactions",
        type=int,
        default=40,
        help="Maximum interactions per user.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of each user's events assigned to test.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    configure_logging()

    args = parse_args()

    config = DataConfig(
        num_users=args.num_users,
        num_items=args.num_items,
        min_interactions_per_user=args.min_interactions,
        max_interactions_per_user=args.max_interactions,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )

    run_pipeline(
        config=config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

