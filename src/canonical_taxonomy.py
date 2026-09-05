from __future__ import annotations


CANONICAL_TAXONOMY = {

    2001: {
        "business_name": "Dining",
        "subcategories": [
            "Restaurants",
            "Fast Food",
            "Meals",
        ],
        "seed_terms": [
            "food",
            "dinner",
            "lunch",
            "breakfast",
            "roti",
            "biryani",
            "pizza",
            "burger",
            "kfc",
            "foodpanda",
            "mcdonald",
            "mcdonalds",
            "restaurant",
            "meal",
            "khana",
        ],
    },

    2002: {
        "business_name": "Fuel",
        "subcategories": [
            "Petrol",
            "Diesel",
            "CNG",
        ],
        "seed_terms": [
            "petrol",
            "fuel",
            "diesel",
            "gasoline",
            "cng",
            "benzine",
            "petroleum",
        ],
    },

    2003: {
        "business_name": "Transport",
        "subcategories": [
            "Ride-sharing",
            "Parking",
            "Tolls",
            "Taxi",
        ],
        "seed_terms": [
            "careem",
            "uber",
            "parking",
            "taxi",
            "toll",
            "toll tax",
        ],
    },

    2004: {
        "business_name": "Utilities & Bills",
        "subcategories": [
            "Electricity",
            "Water",
            "Internet/Telecom",
        ],
        "seed_terms": [
            "electricity",
            "electricity bill",
            "water bill",
            "gas bill",
            "ptcl",
            "cable",
            "utility",
            "bill",
        ],
    },

    2005: {
        "business_name": "Healthcare",
        "subcategories": [
            "Medicine",
            "Doctor",
            "Hospital",
        ],
        "seed_terms": [
            "medicine",
            "medicines",
            "doctor",
            "hospital",
            "medical",
            "clinic",
        ],
    },

    2006: {
        "business_name": "Shopping & Retail",
        "subcategories": [
            "Clothing",
            "General Shopping",
            "Retail",
        ],
        "seed_terms": [
            "shoes",
            "clothes",
            "clothing",
            "shopping",
            "flipkart",
            "daraz",
            "mall",
        ],
    },

    2007: {
        "business_name": "Transfers & Withdrawals",
        "subcategories": [
            "Cash Withdrawal",
            "Transfers",
            "ATM",
        ],
        "seed_terms": [
            "cash",
            "atm",
            "transfer",
            "withdrawal",
            "withdrawl",
            "cash withdrawal",
            "cash withdrawl",
        ],
    },

    2008: {
        "business_name": "Income",
        "subcategories": [
            "Salary",
            "Interest",
            "Deposits",
        ],
        "seed_terms": [
            "salary",
            "interest earned",
            "online deposit",
            "deposit",
            "income",
        ],
    },

    2009: {
        "business_name": "Groceries",
        "subcategories": [
            "Groceries",
            "Bakery/Dairy",
            "Vegetables/Fruits",
        ],
        "seed_terms": [
            "sabzi",
            "vegetables",
            "vegetable",
            "doodh",
            "milk",
            "bread",
            "eggs",
            "egg",
            "fruit",
            "fruits",
            "grocery",
            "groceries",
            "kirana",
            "lauka",
            "karela",
            "bajra",
            "dahi",
            "yogurt",
        ],
    },

    2010: {
        "business_name": "Other",
        "subcategories": [
            "Miscellaneous",
        ],
        "seed_terms": [],
    },
}


def get_canonical_taxonomy() -> dict:
    """Return the complete canonical taxonomy."""
    return CANONICAL_TAXONOMY


def canonical_id_from_business_name(name: str) -> int | None:
    """Return canonical ID from business category name."""

    name = str(name).strip().lower()

    for canonical_id, info in CANONICAL_TAXONOMY.items():

        if info["business_name"].lower() == name:
            return canonical_id

    return None


def canonical_name_from_id(canonical_id: int) -> str | None:
    """Return business name from canonical ID."""

    try:
        canonical_id = int(canonical_id)

    except (TypeError, ValueError):

        return None

    category = CANONICAL_TAXONOMY.get(canonical_id)

    if category is None:

        return None

    return category["business_name"]