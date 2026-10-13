"""Synthetic bank transactions for many realistic users.

Real personal bank data is private and short, so the forecasting models learn from
generated users instead. Every user belongs to one of four archetypes; each archetype
gives ranges, and each user draws their own values from those ranges.

Step 4.1: archetypes and user profiles (this part).
Step 4.2: one function per kind of transaction. Step 4.3: generate the full dataset.
Everything is driven by a numpy random Generator, so a fixed seed gives identical data.
"""

from dataclasses import dataclass, field

import numpy as np

# ---------- archetypes ----------
# Every range is (low, high); a user draws one value uniformly inside it.
# Amounts are monthly euros unless the name says otherwise.
ARCHETYPES: dict[str, dict] = {
    "salaried_single": {
        "share": 0.35,
        "income_type": "salary",
        "salary": (2800, 3800),            # net, paid on payday
        "extra_income": (0, 0),
        "rent": (700, 1100),
        "utilities": (90, 160),            # electricity, internet, phone
        "groceries_weekly": (40, 70),
        "dining": (80, 220),
        "leisure": (40, 140),
        "car_probability": 0.40,
        "subscriptions": (2, 4),           # how many streaming / gym / app subscriptions
        "shock_probability": 0.04,         # chance per month of a rare big cost
    },
    "salaried_family": {
        "share": 0.25,
        "income_type": "salary",
        "salary": (3800, 5500),
        "extra_income": (250, 250),        # Kindergeld per month
        "rent": (1100, 1700),
        "utilities": (160, 260),
        "groceries_weekly": (110, 180),
        "dining": (60, 160),
        "leisure": (60, 180),
        "car_probability": 0.80,
        "subscriptions": (2, 5),
        "shock_probability": 0.06,
    },
    "freelancer": {
        "share": 0.25,
        "income_type": "freelance",
        "salary": (1500, 6000),            # average monthly client income, paid irregularly
        "extra_income": (0, 0),
        "rent": (750, 1200),
        "utilities": (100, 180),
        "groceries_weekly": (45, 80),
        "dining": (100, 260),
        "leisure": (50, 160),
        "car_probability": 0.50,
        "subscriptions": (3, 6),           # includes software tools
        "shock_probability": 0.05,
    },
    "student": {
        "share": 0.15,
        "income_type": "salary",
        "salary": (350, 650),              # part-time job
        "extra_income": (250, 450),        # BAföG or parents
        "rent": (350, 550),                # shared flat
        "utilities": (30, 60),
        "groceries_weekly": (25, 45),
        "dining": (40, 120),
        "leisure": (30, 90),
        "car_probability": 0.10,
        "subscriptions": (1, 3),
        "shock_probability": 0.03,
    },
}

# Subscription pool: (merchant name, monthly price). Users pick a few at random.
SUBSCRIPTION_POOL = [
    ("Netflix.com", 13.99),
    ("Spotify AB", 10.99),
    ("Disney Plus", 9.99),
    ("Amazon Prime", 8.99),
    ("McFit Mitgliedsbeitrag", 24.90),
    ("Urban Sports Club", 49.00),
    ("Apple iCloud", 2.99),
    ("YouTube Premium", 12.99),
    ("Audible", 9.95),
    ("Adobe Creative Cloud", 66.45),
    ("Microsoft 365", 10.00),
    ("Zeit Online Abo", 19.99),
]


@dataclass
class UserProfile:
    """One synthetic user with their own fixed amounts."""

    user_id: str
    archetype: str
    income_type: str             # "salary" or "freelance"
    salary: float                # monthly net salary, or average freelance income
    extra_income: float          # Kindergeld / BAföG / parents, 0 if none
    payday: int                  # day of month the salary arrives (freelancers: unused)
    rent: float
    utilities: float
    groceries_weekly: float
    dining: float
    leisure: float
    has_car: bool
    subscriptions: list[tuple[str, float]] = field(default_factory=list)
    shock_probability: float = 0.0

    @property
    def expected_income(self) -> float:
        return self.salary + self.extra_income


def _draw(rng: np.random.Generator, low_high: tuple[float, float], step: float = 10) -> float:
    """Uniform draw inside the range, rounded to a realistic step (e.g. rent to 10 euros)."""
    low, high = low_high
    return float(round(rng.uniform(low, high) / step) * step) if high > low else float(low)


def make_user(user_id: str, archetype: str, rng: np.random.Generator) -> UserProfile:
    """Create one user of the given archetype with values drawn from its ranges."""
    spec = ARCHETYPES[archetype]
    low, high = spec["subscriptions"]
    n_subs = int(rng.integers(low, high + 1))
    picked = rng.choice(len(SUBSCRIPTION_POOL), size=n_subs, replace=False)
    return UserProfile(
        user_id=user_id,
        archetype=archetype,
        income_type=spec["income_type"],
        salary=_draw(rng, spec["salary"], step=50),
        extra_income=_draw(rng, spec["extra_income"], step=10),
        payday=int(rng.choice([25, 27, 28, 30])) if spec["income_type"] == "salary" else 0,
        rent=_draw(rng, spec["rent"], step=10),
        utilities=_draw(rng, spec["utilities"], step=5),
        groceries_weekly=_draw(rng, spec["groceries_weekly"], step=1),
        dining=_draw(rng, spec["dining"], step=5),
        leisure=_draw(rng, spec["leisure"], step=5),
        has_car=bool(rng.random() < spec["car_probability"]),
        subscriptions=[SUBSCRIPTION_POOL[i] for i in sorted(picked)],
        shock_probability=spec["shock_probability"],
    )


def make_users(n_users: int, seed: int = 42) -> list[UserProfile]:
    """Create n_users profiles; archetypes follow their 'share'. Same seed, same users."""
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    shares = np.array([ARCHETYPES[a]["share"] for a in names])
    archetypes = rng.choice(names, size=n_users, p=shares / shares.sum())
    return [make_user(f"user_{i:03d}", str(a), rng) for i, a in enumerate(archetypes, start=1)]
