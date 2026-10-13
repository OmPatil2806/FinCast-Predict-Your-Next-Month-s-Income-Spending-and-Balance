"""Synthetic bank transactions for many realistic users.

Real personal bank data is private and short, so the forecasting models learn from
generated users instead. Every user belongs to one of four archetypes; each archetype
gives ranges, and each user draws their own values from those ranges.

Step 4.1: archetypes and user profiles (this part).
Step 4.2: one function per kind of transaction. Step 4.3: generate and save the full dataset.

Run on its own with:  python -m src.ingestion.synthetic_generator
Everything is driven by a numpy random Generator, so a fixed seed gives identical data.
"""

import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

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
        "salary": (450, 750),              # part-time job
        "extra_income": (300, 500),        # BAföG or parents
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


MAX_SPENDING_SHARE = 0.92      # people keep expected spending below ~92% of what they earn
WEEKS_PER_MONTH = 4.35


def _expected_spending(user: "UserProfile") -> float:
    """Rough monthly spending of a profile (used only for the affordability check)."""
    shopping = 0.8 * user.leisure + 30
    transport = 3 * 62 if user.has_car else 60
    return (user.rent + user.utilities + sum(p for _, p in user.subscriptions)
            + user.groceries_weekly * WEEKS_PER_MONTH + user.dining + user.leisure + shopping
            + transport + 60 + 90)   # + occasional tickets and seasonal costs spread over the year


def _expected_net_income(user: "UserProfile") -> float:
    if user.income_type == "freelance":
        return 0.92 * user.salary * 0.85   # lean months, then 15% income-tax prepayments
    return user.expected_income


def _make_affordable(user: "UserProfile") -> "UserProfile":
    """Scale rent and day-to-day budgets down when the profile would spend more than it earns."""
    limit = MAX_SPENDING_SHARE * _expected_net_income(user)
    spending = _expected_spending(user)
    if spending <= limit:
        return user
    scalable = user.rent + user.groceries_weekly * WEEKS_PER_MONTH + user.dining + 1.8 * user.leisure
    factor = max(0.5, 1 - (spending - limit) / scalable)
    user.rent = round(user.rent * factor / 10) * 10
    user.groceries_weekly = round(user.groceries_weekly * factor)
    user.dining = round(user.dining * factor / 5) * 5
    user.leisure = round(user.leisure * factor / 5) * 5
    return user


def make_user(user_id: str, archetype: str, rng: np.random.Generator) -> UserProfile:
    """Create one user of the given archetype with values drawn from its ranges."""
    spec = ARCHETYPES[archetype]
    low, high = spec["subscriptions"]
    n_subs = int(rng.integers(low, high + 1))
    max_price = 15 if archetype == "student" else float("inf")   # students skip Adobe, Urban Sports...
    affordable = [i for i, (_, price) in enumerate(SUBSCRIPTION_POOL) if price <= max_price]
    picked = rng.choice(affordable, size=n_subs, replace=False)
    user = UserProfile(
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
    return _make_affordable(user)


def make_users(n_users: int, seed: int = 42) -> list[UserProfile]:
    """Create n_users profiles; archetypes follow their 'share'. Same seed, same users."""
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    shares = np.array([ARCHETYPES[a]["share"] for a in names])
    archetypes = rng.choice(names, size=n_users, p=shares / shares.sum())
    return [make_user(f"user_{i:03d}", str(a), rng) for i, a in enumerate(archetypes, start=1)]


# ---------- step 4.2: transaction generators ----------
# Each function returns the user's rows for one month: dicts with
# date, description, amount (expenses negative, income positive) and category (the true label).

MONTHLY_INFLATION = 0.002          # prices drift up ~2.4% per year
YEARLY_SALARY_RAISE = 0.03         # salaried users get a raise every January
YEARLY_RENT_INCREASE = 0.02

# Several spellings per merchant, like real bank exports, so the categoriser has to learn
# rather than match one exact keyword.
MERCHANTS = {
    "groceries": [
        "REWE Markt Berlin", "REWE City 0815", "Rewe Sagt Danke", "LIDL Filiale 1234",
        "Lidl Dienstleistung", "ALDI SUED", "ALDI Nord Filiale 77", "EDEKA Center",
        "Edeka Mueller", "Netto Marken-Discount", "Kaufland Berlin", "PENNY Markt",
    ],
    "dining": [
        "Restaurant Bella Italia", "Sushi Bar Kyoto", "Burger King 2231", "McDonalds 1180",
        "Cafe Einstein", "Starbucks Coffee", "Lieferando.de", "Wolt Order", "Pizzeria Roma",
        "Doener Kebab Haus", "Vapiano", "Bäckerei Kamps",
    ],
    "leisure": [
        "Kino CineStar", "UCI Kinowelt", "Eventim Tickets", "Thalia Buchhandlung",
        "Steam Purchase", "PlayStation Store", "Museum Ticket", "Bowling Center",
    ],
    "shopping": [
        "Amazon EU Marketplace", "AMZN Mktp DE", "Zalando Bestellung", "H&M Online",
        "IKEA Deutschland", "MediaMarkt", "dm-drogerie markt", "Rossmann Drogerie",
        "Otto Versand", "Decathlon",
    ],
    "fuel": ["Shell Tankstelle", "ARAL Station", "TotalEnergies", "ESSO Station", "JET Tankstelle"],
    "transport_misc": ["DB Bahn Ticket", "DB Fernverkehr", "Uber Trip", "FlixBus", "BVG Einzelticket"],
}
ELECTRICITY = ["Vattenfall Strom Abschlag", "E.ON Energie Abschlag", "Stadtwerke Strom"]
INTERNET = ["Telekom Internet", "Vodafone Kabel", "O2 DSL"]
PHONE = ["Vodafone Mobilfunk", "Telekom Mobilfunk", "Congstar", "Aldi Talk"]
GERMAN_MONTHS = ["Januar", "Februar", "Maerz", "April", "Mai", "Juni", "Juli",
                 "August", "September", "Oktober", "November", "Dezember"]
CLIENTS = ["Mueller GmbH", "Schmidt & Partner", "Nordlicht Media", "Tech Solutions AG",
           "Weber Consulting", "Blau Design Studio", "Fischer IT", "Kraft Agentur"]
SHOCKS = [  # (description, low, high, category)
    ("Autowerkstatt Reparatur", 300, 1400, "Transport"),
    ("Zahnarzt Praxis Dr. Klein", 150, 900, "Health"),
    ("MediaMarkt Laptop", 600, 1500, "Shopping"),
    ("Saturn Waschmaschine", 400, 800, "Shopping"),
    ("Tierarzt Rechnung", 80, 450, "Health"),
    ("ADAC Abschleppdienst", 120, 350, "Transport"),
]


def _stable_int(*parts) -> int:
    """Deterministic number from text (unlike hash(), it is the same in every Python run)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def _date(month_start: pd.Timestamp, day: int) -> pd.Timestamp:
    """Day of the month, clamped to the month's last day (e.g. 30 -> 28 Feb)."""
    return month_start + pd.Timedelta(days=min(day, month_start.days_in_month) - 1)


def _random_day(rng: np.random.Generator, month_start: pd.Timestamp) -> pd.Timestamp:
    return _date(month_start, int(rng.integers(1, month_start.days_in_month + 1)))


def _row(date, description: str, amount: float, category: str) -> dict:
    return {"date": date, "description": description, "amount": round(float(amount), 2), "category": category}


def _split(rng: np.random.Generator, total: float, parts: int) -> np.ndarray:
    """Split a total into uneven positive parts."""
    weights = rng.dirichlet(np.full(parts, 2.0))
    return total * weights


def _years_since_start(month_start: pd.Timestamp, start: pd.Timestamp) -> int:
    return month_start.year - start.year


def salary(user: UserProfile, month_start, start, rng) -> list[dict]:
    rows = []
    years = _years_since_start(month_start, start)
    if user.income_type == "salary":
        pay = user.salary * (1 + YEARLY_SALARY_RAISE) ** years
        employer = "Studentenwerk Minijob" if user.archetype == "student" else "Gehalt ACME GmbH"
        rows.append(_row(_date(month_start, user.payday), employer, pay, "Income"))
    else:
        # freelancers: uneven monthly total, sometimes almost nothing, paid by 1-4 clients
        level = rng.lognormal(mean=-0.12, sigma=0.5)
        if rng.random() < 0.10:
            level = rng.uniform(0.0, 0.2)
        total = user.salary * level
        if total >= 50:
            n_payments = int(rng.integers(1, 5))
            for amount in _split(rng, total, n_payments):
                invoice = f"RE-{month_start:%Y%m}-{int(rng.integers(10, 99))}"
                client = CLIENTS[int(rng.integers(len(CLIENTS)))]
                rows.append(_row(_random_day(rng, month_start), f"Rechnung {invoice} {client}", amount, "Income"))
        # quarterly income-tax prepayment
        if month_start.month in (3, 6, 9, 12):
            rows.append(_row(_date(month_start, 10), "Finanzamt Einkommensteuer Vorauszahlung",
                             -0.15 * 3 * user.salary, "Other"))
    if user.extra_income:
        source = "Familienkasse Kindergeld" if user.archetype == "salaried_family" else "BAfoeG Amt"
        rows.append(_row(_date(month_start, 5), source, user.extra_income, "Income"))
    return rows


def rent(user: UserProfile, month_start, start, rng) -> list[dict]:
    amount = user.rent * (1 + YEARLY_RENT_INCREASE) ** _years_since_start(month_start, start)
    day = 1 if rng.random() < 0.9 else int(rng.integers(2, 4))
    return [_row(_date(month_start, day), f"Miete Wohnung {GERMAN_MONTHS[month_start.month - 1]}", -amount, "Rent")]


def utilities(user: UserProfile, month_start, price: float, rng) -> list[dict]:
    """Electricity (varies a little), internet and phone (fixed), sharing the user's utilities budget."""
    pick = _stable_int(user.user_id, "utilities")
    electricity = 0.45 * user.utilities * price * rng.uniform(0.95, 1.05)
    return [
        _row(_date(month_start, 2), ELECTRICITY[pick % len(ELECTRICITY)], -electricity, "Utilities"),
        _row(_date(month_start, 3), INTERNET[pick % len(INTERNET)], -round(0.35 * user.utilities, 2), "Utilities"),
        _row(_date(month_start, 15), PHONE[pick % len(PHONE)], -round(0.20 * user.utilities, 2), "Utilities"),
    ]


def subscriptions(user: UserProfile, month_start, rng) -> list[dict]:
    return [
        _row(_date(month_start, _stable_int(user.user_id, name) % 28 + 1), name, -price, "Subscriptions")
        for name, price in user.subscriptions
    ]


def groceries(user: UserProfile, month_start, price: float, rng) -> list[dict]:
    rows = []
    days = month_start.days_in_month
    for week_start in range(1, days + 1, 7):
        week_len = min(7, days - week_start + 1)
        budget = user.groceries_weekly * price * (week_len / 7) * rng.normal(1.0, 0.15)
        n_shops = int(rng.integers(1, 4))
        for amount in _split(rng, max(budget, 5), n_shops):
            day = week_start + int(rng.integers(0, week_len))
            shop = MERCHANTS["groceries"][int(rng.integers(len(MERCHANTS["groceries"])))]
            rows.append(_row(_date(month_start, day), shop, -max(amount, 1.5), "Groceries"))
    return rows


def _spread(user_budget: float, merchants: list[str], category: str, typical_count: int,
            month_start, price: float, rng) -> list[dict]:
    """Spend a noisy monthly budget across several payments at random merchants."""
    budget = user_budget * price * rng.gamma(shape=6, scale=1 / 6)
    count = max(1, int(rng.poisson(typical_count)))
    return [
        _row(_random_day(rng, month_start), merchants[int(rng.integers(len(merchants)))], -max(a, 2.0), category)
        for a in _split(rng, budget, count)
    ]


def dining(user: UserProfile, month_start, price: float, rng) -> list[dict]:
    return _spread(user.dining, MERCHANTS["dining"], "Dining", 6, month_start, price, rng)


def leisure(user: UserProfile, month_start, price: float, rng) -> list[dict]:
    rows = _spread(user.leisure, MERCHANTS["leisure"], "Leisure", 2, month_start, price, rng)
    # online and high-street shopping scales with the leisure budget
    rows += _spread(0.8 * user.leisure + 30, MERCHANTS["shopping"], "Shopping", 3, month_start, price, rng)
    return rows


def transport(user: UserProfile, month_start, price: float, rng) -> list[dict]:
    rows = []
    if user.has_car:
        for _ in range(int(rng.integers(2, 5))):
            rows.append(_row(_random_day(rng, month_start), MERCHANTS["fuel"][int(rng.integers(5))],
                             -rng.uniform(45, 80) * price, "Transport"))
    else:
        # Deutschlandticket: 49 EUR until Dec 2024, 58 EUR in 2025, 63 EUR from 2026
        ticket = 49.0 if month_start.year <= 2024 else (58.0 if month_start.year == 2025 else 63.0)
        rows.append(_row(_date(month_start, 1), "Deutschlandticket Abo", -ticket, "Transport"))
    for _ in range(int(rng.poisson(1.2))):
        rows.append(_row(_random_day(rng, month_start), MERCHANTS["transport_misc"][int(rng.integers(5))],
                         -rng.uniform(8, 90) * price, "Transport"))
    return rows


def seasonal(user: UserProfile, month_start, price: float, rng) -> list[dict]:
    rows = []
    key = (user.user_id, month_start.year)
    size = {"salaried_family": 1.6, "student": 0.4}.get(user.archetype, 1.0)

    if month_start.month == 12:  # Christmas gifts
        for amount in _split(rng, rng.uniform(150, 450) * size * price, int(rng.integers(2, 5))):
            shop = MERCHANTS["shopping"][int(rng.integers(len(MERCHANTS["shopping"])))]
            rows.append(_row(_random_day(rng, month_start), f"{shop} Geschenke", -amount, "Shopping"))

    trip_month = 7 + _stable_int(*key, "trip") % 2          # July or August, per user and year
    takes_trip = _stable_int(*key, "takes_trip") % 10 < (5 if user.archetype == "student" else 8)
    if takes_trip and month_start.month == trip_month:
        rows.append(_row(_date(month_start, int(rng.integers(1, 10))), "Lufthansa Flug",
                         -rng.uniform(150, 400) * size * price, "Leisure"))
        rows.append(_row(_date(month_start, int(rng.integers(10, 20))), "Booking.com Hotel",
                         -rng.uniform(250, 700) * size * price, "Leisure"))

    insurance_month = _stable_int(user.user_id, "insurance") % 12 + 1   # once a year, user-specific month
    if month_start.month == insurance_month:
        rows.append(_row(_date(month_start, 3), "Allianz Haftpflicht Versicherung",
                         -rng.uniform(70, 140) * size, "Other"))
        if user.has_car:
            rows.append(_row(_date(month_start, 4), "HUK-Coburg Kfz Versicherung",
                             -rng.uniform(350, 750), "Transport"))

    if user.archetype == "salaried_family" and month_start.month == 9:  # back to school
        rows.append(_row(_date(month_start, 5), "Schulbedarf Mcpaper",
                         -rng.uniform(80, 220) * price, "Other"))
    return rows


def shocks(user: UserProfile, month_start, price: float, rng) -> list[dict]:
    if rng.random() >= user.shock_probability:
        return []
    candidates = [s for s in SHOCKS if user.has_car or s[3] != "Transport"]
    description, low, high, category = candidates[int(rng.integers(len(candidates)))]
    return [_row(_random_day(rng, month_start), description, -rng.uniform(low, high) * price, category)]


def generate_month(user: UserProfile, month_start: pd.Timestamp, start: pd.Timestamp,
                   rng: np.random.Generator) -> list[dict]:
    """All transactions of one user in one month, sorted by date."""
    months_passed = (month_start.year - start.year) * 12 + month_start.month - start.month
    price = (1 + MONTHLY_INFLATION) ** months_passed
    rows = (
        salary(user, month_start, start, rng)
        + rent(user, month_start, start, rng)
        + utilities(user, month_start, price, rng)
        + subscriptions(user, month_start, rng)
        + groceries(user, month_start, price, rng)
        + dining(user, month_start, price, rng)
        + leisure(user, month_start, price, rng)
        + transport(user, month_start, price, rng)
        + seasonal(user, month_start, price, rng)
        + shocks(user, month_start, price, rng)
    )
    return sorted(rows, key=lambda r: r["date"])


# ---------- step 4.3: full dataset ----------

OUTPUT_COLUMNS = ["user_id", "archetype", "date", "description", "amount", "category"]


def generate_dataset(n_users: int = 300, months: int = 36, start: str = "2023-10-01",
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate all users and their transactions.

    Returns (transactions, users): transactions has OUTPUT_COLUMNS with the true category
    as label; users has one row per profile (archetype, income, rent, car, ...).
    """
    users = make_users(n_users, seed=seed)
    rng = np.random.default_rng(seed + 1)
    start_ts = pd.Timestamp(start)
    month_starts = pd.date_range(start_ts, periods=months, freq="MS")

    rows = []
    for user in users:
        for month_start in month_starts:
            for row in generate_month(user, month_start, start_ts, rng):
                row["user_id"] = user.user_id
                row["archetype"] = user.archetype
                rows.append(row)

    transactions = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    transactions = transactions.sort_values(["user_id", "date"], kind="stable", ignore_index=True)

    profiles = pd.DataFrame([
        {**{k: v for k, v in asdict(u).items() if k != "subscriptions"},
         "expected_income": u.expected_income,
         "n_subscriptions": len(u.subscriptions),
         "subscriptions": "; ".join(name for name, _ in u.subscriptions)}
        for u in users
    ])
    return transactions, profiles


def save_dataset(transactions: pd.DataFrame, users: pd.DataFrame, out_dir: str | Path,
                 transactions_file: str = "synthetic_transactions.csv",
                 users_file: str = "synthetic_users.csv") -> tuple[Path, Path]:
    """Write both tables as CSV (dates as YYYY-MM-DD) and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tx_path, users_path = out_dir / transactions_file, out_dir / users_file
    transactions.to_csv(tx_path, index=False, date_format="%Y-%m-%d")
    users.to_csv(users_path, index=False)
    return tx_path, users_path


def main() -> None:
    from src.utils.helpers import load_config, resolve_path

    config = load_config()
    settings = config["synthetic"]
    began = time.perf_counter()
    transactions, users = generate_dataset(
        n_users=settings["n_users"], months=settings["months"],
        start=settings["start"], seed=config["project"]["seed"],
    )
    tx_path, users_path = save_dataset(
        transactions, users, resolve_path(config["paths"]["data_raw"]),
        settings["transactions_file"], settings["users_file"],
    )
    print(f"{len(transactions):,} transactions for {len(users)} users "
          f"({transactions['date'].min():%Y-%m} to {transactions['date'].max():%Y-%m}) "
          f"in {time.perf_counter() - began:.1f}s")
    print(f"-> {tx_path}\n-> {users_path}")


if __name__ == "__main__":
    main()
