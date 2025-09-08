import duckdb as db
import importlib.resources as ir
import pandas as pd

weightloss_root = ir.files("weightloss")

def create_profile(name: str, age: int, height: float, goal_weight: float):
    """Create or overwrite the single user profile in the DuckDB database."""
    con = db.connect(str(weightloss_root.joinpath('data', 'weightloss.db')))
    try:
        # Try to get the last recorded weight; fall back to 150.0
        try:
            row = con.execute(
                "SELECT weight FROM weight_log ORDER BY date DESC LIMIT 1"
            ).fetchone()
            weight = row[0] if row and row[0] is not None else 150.0
        except db.CatalogException:
            weight = 150.0

        # Create the table with no id column
        con.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                name TEXT,
                age INTEGER,
                height DOUBLE,
                weight DOUBLE,
                goal_weight DOUBLE
            )
        """)

        # Clear any existing row (since you only want one profile)
        con.execute("DELETE FROM user_profile")

        # Insert the new/updated profile
        con.execute("""
            INSERT INTO user_profile (name, age, height, weight, goal_weight)
            VALUES (?, ?, ?, ?, ?)
        """, (name, age, height, weight, goal_weight))

        print(f"Profile created/updated for {name}")
    finally:
        con.close()

def fetch_profile():
    """Fetch the user profile from the DuckDB database."""
    con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))
    if not con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profile';").fetchone():
        con.close()
        return pd.DataFrame()
    df = con.execute("SELECT * FROM user_profile LIMIT 1").df()
    con.close()
    # Convert to native Python types to avoid numpy int/float issues
    return df.astype(object)

def edit_profile(name: str = None, age: int = None, height: float = None, goal_weight: float = None):
    """Edit the user profile in the DuckDB database."""
    con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))
    profile = fetch_profile()
    if profile.empty:
        print("No profile found to edit.")
        return
    current = profile.iloc[0]
    name = name if name is not None else current['name']
    age = age if age is not None else current['age']
    height = height if height is not None else current['height']
    goal_weight = goal_weight if goal_weight is not None else current['goal_weight']
    con.execute("""
        UPDATE user_profile
        SET name = ?, age = ?, height = ?, goal_weight = ?
        WHERE id = ?
    """, (name, age, height, goal_weight, current['id']))
    con.close()
    print(f"Profile updated for {name}")