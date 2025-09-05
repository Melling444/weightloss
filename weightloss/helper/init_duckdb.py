import duckdb as db
import importlib.resources as ir

def create_db():
    """Create an empty DuckDB database for weight loss tracking."""
    weightloss_root = ir.files("weightloss")
    con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))
    con.execute("CREATE TABLE IF NOT EXISTS weight_log (date DATE, weight FLOAT, exercise_minutes INT, notes TEXT)")
    con.close()
    print("DuckDB database for weight loss tracking created.")

def restart_db():
    """Restart the DuckDB database by deleting and recreating it."""
    import os
    weightloss_root = ir.files("weightloss")
    db_path = weightloss_root.joinpath('data', 'weightloss.db')
    if os.path.exists(db_path):
        os.remove(db_path)
    create_db()
    print("DuckDB database for weight loss tracking restarted.")

def remove_db():
    """Remove the DuckDB database file."""
    import os
    weightloss_root = ir.files("weightloss")
    db_path = weightloss_root.joinpath('data', 'weightloss.db')
    if os.path.exists(db_path):
        os.remove(db_path)
        print("DuckDB database for weight loss tracking removed.")
    else:
        print("No DuckDB database file found to remove.")

if __name__ == "__main__":
    create_db()