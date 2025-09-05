import duckdb as db
import importlib.resources as ir
import pandas as pd

def add_log(date = None, weight = None, exercise_minutes = None, notes = None):
    """Update the DuckDB database with a new weight log entry."""
    weightloss_root = ir.files("weightloss")
    con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))
    if date is None:
        date = pd.Timestamp.now().date()
    con.execute(
        "INSERT INTO weight_log (date, weight, exercise_minutes, notes) VALUES (?, ?, ?, ?)",
        (date, weight, exercise_minutes, notes)
    )
    con.close()
    print(f"Added entry: {date}, {weight}, {exercise_minutes}, {notes}")

def fetch_logs():
    """Fetch all weight log entries from the DuckDB database."""
    weightloss_root = ir.files("weightloss")
    con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))
    df = con.execute("SELECT * FROM weight_log ORDER BY date DESC").df()
    con.close()
    return df

def delete_log(date = None):
    """Delete a weight log entry by date."""
    if date is None:
        print("Date must be provided to delete a log entry.")
        return
    weightloss_root = ir.files("weightloss")
    con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))
    con.execute("DELETE FROM weight_log WHERE date = ?", (date,))
    con.close()
    print(f"Deleted entry for date: {date}")

def edit_log(date = None, weight = None, exercise_minutes = None, notes = None):
    """Edit an existing weight log entry by date."""
    if date is None:
        print("Date must be provided to edit a log entry.")
        return
    weightloss_root = ir.files("weightloss")
    con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))
    con.execute(
        "UPDATE weight_log SET weight = ?, exercise_minutes = ?, notes = ? WHERE date = ?",
        (weight, exercise_minutes, notes, date)
    )
    con.close()
    print(f"Edited entry for date: {date} to {weight}, {exercise_minutes}, {notes}")

if __name__ == "__main__":
    fetch_logs()
    print("Database operations completed.")