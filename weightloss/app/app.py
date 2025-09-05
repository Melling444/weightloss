import pandas as pd
from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import duckdb as db
from weightloss.helper.db_update import add_log, fetch_logs, delete_log, edit_log
from weightloss.helper.init_duckdb import create_db, restart_db, remove_db
import importlib.resources as ir

weightloss_root = ir.files("weightloss")

# con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))

# con.close()

app_ui = ui.page_fluid(
    ui.h1("Weight Loss App"),
    ui.p("This is a simple Shiny app for weight loss tracking and data analysis."),
    ui.page_fillable(
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_date("date", "Date", value=pd.Timestamp.now().date()),
                ui.input_numeric("weight", "Weight (lbs)", value=150, min=50, max=500, step=0.1),
                ui.input_numeric("exercise_minutes", "Exercise Minutes", value=30, min=0, max=300, step=1),
                ui.input_text("notes", "Notes", placeholder="Any additional notes..."),
                ui.input_action_button("add_log_btn", "Add Log Entry"),
                ui.input_date("edit_date", "Select Date to Edit", value=pd.Timestamp.now().date()),
                ui.input_action_button("edit_log_btn", "Edit Entry"),
                ui.input_action_button("delete_log_btn", "Delete Today's Entry"),
                ui.input_action_button("restart_db_btn", "Restart Database"),
            ),
            ui.card(
                ui.h3("Weight Log Entries"),
                ui.output_data_frame("log_table"),
                ui.h3("Weight Over Time"),
                ui.output_plot("weight_plot")
            )
        ),
        )
    )

def server(input, output, session):

    data_trigger = reactive.Value(0)
    def _refresh():
        data_trigger.set(data_trigger() + 1)
    
    # Add
    @reactive.Effect
    @reactive.event(input.add_log_btn)
    def _():
        add_log(date=input.date(), weight=input.weight(),
                exercise_minutes=input.exercise_minutes(), notes=input.notes())
        ui.update_date("date", value=pd.Timestamp.now().date()),
        ui.update_numeric("weight", value=150)
        ui.update_numeric("exercise_minutes", value=30)
        ui.update_text("notes", placeholder="")
        _refresh()

    # Delete
    @reactive.Effect
    @reactive.event(input.delete_log_btn)
    def _():
        delete_log(date=input.date())
        _refresh()

    # Restart DB
    @reactive.Effect
    @reactive.event(input.restart_db_btn)
    def _():
        restart_db()
        _refresh()
    
    # Edit
    @reactive.Effect
    @reactive.event(input.edit_log_btn)
    def _():
        m = ui.modal(
            ui.h3("Edit Log Entry"),
            ui.input_numeric("edit_weight", "Weight (lbs)", value=150, min=50, max=500, step=0.1),
            ui.input_numeric("edit_exercise_minutes", "Exercise Minutes", value=30, min=0, max=300, step=1),
            ui.input_text("edit_notes", "Notes", placeholder="Any additional notes..."),
            ui.input_action_button("confirm_edit_btn", "Confirm Edit"),
        )
        ui.modal_show(m)
    
    @reactive.Effect
    @reactive.event(input.confirm_edit_btn)
    def _():
        edit_log(date=input.edit_date(), weight=input.edit_weight(),
                    exercise_minutes=input.edit_exercise_minutes(), notes=input.edit_notes())
        ui.modal_remove()
        _refresh()

    def logs():
        data_trigger()  # depend on the trigger
        try:
            return fetch_logs()
        except Exception:
            # Be resilient if DB was removed/recreated
            return pd.DataFrame(columns=["date", "weight", "exercise_minutes", "notes"])

    # 4) Outputs use the reactive calc
    @output
    @render.data_frame
    def log_table():
        return logs()

    @output
    @render.plot
    def weight_plot():
        df = logs()
        if df.empty:
            return
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['weight'], marker='o')
        ax.set_xlabel('Date')
        ax.set_xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Weight (lbs)')
        ax.set_title('Weight Over Time')
        ax.grid(True)
        return fig

app = App(app_ui, server)
