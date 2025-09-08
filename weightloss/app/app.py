import pandas as pd
import numpy as np
from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import duckdb as db
from weightloss.helper.db_update import add_log, fetch_logs, delete_log, edit_log
from weightloss.helper.init_duckdb import create_db, restart_db, remove_db
from weightloss.helper.profile import create_profile, fetch_profile, edit_profile
import google.generativeai as genai
import importlib.resources as ir
from dotenv import load_dotenv
import os

weightloss_root = ir.files("weightloss")
API_path = weightloss_root.joinpath('app', '.env')
load_dotenv(dotenv_path=API_path)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash",)
css_path = weightloss_root.joinpath("app", "style.css")
runner_image = mpimg.imread(weightloss_root.joinpath("content", "man_running.png"))
background_image = mpimg.imread(weightloss_root.joinpath("content", "background.png"))

# con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))

# con.close()

app_ui = ui.page_fluid(
    ui.include_css(css_path),
    ui.h1("Weight Loss App"),
    ui.p("This is a simple Shiny app for weight loss tracking and data analysis.", class_="app-description"),
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
                ui.input_action_button("profile_btn", "View/Edit Profile"),
                ui.input_action_button("restart_db_btn", "Restart Database"),
            ),
            ui.card(
                ui.h3("AI Suggestions"),
                ui.chat_ui(id="ai_chat", messages = ["**Hello!** I can provide weight loss tips and suggestions. Ask me anything!"]),
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

    # View/Edit Profile
    @reactive.Effect
    @reactive.event(input.profile_btn)
    def _():
        profile = fetch_profile()
        if not profile.empty:
            current = profile.iloc[0]
            ui.update_text("profile_name", value=current['name'])
            ui.update_numeric("profile_age", value=current['age'])
            ui.update_numeric("profile_height", value=current['height'])
            ui.update_numeric("profile_goal_weight", value=current['goal_weight'])
        else:
            current = {'name': '', 'age': 0, 'height': 0.0, 'goal_weight': 0.0}
        m = ui.modal(
            ui.h3("View/Edit Profile"),
            ui.input_text("profile_name", "Name", value=str(current['name'])),
            ui.input_numeric("profile_age", "Age", value=int(current['age']), min=0),
            ui.input_numeric("profile_height", "Height (inches)", value=float(current['height']), min=0),
            ui.input_numeric("profile_goal_weight", "Goal Weight (lbs)", value=float(current['goal_weight']), min=0),
            ui.input_action_button("confirm_profile_btn", "Save Profile"),
        )
        ui.modal_show(m)
    
    #Confirm Profile Edit
    @reactive.Effect
    @reactive.event(input.confirm_profile_btn)
    def _():
        if fetch_profile().empty:
            create_profile(name=input.profile_name(), age=input.profile_age(),
                           height=input.profile_height(), goal_weight=input.profile_goal_weight())
        else:
            edit_profile(name=input.profile_name(), age=input.profile_age(),
                          height=input.profile_height(), goal_weight=input.profile_goal_weight())
        ui.modal_remove()
        _refresh()

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
        df = logs()
        df = df.head(8)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return render.DataGrid(df, height='150px')
    
    def get_marker_image(path, zoom=0.1):
        return OffsetImage(path, zoom=zoom)

    @output
    @render.plot
    def weight_plot():
        df = logs()
        # Ensure that we have a max of 8 points to plot
        df = df.head(8)
        if df.empty:
            return
        fig, ax = plt.subplots()

        x_min, x_max = df['date'].min(), df['date'].max()
        y_min, y_max = df['weight'].min(), df['weight'].max()

        x_range = (x_max - x_min)
        if pd.isna(x_range) or x_range == pd.Timedelta(0):
            # single timestamp → pad ±1 day
            x_margin = pd.Timedelta(days=1)
        else:
            x_margin = max(pd.Timedelta(seconds=x_range.total_seconds() * 0.05),
                        pd.Timedelta(hours=12))  # minimum half-day pad

        y_range = (y_max - y_min)
        if not np.isfinite(y_range) or y_range == 0:
            # flat value → pad ±max(1 lb, 5% of abs level)
            y_margin = max(1.0, abs(y_max) * 0.05 if np.isfinite(y_max) else 1.0)
        else:
            y_margin = max(y_range * 0.10, 1.0)  # at least 1 lb

        ax.imshow(
            background_image,
            aspect='auto',
            extent=[x_min - x_margin, x_max + x_margin,
                    y_min - y_margin, y_max + y_margin],
            zorder=0
    )

        ax.plot(df['date'], df['weight'], linestyle='-', color='red')

        imagebox = get_marker_image(runner_image, zoom=0.04)
        for(x, y) in zip(df['date'], df['weight']):
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)   
        ax.set_xlabel('Date')
        ax.set_xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Weight (lbs)')
        ax.grid(True, color='white', linestyle='--',)
        return fig
    
    chat = ui.Chat(id = "ai_chat")
    @chat.on_user_submit
    async def handle_user_input(user_input: str):
        prompt = f"You are a helpful AI assistant trained to provide weight loss tips and health suggestions. Answer the user's question: {user_input}"
        response = model.generate_content(prompt, stream=True)
        await chat.append_message_stream(response)    

app = App(app_ui, server)
