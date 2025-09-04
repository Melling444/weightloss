import pandas as pd
from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import duckdb as db
import importlib.resources as ir

weightloss_root = ir.files("weightloss")

# con = db.connect(weightloss_root.joinpath('data', 'weightloss.db'))

# con.close()

app_ui = ui.page_fluid(
    ui.h1("Weight Loss App"),
    ui.p("This is a simple Shiny app for weight loss tracking and data analysis."),
    
)

def server(input, output, session):
    pass

app = App(app_ui, server)
