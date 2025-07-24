import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
import os
import json

dpg.create_context()

COMMON_FILES_JSON = """
{
    "Sample A": "C:/Users/cwinklm/Documents/aggrigator_experiments/output/tables/aggregation_value_summary_joint_noise_wormbodies_fgbg_nematodes_dropout_pu_combined.csv",
    "Sample B": "data/sample_b.csv",
    "Test Set": "data/test.csv"
}
"""



# Storage
data_store = {
    "df": None,
    "hovered_point": None,
    "common_files": json.loads(COMMON_FILES_JSON)
}

# Callbacks
def load_csv_callback(sender, app_data, user_data):
    path = dpg.get_value(user_data)
    if not os.path.isfile(path):
        dpg.set_value("status", f"File not found: {path}")
        return
    try:
        df = pd.read_csv(path)
        data_store["df"] = df
        dpg.set_value("status", f"Loaded: {os.path.basename(path)}")
        update_column_options()
    except Exception as e:
        dpg.set_value("status", f"Error loading file: {e}")
# def load_csv_callback(sender, app_data, user_data):
#     path = dpg.get_value(user_data)
#     if not os.path.isfile(path):
#         dpg.set_value("status", f"File not found: {path}")
#         return
#     try:
#         data_store["df"] = pd.read_csv(path)
#         dpg.set_value("status", f"Loaded {path}")
#         update_column_options()
#     except Exception as e:
#         dpg.set_value("status", f"Failed to load: {e}")

def update_column_options():
    if data_store["df"] is not None:
        cols = list(data_store["df"].columns)
        dpg.configure_item("x_col", items=cols)
        dpg.configure_item("y_col", items=cols)

def plot_callback():
    dpg.delete_item("plot_series", children_only=True)
    df = data_store["df"]
    if df is None:
        return
    x_col = dpg.get_value("x_col")
    y_col = dpg.get_value("y_col")
    if x_col not in df.columns or y_col not in df.columns:
        return
    x_data = df[x_col].to_numpy()
    y_data = df[y_col].to_numpy()
    dpg.add_scatter_series(x_data, y_data, label="Data", parent="y_axis", tag="scatter_series")

def click_callback(sender, app_data):
    if not dpg.is_item_hovered("plot_area"):
        return

    # Get mouse position in plot coordinates
    plot_mouse_pos = dpg.get_plot_mouse_pos()
    x_val, y_val = plot_mouse_pos

    df = data_store["df"]
    if df is None:
        return

    x_col = dpg.get_value("x_col")
    y_col = dpg.get_value("y_col")
    if x_col not in df.columns or y_col not in df.columns:
        return

    # Find nearest point
    distances = np.sqrt((df[x_col] - x_val) ** 2 + (df[y_col] - y_val) ** 2)
    closest_idx = distances.idxmin()
    row = df.iloc[closest_idx]
    info = f"Index: {closest_idx}\n" + "\n".join([f"{col}: {row[col]}" for col in ["uq_map_name"]])
    dpg.set_value("point_info", info)
    data_store["hovered_point"] = row

def save_point_callback():
    row = data_store["hovered_point"]
    if row is not None:
        with open("saved_points.csv", "a") as f:
            f.write(",".join(str(v) for v in row.values) + "\n")
        dpg.set_value("status", "Point saved to saved_points.csv")

# GUI
with dpg.window(label="Scatterplot Explorer", width=800, height=600):


    with dpg.group(horizontal=True):
        dpg.add_combo(list(data_store["common_files"].keys()), 
                    label="Common Files", 
                    width=150,
                    callback=lambda s,a,u: dpg.set_value("file_path", data_store["common_files"][a]))

        dpg.add_input_text(label="File Path", tag="file_path", width=400)

        dpg.add_button(label="Browse", callback=lambda: dpg.show_item("file_dialog"))

        dpg.add_button(label="Load", callback=load_csv_callback, user_data="file_path")
    # dpg.add_text("Load CSV:")
    # dpg.add_input_text(label="File Path", tag="file_path")
    # dpg.add_button(label="Load from Path", callback=load_csv_callback, user_data="file_path")

    # dpg.add_combo(data_store["common_files"], label="Select from list", tag="file_dropdown")
    # dpg.add_button(label="Load from Dropdown", callback=load_csv_callback, user_data="file_dropdown")

    # dpg.add_button(label="Browse...", callback=lambda: dpg.show_item("file_dialog"))
    
    dpg.add_text("Choose columns:")
    dpg.add_combo([], label="X Axis", tag="x_col")
    dpg.add_combo([], label="Y Axis", tag="y_col")
    dpg.add_button(label="Plot", callback=plot_callback)

    with dpg.plot(label="Scatterplot", height=400, width=-1, tag="plot_area"):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="X Axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis")
        dpg.set_item_callback("plot_area", click_callback)
        dpg.add_scatter_series([], [], parent="y_axis", tag="scatter_series")
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=click_callback)

    dpg.add_text("Point Info:")
    dpg.add_input_text(tag="point_info", multiline=True, readonly=True, width=-1, height=100)

    dpg.add_button(label="Save Point", callback=save_point_callback)
    dpg.add_text("", tag="status", color=[0, 200, 0])

# File dialog (hidden)
with dpg.file_dialog(directory_selector=False, show=False, callback=lambda s, a: dpg.set_value("file_path", a["file_path_name"]), tag="file_dialog"):
    dpg.add_file_extension(".csv", color=(0, 255, 0, 255))

dpg.create_viewport(title="CSV Scatterplot Viewer", width=820, height=700)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
