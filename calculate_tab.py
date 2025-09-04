import gradio as gr
from regex import D
from utils import get_test_folders, process_folder, get_image_for_display
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import math
import pandas as pd

def calculate_euclidean_distance(p1, p2, dims):
    """Calculates the scaled Euclidean distance between two normalized points."""
    return math.sqrt(((p1[0] - p2[0]) * dims[0])**2 + (((p1[1] - p2[1]) * dims[1])**2))

def create_distance_plot(interactions, test_id, dims, current_image_index):
    """Creates a line plot of distances between interaction points."""
    if not interactions or not test_id or test_id not in interactions or not dims:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data to display.", ha='center', va='center')
        return fig, ""

    interaction_data = interactions[test_id]
    
    sorted_img_ids = sorted(interaction_data.keys())

    interaction_points = []
    for img_id in sorted_img_ids:
        interaction = interaction_data[img_id]
        params = interaction.get("interaction_parameters", {})
        grounding = params.get("grounding")
        
        if not grounding:
            continue

        if interaction.get("interaction_type") == "slide":
            if len(grounding) == 2:
                interaction_points.append({"start": grounding[0], "end": grounding[1]})
        else:
            interaction_points.append({"start": grounding, "end": grounding})

    if len(interaction_points) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough interaction points to draw a plot.", ha='center', va='center')
        return fig, ""

    distances = [calculate_euclidean_distance(interaction_points[i]['start'], interaction_points[i-1]['end'], dims) for i in range(1, len(interaction_points))]

    if not distances:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No distances to plot.", ha='center', va='center')
        return fig, ""

    mean_dist = pd.Series(distances).mean()
    std_dist = pd.Series(distances).std()

    stats_text = f"""<b>Descriptive Statistics:</b><br>
Mean: {mean_dist:.2f}<br>
Std Dev: {std_dist:.2f}"""

    fig, ax = plt.subplots()
    x_values = [f"{i}-{i+1}" for i in range(1, len(distances) + 1)]
    ax.plot(x_values, distances, marker='o', label='Distance')

    if current_image_index > 0:
        highlight_index = current_image_index
        if highlight_index <= len(distances):
            ax.plot(x_values[highlight_index - 1], distances[highlight_index - 1], 'r*', markersize=15, label=f'Current Image')

    ax.axhline(y=mean_dist, color='r', linestyle='--', label='Mean')
    
    ax.set_title(f"Interaction Distances for {test_id}")
    ax.set_xlabel("Interaction Step")
    ax.set_ylabel("Euclidean Distance (pixels)")
    ax.grid(True)
    ax.legend()
    
    return fig, stats_text


def calculate_tab():
    with gr.TabItem("Load Calculate"):
        # States for load calculate tab
        calc_image_groups_state = gr.State({})
        calc_folder_path_state = gr.State("")
        calc_interactions_state = gr.State({})
        calc_current_test_id_state = gr.State("")
        calc_current_image_index_state = gr.State(0)
        calc_image_dimensions_state = gr.State()


        with gr.Row():
            calc_folder_input = gr.Dropdown(label="Select Test Folder", choices=get_test_folders(), interactive=True)
            calc_start_button = gr.Button("Start")

        # Sub-tabs
        with gr.Tabs() as calc_sub_tabs:
            
            with gr.TabItem("Simple Path"):
                gr.Markdown("### Simple Path Analysis")
                test_id_dropdown_simple = gr.Dropdown(label="Test ID", interactive=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        img_id_label_simple = gr.Label(label="Image ID")
                        image_display_simple = gr.Image(label="Image", interactive=False, type="pil", height=300)
                        with gr.Row():
                            prev_button_simple = gr.Button("Previous")
                            next_button_simple = gr.Button("Next")
                    with gr.Column(scale=2):
                        plot_display_simple = gr.Plot(label="Interaction Distance Plot")
                    with gr.Column(scale=1):
                        stats_label_simple = gr.Markdown()

            with gr.TabItem("Advanced Path"):
                gr.Markdown("### Advanced Path Analysis")
                gr.Markdown("*Coming soon...*")

            with gr.TabItem("Standalone"):
                gr.Markdown("### Standalone Analysis")
                gr.Markdown("*Coming soon...*")

        def calc_start_process(folder_path):
            # This function is called when the main "Start" button is clicked
            images, test_id, message, image_groups, dims, interactions = process_folder(folder_path)
            if not images:
                gr.Warning("No valid data found in the selected folder.", duration=2)
                return {}, folder_path, {}, gr.update(choices=[], value=None)
            else:
                gr.Info(f"Successfully loaded data from {folder_path}", duration=2)
                test_ids = sorted(list(image_groups.keys()))
                return image_groups, folder_path, interactions, gr.update(choices=test_ids, value=test_ids[0] if test_ids else None)

        calc_start_button.click(
            fn=calc_start_process,
            inputs=[calc_folder_input],
            outputs=[calc_image_groups_state, calc_folder_path_state, calc_interactions_state, test_id_dropdown_simple]
        )

        def on_test_id_select_simple(test_id, image_groups, interactions):
            if not test_id or not image_groups or not interactions:
                return None, 0, None, "", None, "", gr.update(interactive=False), gr.update(interactive=False), test_id

            images = image_groups.get(test_id, [])
            if not images:
                return None, 0, None, "", None, "", gr.update(interactive=False), gr.update(interactive=False), test_id

            image_path = images[0]
            img_id = os.path.basename(image_path)
            with Image.open(image_path) as img:
                dims = img.size

            display_image = get_image_for_display(image_path, test_id, interactions)
            img_label = f"{img_id} (1/{len(images)})"
            
            plot, stats = create_distance_plot(interactions, test_id, dims, 0)

            return (
                display_image, 0, dims, img_label, plot, stats,
                gr.update(interactive=False), # prev
                gr.update(interactive=len(images) > 1), # next
                test_id
            )

        test_id_dropdown_simple.change(
            fn=on_test_id_select_simple,
            inputs=[test_id_dropdown_simple, calc_image_groups_state, calc_interactions_state],
            outputs=[
                image_display_simple, calc_current_image_index_state, calc_image_dimensions_state,
                img_id_label_simple, plot_display_simple, stats_label_simple, prev_button_simple, next_button_simple,
                calc_current_test_id_state
            ]
        )

        def change_image_simple(direction, test_id, index, image_groups, interactions, dims):
            new_index = index + direction
            images = image_groups.get(test_id, [])

            if not (0 <= new_index < len(images)):
                return gr.update(), new_index, gr.update(), gr.update(), gr.update(), gr.update()

            image_path = images[new_index]
            img_id = os.path.basename(image_path)
            
            display_image = get_image_for_display(image_path, test_id, interactions)
            img_label = f"{img_id} ({new_index + 1}/{len(images)})"
            
            plot, stats = create_distance_plot(interactions, test_id, dims, new_index)

            return (
                display_image, new_index, img_label, plot, stats,
                gr.update(interactive=new_index > 0), 
                gr.update(interactive=new_index < len(images) - 1)
            )

        prev_button_simple.click(
            fn=lambda test_id, index, groups, inter, dims: change_image_simple(-1, test_id, index, groups, inter, dims),
            inputs=[calc_current_test_id_state, calc_current_image_index_state, calc_image_groups_state, calc_interactions_state, calc_image_dimensions_state],
            outputs=[image_display_simple, calc_current_image_index_state, img_id_label_simple, plot_display_simple, stats_label_simple, prev_button_simple, next_button_simple]
        )

        next_button_simple.click(
            fn=lambda test_id, index, groups, inter, dims: change_image_simple(1, test_id, index, groups, inter, dims),
            inputs=[calc_current_test_id_state, calc_current_image_index_state, calc_image_groups_state, calc_interactions_state, calc_image_dimensions_state],
            outputs=[image_display_simple, calc_current_image_index_state, img_id_label_simple, plot_display_simple, stats_label_simple, prev_button_simple, next_button_simple]
        )