import gradio as gr
from regex import D
from utils import get_test_folders, process_folder, get_image_for_display
import os
from PIL import Image
import plotly.graph_objects as go
import numpy as np
import math
import pandas as pd

def calculate_euclidean_distance(p1, p2, dims):
    """Calculates the scaled Euclidean distance between two normalized points."""
    return math.sqrt(((p1[0] - p2[0]) * dims[0])**2 + (((p1[1] - p2[1]) * dims[1])**2))

def create_distance_plot(interactions, test_id, dims, current_image_index):
    """Creates a line plot of distances between interaction points."""
    if not interactions or not test_id or test_id not in interactions or not dims:
        fig = go.Figure()
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, annotations=[{"text": "No data to display.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
        return fig, "", "", ""

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
        fig = go.Figure()
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, annotations=[{"text": "Not enough interaction points to draw a plot.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
        return fig, "", "", ""

    distances = [calculate_euclidean_distance(interaction_points[i]['start'], interaction_points[i-1]['end'], dims) for i in range(1, len(interaction_points))]

    if not distances:
        fig = go.Figure()
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, annotations=[{"text": "No distances to plot.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
        return fig, "", "", ""

    mean_dist = pd.Series(distances).mean()
    std_dist = pd.Series(distances).std()

    mean_dist_without_current = None
    operation_quality_score = None
    if current_image_index is not None and len(distances) > 1:
        indices_to_remove = []
        if current_image_index > 0:
            indices_to_remove.append(current_image_index - 1)
        if current_image_index < len(distances):
            indices_to_remove.append(current_image_index)
        
        remaining_distances = [d for i, d in enumerate(distances) if i not in indices_to_remove]
        
        if remaining_distances:
            mean_dist_without_current = pd.Series(remaining_distances).mean()
            if mean_dist is not None and mean_dist > 0:
                operation_quality_score = (mean_dist_without_current - mean_dist) / mean_dist
            else:
                operation_quality_score = 0.0

    average_operation_quality_score = None
    if len(distances) > 1:
        all_scores = []
        num_interaction_points = len(interaction_points)
        for i in range(num_interaction_points):
            indices_to_remove = []
            if i > 0:
                indices_to_remove.append(i - 1)
            if i < len(distances):
                indices_to_remove.append(i)
            
            if not indices_to_remove:
                continue

            remaining_distances = [d for j, d in enumerate(distances) if j not in indices_to_remove]

            if remaining_distances:
                mean_dist_without_i = pd.Series(remaining_distances).mean()
                if mean_dist is not None and mean_dist > 0:
                    score = (mean_dist_without_i - mean_dist) / mean_dist
                    all_scores.append(score)
        
        if all_scores:
            average_operation_quality_score = pd.Series(all_scores).mean()

    stats_basic = f"""<b>Basic Statistics:</b><br>
Mean: {mean_dist:.2f}<br>
Std Dev: {std_dist:.2f}"""

    stats_mean_wo_current = ""
    if mean_dist_without_current is not None:
        stats_mean_wo_current = f"<b>Mean (w/o current):</b><br>{mean_dist_without_current:.2f}"

    stats_score = ""
    if operation_quality_score is not None:
        color = "green" if operation_quality_score >= 0 else "red"
        stats_score += f"<b>当前操作得分: <span style='color:{color};'>{operation_quality_score:.2%}</span></b>"
    if average_operation_quality_score is not None:
        if stats_score:
            stats_score += "<br>"
        stats_score += f"平均得分: {average_operation_quality_score:.2%}"
    
    if operation_quality_score is not None or average_operation_quality_score is not None:
        stats_score += """<br><b>得分释义:</b><br>
该分数衡量当前操作对交互路径的影响。<br>
<span style='color:green;'>正分 (越高越好):</span> 高效交互, 缩短了路径。<br>
<span style='color:red;'>负分 (越低越差):</span> 低效交互, 拉长了路径。"""

    x_values = [f"{i}-{i+1}" for i in range(1, len(distances) + 1)]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=[mean_dist] * len(x_values),
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(255, 0, 0, 0.1)',
    ))

    fig.add_trace(go.Scatter(x=x_values, y=distances, mode='lines+markers'))

    if current_image_index > 0:
        highlight_index = current_image_index
        if highlight_index <= len(distances):
            fig.add_trace(go.Scatter(
                x=[x_values[highlight_index - 1]],
                y=[distances[highlight_index - 1]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='circle-open-dot'),
            ))

    fig.add_hline(y=mean_dist, line_dash="dot", line_color="red")
    if mean_dist_without_current is not None:
        fig.add_hline(y=mean_dist_without_current, line_dash="dash", line_color="#636EFA")

    fig.update_layout(
        title_text=f"Interaction Distances for {test_id}",
        xaxis_title="Interaction Step",
        yaxis_title="Euclidean Distance (pixels)",
        xaxis=dict(type='category'),
        showlegend=False,
        plot_bgcolor='white',
        # paper_bgcolor='white',
        dragmode=False
    )
    
    return fig, stats_basic, stats_mean_wo_current, stats_score


def get_distances_for_test_id(interactions, test_id, dims):
    """Helper function to calculate distances for a given test_id."""
    if not interactions or not test_id or test_id not in interactions or not dims:
        return [], 0, 0

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
        return [], 0, 0

    distances = [calculate_euclidean_distance(interaction_points[i]['start'], interaction_points[i-1]['end'], dims) for i in range(1, len(interaction_points))]
    
    if not distances:
        return [], 0, 0

    mean_dist = pd.Series(distances).mean()
    std_dist = pd.Series(distances).std()
    
    return distances, mean_dist, std_dist

def create_comparison_plot(interactions, test_id1, test_id2, dims1, dims2, current_image_index1, current_image_index2):
    """Creates a line plot comparing distances of two test_ids."""
    
    distances1, mean_dist1, std_dist1 = get_distances_for_test_id(interactions, test_id1, dims1)
    distances2, mean_dist2, std_dist2 = get_distances_for_test_id(interactions, test_id2, dims2)

    if not distances1 and not distances2:
        fig = go.Figure()
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, annotations=[{"text": "No data to display for either Test ID.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}])
        return fig, ""

    fig = go.Figure()
    
    stats_text = f"""<b>Comparison Statistics:</b><br>
<b>{test_id1} (Red):</b><br>
Mean: {mean_dist1:.2f}, Std Dev: {std_dist1:.2f}<br>
<b>{test_id2} (Blue):</b><br>
Mean: {mean_dist2:.2f}, Std Dev: {std_dist2:.2f}"""

    # Plot for test_id1
    if distances1:
        x_values1 = [f"{i}-{i+1}" for i in range(1, len(distances1) + 1)]
        fig.add_trace(go.Scatter(x=x_values1, y=distances1, mode='lines+markers', name=test_id1, line=dict(color='red')))
        fig.add_hline(y=mean_dist1, line_dash="dot", line_color="red")
        if current_image_index1 > 0:
            highlight_index = current_image_index1
            if highlight_index <= len(distances1):
                fig.add_trace(go.Scatter(
                    x=[x_values1[highlight_index - 1]],
                    y=[distances1[highlight_index - 1]],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='circle-open-dot'),
                    name=f'{test_id1} current'
                ))

    # Plot for test_id2
    if distances2:
        x_values2 = [f"{i}-{i+1}" for i in range(1, len(distances2) + 1)]
        fig.add_trace(go.Scatter(x=x_values2, y=distances2, mode='lines+markers', name=test_id2, line=dict(color='#636EFA')))
        fig.add_hline(y=mean_dist2, line_dash="dot", line_color="#636EFA")
        if current_image_index2 > 0:
            highlight_index = current_image_index2
            if highlight_index <= len(distances2):
                fig.add_trace(go.Scatter(
                    x=[x_values2[highlight_index - 1]],
                    y=[distances2[highlight_index - 1]],
                    mode='markers',
                    marker=dict(color='blue', size=12, symbol='circle-open-dot'),
                    name=f'{test_id2} current'
                ))

    fig.update_layout(
        title_text=f"Comparison of Interaction Distances: {test_id1} vs {test_id2}",
        xaxis_title="Interaction Step",
        yaxis_title="Euclidean Distance (pixels)",
        xaxis=dict(type='category'),
        plot_bgcolor='white',
        dragmode=False,
        showlegend=False
    )
    
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
        calc_current_test_id_compare_state = gr.State("")
        calc_current_image_index_compare_state = gr.State(0)
        calc_image_dimensions_compare_state = gr.State()


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
                    with gr.Column(scale=3):
                        plot_display_simple = gr.Plot(label="Interaction Distance Plot")
                        with gr.Row():
                            with gr.Column(scale=1):
                                stats_basic_simple = gr.Markdown()
                                stats_mean_wo_current_simple = gr.Markdown()
                                
                            with gr.Column(scale=2):
                                stats_score_simple = gr.Markdown()

                gr.Markdown("---")
                gr.Markdown("### Compare with another Test ID")
                test_id_dropdown_compare = gr.Dropdown(label="Select Test ID to Compare", interactive=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        img_id_label_compare = gr.Label(label="Image ID")
                        image_display_compare = gr.Image(label="Image", interactive=False, type="pil", height=300)
                        with gr.Row():
                            prev_button_compare = gr.Button("Previous")
                            next_button_compare = gr.Button("Next")
                    with gr.Column(scale=3):
                        comparison_plot_display = gr.Plot(label="Comparison Distance Plot")
                        comparison_stats_label = gr.Markdown()

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
                return {}, folder_path, {}, gr.update(choices=[], value=None), gr.update(choices=[], value=None)
            else:
                gr.Info(f"Successfully loaded data from {folder_path}", duration=2)
                test_ids = sorted(list(image_groups.keys()))
                return image_groups, folder_path, interactions, gr.update(choices=test_ids, value=test_ids[0] if test_ids else None), gr.update(choices=test_ids, value=None)

        calc_start_button.click(
            fn=calc_start_process,
            inputs=[calc_folder_input],
            outputs=[calc_image_groups_state, calc_folder_path_state, calc_interactions_state, test_id_dropdown_simple, test_id_dropdown_compare]
        )

        def on_test_id_select_simple(test_id, image_groups, interactions):
            if not test_id or not image_groups or not interactions:
                return None, 0, None, "", None, "", "", "", gr.update(interactive=False), gr.update(interactive=False), test_id, gr.update(choices=[]), None, "", None, "", gr.update(interactive=False), gr.update(interactive=False), None, ""

            images = image_groups.get(test_id, [])
            if not images:
                return None, 0, None, "", None, "", "", "", gr.update(interactive=False), gr.update(interactive=False), test_id, gr.update(choices=[]), None, "", None, "", gr.update(interactive=False), gr.update(interactive=False), None, ""

            image_path = images[0]
            img_id = os.path.basename(image_path)
            with Image.open(image_path) as img:
                dims = img.size

            display_image = get_image_for_display(image_path, test_id, interactions, draw_trajectory=True)
            img_label = f"{img_id} (1/{len(images)})"
            
            plot, stats_basic, stats_mean_wo_current, stats_score = create_distance_plot(interactions, test_id, dims, 0)

            all_test_ids = sorted(list(image_groups.keys()))
            compare_choices = [tid for tid in all_test_ids if tid != test_id]

            return (
                display_image, 0, dims, img_label, plot, stats_basic, stats_mean_wo_current, stats_score,
                gr.update(interactive=False), # prev
                gr.update(interactive=len(images) > 1), # next
                test_id,
                gr.update(choices=compare_choices, value=None),
                # Reset compare view
                None, "", None, "", gr.update(interactive=False), gr.update(interactive=False), None, ""
            )

        test_id_dropdown_simple.change(
            fn=on_test_id_select_simple,
            inputs=[test_id_dropdown_simple, calc_image_groups_state, calc_interactions_state],
            outputs=[
                image_display_simple, calc_current_image_index_state, calc_image_dimensions_state,
                img_id_label_simple, plot_display_simple, 
                stats_basic_simple, stats_mean_wo_current_simple, stats_score_simple,
                prev_button_simple, next_button_simple,
                calc_current_test_id_state, test_id_dropdown_compare,
                image_display_compare, img_id_label_compare, comparison_plot_display, comparison_stats_label, 
                prev_button_compare, next_button_compare, calc_current_test_id_compare_state, calc_current_image_index_compare_state
            ]
        )

        def on_test_id_select_compare(test_id_compare, test_id_simple, image_groups, interactions, current_image_index_simple, dims_simple):
            if not test_id_compare:
                return None, 0, None, "", None, "", gr.update(interactive=False), gr.update(interactive=False), None

            images = image_groups.get(test_id_compare, [])
            if not images:
                return None, 0, None, "", None, "", gr.update(interactive=False), gr.update(interactive=False), None

            image_path = images[0]
            img_id = os.path.basename(image_path)
            with Image.open(image_path) as img:
                dims = img.size

            display_image = get_image_for_display(image_path, test_id_compare, interactions, draw_trajectory=True)
            img_label = f"{img_id} (1/{len(images)})"

            plot, stats = create_comparison_plot(interactions, test_id_simple, test_id_compare, dims_simple, dims, current_image_index_simple, 0)

            return (
                display_image,
                0,
                dims,
                img_label,
                plot,
                stats,
                gr.update(interactive=False),
                gr.update(interactive=len(images) > 1),
                test_id_compare
            )

        test_id_dropdown_compare.change(
            fn=on_test_id_select_compare,
            inputs=[test_id_dropdown_compare, calc_current_test_id_state, calc_image_groups_state, calc_interactions_state, calc_current_image_index_state, calc_image_dimensions_state],
            outputs=[
                image_display_compare, calc_current_image_index_compare_state, calc_image_dimensions_compare_state,
                img_id_label_compare, comparison_plot_display, comparison_stats_label, 
                prev_button_compare, next_button_compare, calc_current_test_id_compare_state
            ]
        )

        def change_image_simple(direction, test_id, index, image_groups, interactions, dims, dims_compare, test_id_compare, current_image_index_compare):
            new_index = index + direction
            images = image_groups.get(test_id, [])

            if not (0 <= new_index < len(images)):
                return gr.update(), new_index, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            image_path = images[new_index]
            img_id = os.path.basename(image_path)
            
            display_image = get_image_for_display(image_path, test_id, interactions, draw_trajectory=True)
            img_label = f"{img_id} ({new_index + 1}/{len(images)})"
            
            plot, stats_basic, stats_mean_wo_current, stats_score = create_distance_plot(interactions, test_id, dims, new_index)

            compare_plot, compare_stats = gr.update(), gr.update()
            if test_id_compare:
                compare_plot, compare_stats = create_comparison_plot(interactions, test_id, test_id_compare, dims, dims_compare, new_index, current_image_index_compare)

            return (
                display_image, new_index, img_label, plot, stats_basic, stats_mean_wo_current, stats_score,
                gr.update(interactive=new_index > 0), 
                gr.update(interactive=new_index < len(images) - 1),
                compare_plot,
                compare_stats
            )

        prev_button_simple.click(
            fn=lambda test_id, index, groups, inter, dims, dims_comp, t_id_comp, idx_comp: change_image_simple(-1, test_id, index, groups, inter, dims, dims_comp, t_id_comp, idx_comp),
            inputs=[calc_current_test_id_state, calc_current_image_index_state, calc_image_groups_state, calc_interactions_state, calc_image_dimensions_state, calc_image_dimensions_compare_state, calc_current_test_id_compare_state, calc_current_image_index_compare_state],
            outputs=[
                image_display_simple, calc_current_image_index_state, img_id_label_simple, 
                plot_display_simple, 
                stats_basic_simple, stats_mean_wo_current_simple, stats_score_simple,
                prev_button_simple, next_button_simple,
                comparison_plot_display, comparison_stats_label
            ]
        )

        next_button_simple.click(
            fn=lambda test_id, index, groups, inter, dims, dims_comp, t_id_comp, idx_comp: change_image_simple(1, test_id, index, groups, inter, dims, dims_comp, t_id_comp, idx_comp),
            inputs=[calc_current_test_id_state, calc_current_image_index_state, calc_image_groups_state, calc_interactions_state, calc_image_dimensions_state, calc_image_dimensions_compare_state, calc_current_test_id_compare_state, calc_current_image_index_compare_state],
            outputs=[
                image_display_simple, calc_current_image_index_state, img_id_label_simple, 
                plot_display_simple, 
                stats_basic_simple, stats_mean_wo_current_simple, stats_score_simple,
                prev_button_simple, next_button_simple,
                comparison_plot_display, comparison_stats_label
            ]
        )

        def change_image_compare(direction, test_id_simple, test_id_compare, index, image_groups, interactions, dims_simple, dims_compare, current_image_index_simple):
            new_index = index + direction
            images = image_groups.get(test_id_compare, [])

            if not (0 <= new_index < len(images)):
                return gr.update(), new_index, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            image_path = images[new_index]
            img_id = os.path.basename(image_path)
            
            display_image = get_image_for_display(image_path, test_id_compare, interactions)
            img_label = f"{img_id} ({new_index + 1}/{len(images)})"
            
            plot, stats = create_comparison_plot(interactions, test_id_simple, test_id_compare, dims_simple, dims_compare, current_image_index_simple, new_index)

            return (
                display_image, new_index, img_label, plot, stats,
                gr.update(interactive=new_index > 0), 
                gr.update(interactive=new_index < len(images) - 1)
            )

        prev_button_compare.click(
            fn=lambda t_id_simple, t_id_comp, idx, groups, inter, dims_simple, dims_comp, idx_simple: change_image_compare(-1, t_id_simple, t_id_comp, idx, groups, inter, dims_simple, dims_comp, idx_simple),
            inputs=[calc_current_test_id_state, calc_current_test_id_compare_state, calc_current_image_index_compare_state, calc_image_groups_state, calc_interactions_state, calc_image_dimensions_state, calc_image_dimensions_compare_state, calc_current_image_index_state],
            outputs=[image_display_compare, calc_current_image_index_compare_state, img_id_label_compare, comparison_plot_display, comparison_stats_label, prev_button_compare, next_button_compare]
        )

        next_button_compare.click(
            fn=lambda t_id_simple, t_id_comp, idx, groups, inter, dims_simple, dims_comp, idx_simple: change_image_compare(1, t_id_simple, t_id_comp, idx, groups, inter, dims_simple, dims_comp, idx_simple),
            inputs=[calc_current_test_id_state, calc_current_test_id_compare_state, calc_current_image_index_compare_state, calc_image_groups_state, calc_interactions_state, calc_image_dimensions_state, calc_image_dimensions_compare_state, calc_current_image_index_state],
            outputs=[image_display_compare, calc_current_image_index_compare_state, img_id_label_compare, comparison_plot_display, comparison_stats_label, prev_button_compare, next_button_compare]
        )