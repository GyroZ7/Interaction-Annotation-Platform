import gradio as gr
from PIL import Image
from utils import get_image_for_display, get_test_folders, process_folder
import os
import json

def annotation_tab():
    with gr.TabItem("Interaction Annotate"):
        # States for annotation tab
        image_groups_state = gr.State({})
        current_test_id_state = gr.State("")
        current_image_index_state = gr.State(0)
        image_dimensions_state = gr.State()
        interactions_state = gr.State({})
        folder_path_state = gr.State("")

        with gr.Row():
            folder_input = gr.Dropdown(label="Select Test Folder", choices=get_test_folders(), interactive=True)
            start_button = gr.Button("Start")

        with gr.Row():
            with gr.Column(scale=3):
                # You can adjust the height to change the size of the image display.
                image_display = gr.Image(label="Image", interactive=True, type="pil", height=512)
            with gr.Column(scale=1):
                test_id_dropdown = gr.Dropdown(label="Test ID", interactive=True)
                img_id_label = gr.Label(label="Image ID")
                with gr.Row():
                    prev_button = gr.Button("Previous")
                    next_button = gr.Button("Next")
                with gr.Group():
                    tool_selector = gr.Radio(["click", "multiclick", "longpress", "slide"], label="Tool", value="click")
                    multiclick_clicks = gr.Number(label="Clicks", value=2, interactive=True, visible=False, precision=0)
                    longpress_duration = gr.Number(label="Duration (ms)", value=1000, interactive=True, visible=False, precision=0)
                    slide_duration = gr.Number(label="Duration (ms)", value=1000, interactive=True, visible=False, precision=0)
                    grounding_label = gr.Textbox(label="Grounding", interactive=False)
                export_button = gr.Button("Export Interaction")

        def handle_tool_change(tool_type):
            return {
                multiclick_clicks: gr.update(visible=tool_type == 'multiclick'),
                longpress_duration: gr.update(visible=tool_type == 'longpress'),
                slide_duration: gr.update(visible=tool_type == 'slide'),
            }

        tool_selector.change(
            handle_tool_change,
            [tool_selector],
            [multiclick_clicks, longpress_duration, slide_duration]
        )

        def handle_image_click(evt: gr.SelectData, dims, interactions, test_id, image_groups, index, tool_type, clicks, duration, slide_duration):
            if tool_type not in ['click', 'multiclick', 'longpress', 'slide'] or not dims or not test_id:
                current_image_path = image_groups[test_id][index]
                display_image = get_image_for_display(current_image_path, test_id, interactions)
                return interactions, grounding_label.value, display_image, gr.update(), gr.update(), gr.update()

            width, height = dims
            norm_x = evt.index[0] / width
            norm_y = evt.index[1] / height
            new_grounding_point = [norm_x, norm_y]

            img_path = image_groups[test_id][index]
            img_id = os.path.basename(img_path)

            if test_id not in interactions:
                interactions[test_id] = {}

            current_interaction = interactions.get(test_id, {}).get(img_id, {})
            if current_interaction.get("interaction_type") != tool_type:
                current_interaction = {
                    "interaction_type": tool_type,
                    "interaction_parameters": {}
                }

            interaction_params = current_interaction.get("interaction_parameters", {})

            if tool_type == 'slide':
                existing_grounding = interaction_params.get("grounding", [])
                if not isinstance(existing_grounding, list) or (existing_grounding and not isinstance(existing_grounding[0], list)):
                    existing_grounding = []

                if len(existing_grounding) < 2:
                    existing_grounding.append(new_grounding_point)
                else:
                    existing_grounding[0] = existing_grounding[1]
                    existing_grounding[1] = new_grounding_point
                
                interaction_params["grounding"] = existing_grounding
                interaction_params['duration'] = slide_duration
                if 'clicks' in interaction_params: del interaction_params['clicks']
                grounding_text = ", ".join([f"({p[0]:.4f}, {p[1]:.4f})" for p in existing_grounding])
            else:
                interaction_params["grounding"] = new_grounding_point
                grounding_text = f"({norm_x:.4f}, {norm_y:.4f})"
                if 'duration' in interaction_params: del interaction_params['duration']
                if 'clicks' in interaction_params: del interaction_params['clicks']

                if tool_type == 'multiclick':
                    interaction_params['clicks'] = clicks
                elif tool_type == 'longpress':
                    interaction_params['duration'] = duration

            interactions[test_id][img_id] = {
                "interaction_type": tool_type,
                "interaction_parameters": interaction_params
            }
            
            display_image = get_image_for_display(img_path, test_id, interactions)

            images = image_groups.get(test_id, [])
            disable_buttons = tool_type == 'slide' and len(interaction_params.get("grounding", [])) == 1
            
            prev_interactive = (index > 0) and not disable_buttons
            next_interactive = (index < len(images) - 1) and not disable_buttons
            export_interactive = not disable_buttons

            return interactions, grounding_text, display_image, gr.update(interactive=prev_interactive), gr.update(interactive=next_interactive), gr.update(interactive=export_interactive)

        image_display.select(
            handle_image_click, 
            [image_dimensions_state, interactions_state, current_test_id_state, image_groups_state, current_image_index_state, tool_selector, multiclick_clicks, longpress_duration, slide_duration],
            [interactions_state, grounding_label, image_display, prev_button, next_button, export_button]
        )

        def start_process(folder_path):
            images, test_id, message, image_groups, dims, interactions = process_folder(folder_path)

            test_ids = sorted(list(image_groups.keys()))
            
            if not images:
                gr.Warning("No valid data found in the selected folder.", duration=2)
                return {}, "", 0, gr.update(choices=[], value=None), None, None, {}, folder_path, "", gr.update(interactive=False), gr.update(interactive=False)
            
            gr.Info(f"Successfully loaded data from {folder_path}", duration=2)
            first_image_path = images[0]
            img_id = os.path.basename(first_image_path)
            display_image = get_image_for_display(first_image_path, test_id, interactions)
            img_label = f"{img_id} (1/{len(images)})"

            return (
                image_groups,
                test_id,
                0,
                gr.update(choices=test_ids, value=test_id),
                display_image,
                dims,
                interactions,
                folder_path,
                img_label,
                gr.update(interactive=False), # Disable prev
                gr.update(interactive=len(images) > 1) # Enable next if more than 1 image
            )

        start_button.click(
            fn=start_process,
            inputs=[folder_input],
            outputs=[
                image_groups_state,
                current_test_id_state,
                current_image_index_state,
                test_id_dropdown,
                image_display,
                image_dimensions_state,
                interactions_state,
                folder_path_state,
                img_id_label,
                prev_button,
                next_button
            ],
        )

        def update_gallery(test_id, image_groups, interactions):
            if not test_id or not image_groups:
                return None, 0, None, "", interactions, "", gr.update(interactive=False), gr.update(interactive=False), "click", 2, 1000, 1000, test_id, gr.update(interactive=True)

            images = image_groups.get(test_id, [])
            if not images:
                return None, 0, None, "", interactions, "", gr.update(interactive=False), gr.update(interactive=False), "click", 2, 1000, 1000, test_id, gr.update(interactive=True)
            
            image_path = images[0]
            img_id = os.path.basename(image_path)
            with Image.open(image_path) as img:
                dims = img.size

            tool_type = "click"
            clicks = 2
            duration = 1000
            slide_duration = 1000
            grounding_text = ""
            disable_buttons = False

            if test_id in interactions and img_id in interactions.get(test_id, {}):
                interaction_data = interactions[test_id][img_id]
                tool_type = interaction_data.get("interaction_type", "click")
                interaction_params = interaction_data.get("interaction_parameters", {})
                grounding = interaction_params.get("grounding")

                if grounding:
                    if tool_type == 'slide':
                        grounding_text = ", ".join([f"({p[0]:.4f}, {p[1]:.4f})" for p in grounding])
                        if len(grounding) == 1:
                            disable_buttons = True
                    else:
                        grounding_text = f"({grounding[0]:.4f}, {grounding[1]:.4f})"

                if tool_type == 'multiclick':
                    clicks = interaction_params.get('clicks', 2)
                elif tool_type == 'longpress':
                    duration = interaction_params.get('duration', 1000)
                elif tool_type == 'slide':
                    slide_duration = interaction_params.get('duration', 1000)

            display_image = get_image_for_display(image_path, test_id, interactions)
            img_label = f"{img_id} (1/{len(images)})"

            return (
                display_image, 0, dims, grounding_text, interactions, img_label, 
                gr.update(interactive=False), gr.update(interactive=len(images) > 1 and not disable_buttons),
                tool_type, clicks, duration, slide_duration, test_id, gr.update(interactive=not disable_buttons)
            )

        test_id_dropdown.change(
            fn=update_gallery,
            inputs=[test_id_dropdown, image_groups_state, interactions_state],
            outputs=[
                image_display, current_image_index_state, image_dimensions_state, 
                grounding_label, interactions_state, img_id_label, prev_button, next_button,
                tool_selector, multiclick_clicks, longpress_duration, slide_duration, current_test_id_state, export_button
            ]
        )

        def change_image(direction, test_id, index, image_groups, interactions, tool_type, clicks, duration, slide_duration):
            new_index = index + direction
            images = image_groups.get(test_id, [])

            if not (0 <= new_index < len(images)):
                return gr.update(), new_index, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            image_path = images[new_index]
            img_id = os.path.basename(image_path)
            
            tool_type = "click"
            clicks = 2
            duration = 1000
            slide_duration = 1000
            grounding_text = ""
            disable_buttons = False

            if test_id in interactions and img_id in interactions.get(test_id, {}):
                interaction_data = interactions[test_id][img_id]
                tool_type = interaction_data.get("interaction_type", "click")
                interaction_params = interaction_data.get("interaction_parameters", {})
                grounding = interaction_params.get("grounding")

                if grounding:
                    if tool_type == 'slide':
                        grounding_text = ", ".join([f"({p[0]:.4f}, {p[1]:.4f})" for p in grounding])
                        if len(grounding) == 1:
                            disable_buttons = True
                    else:
                        grounding_text = f"({grounding[0]:.4f}, {grounding[1]:.4f})"

                if tool_type == 'multiclick':
                    clicks = interaction_params.get('clicks', 2)
                elif tool_type == 'longpress':
                    duration = interaction_params.get('duration', 1000)
                elif tool_type == 'slide':
                    slide_duration = interaction_params.get('duration', 1000)

            display_image = get_image_for_display(image_path, test_id, interactions)
            img_label = f"{img_id} ({new_index + 1}/{len(images)})"

            return (
                display_image, new_index, img_label, grounding_text, 
                gr.update(interactive=new_index > 0 and not disable_buttons), 
                gr.update(interactive=new_index < len(images) - 1 and not disable_buttons),
                tool_type, clicks, duration, slide_duration, gr.update(interactive=not disable_buttons)
            )

        prev_button.click(
            fn=lambda test_id, index, image_groups, interactions, tool_type, clicks, duration, slide_duration: change_image(-1, test_id, index, image_groups, interactions, tool_type, clicks, duration, slide_duration),
            inputs=[current_test_id_state, current_image_index_state, image_groups_state, interactions_state, tool_selector, multiclick_clicks, longpress_duration, slide_duration],
            outputs=[
                image_display, current_image_index_state, img_id_label, grounding_label, 
                prev_button, next_button, tool_selector, multiclick_clicks, longpress_duration, slide_duration, export_button
            ]
        )

        next_button.click(
            fn=lambda test_id, index, image_groups, interactions, tool_type, clicks, duration, slide_duration: change_image(1, test_id, index, image_groups, interactions, tool_type, clicks, duration, slide_duration),
            inputs=[current_test_id_state, current_image_index_state, image_groups_state, interactions_state, tool_selector, multiclick_clicks, longpress_duration, slide_duration],
            outputs=[
                image_display, current_image_index_state, img_id_label, grounding_label, 
                prev_button, next_button, tool_selector, multiclick_clicks, longpress_duration, slide_duration, export_button
            ]
        )

        def export_interactions(interactions, folder_path):
            if not folder_path or not interactions:
                gr.Warning("No interactions to export!", duration=2)
                return

            base_folder_path = os.path.join("test_folder", folder_path)
            export_dir = os.path.join(base_folder_path, "test_img")
            
            try:
                os.makedirs(export_dir, exist_ok=True)
                export_path = os.path.join(export_dir, "interactions.json")
                with open(export_path, 'w') as f:
                    json.dump(interactions, f, indent=4)
                gr.Info(f"Interactions exported to {export_path}", duration=2)
            except Exception as e:
                gr.Warning(f"Error exporting interactions: {e}", duration=2)

        export_button.click(
            export_interactions,
            [interactions_state, folder_path_state],
            []
        )