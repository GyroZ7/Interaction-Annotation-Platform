import gradio as gr
import os
import json
import math
from PIL import Image, ImageDraw, ImageColor

def get_test_folders(base_dir="test_folder"):
    if not os.path.isdir(base_dir):
        return []
    return [f.name for f in os.scandir(base_dir) if f.is_dir()]

def draw_point_on_image(image_path, normalized_coords, color="red", interaction_type="click"):
    with Image.open(image_path) as base_img:
        base_img = base_img.convert("RGBA")

        # Create a transparent overlay for drawing
        overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        width, height = base_img.size
        
        # You can adjust the radius to change the size of the point.
        total_radius = 100
        solid_radius = 25
        
        try:
            rgb_color = ImageColor.getrgb(color)
        except ValueError:
            rgb_color = (255, 0, 0) # Default to red

        coords_to_draw = []
        if normalized_coords:
            # Check if it's a list of points e.g. [[0.5, 0.5], [0.6, 0.6]]
            if isinstance(normalized_coords, list) and len(normalized_coords) > 0 and isinstance(normalized_coords[0], (list, tuple)):
                coords_to_draw = normalized_coords
            else: # A single point e.g. [0.5, 0.5] or (0.5, 0.5)
                coords_to_draw = [normalized_coords]

        for coords in coords_to_draw:
            x = coords[0] * width
            y = coords[1] * height
            # The following code creates a pseudo-blur effect with a solid center.
            
            # Draw the blurred part by drawing semi-transparent concentric circles.
            for i in range(total_radius, 0, -1):
                # To make the blur lighter, the exponent is changed from 2 to 1.5
                alpha = int(255 * (1 - (i / total_radius))**1.5)
                fill_color = rgb_color + (alpha,)
                draw.ellipse((x - i, y - i, x + i, y + i), fill=fill_color, outline=None)

            # Draw the solid inner circle on top of the blur.
            draw.ellipse((x - solid_radius, y - solid_radius, x + solid_radius, y + solid_radius), fill=rgb_color + (255,), outline=None)

        # Draw arrow for slide interaction
        if interaction_type == 'slide' and len(coords_to_draw) == 2:
            start_point_norm = coords_to_draw[0]
            end_point_norm = coords_to_draw[1]

            x1 = start_point_norm[0] * width
            y1 = start_point_norm[1] * height
            x2 = end_point_norm[0] * width
            y2 = end_point_norm[1] * height

            # --- Adjustable arrow parameters ---
            # You can adjust the width of the arrow line.
            arrow_line_width = 10
            # You can adjust the length of the arrowhead.
            arrowhead_length = 80
            # You can adjust the angle of the arrowhead.
            arrowhead_angle = math.pi / 8 # 22.5 degrees
            # --- End of adjustable parameters ---

            # Draw line
            draw.line([(x1, y1), (x2, y2)], fill=rgb_color + (255,), width=arrow_line_width)

            # Draw arrowhead
            angle = math.atan2(y1 - y2, x1 - x2)
            
            x_arrow1 = x2 + arrowhead_length * math.cos(angle - arrowhead_angle)
            y_arrow1 = y2 + arrowhead_length * math.sin(angle - arrowhead_angle)
            x_arrow2 = x2 + arrowhead_length * math.cos(angle + arrowhead_angle)
            y_arrow2 = y2 + arrowhead_length * math.sin(angle + arrowhead_angle)

            draw.polygon([(x2, y2), (x_arrow1, y_arrow1), (x_arrow2, y_arrow2)], fill=rgb_color + (255,))

        # Alpha composite the overlay onto the original image
        combined = Image.alpha_composite(base_img, overlay)
        
        return combined.convert("RGB")

def get_image_for_display(image_path, test_id, interactions):
    img_id = os.path.basename(image_path)
    if test_id in interactions and img_id in interactions[test_id]:
        interaction_data = interactions[test_id][img_id]
        if "interaction_parameters" in interaction_data and "grounding" in interaction_data["interaction_parameters"]:
            coords = interaction_data["interaction_parameters"]["grounding"]
            if not coords:
                return Image.open(image_path)
            interaction_type = interaction_data.get("interaction_type", "click")
            color = "red" # default
            if interaction_type == 'multiclick':
                color = "orange"
            elif interaction_type == 'longpress':
                color = "blue"
            elif interaction_type == 'slide':
                color = "green"
            return draw_point_on_image(image_path, coords, color, interaction_type=interaction_type)
    return Image.open(image_path)

def update_and_get_interactions(interactions, test_id, index, image_groups, tool_type, clicks, duration, slide_duration):
    if not test_id or not image_groups:
        return interactions
    images = image_groups.get(test_id, [])
    if not images or not (0 <= index < len(images)):
        return interactions

    img_path = images[index]
    img_id = os.path.basename(img_path)

    if test_id in interactions and img_id in interactions.get(test_id, {}):
        # Update existing interaction
        interaction_data = interactions[test_id][img_id]
        
        # Only update if there is a grounding point
        if "grounding" not in interaction_data.get("interaction_parameters", {}) or not interaction_data.get("interaction_parameters", {}).get("grounding"):
            return interactions

        interaction_data['interaction_type'] = tool_type
        
        params = interaction_data.get('interaction_parameters', {})
        if tool_type == 'multiclick':
            params['clicks'] = clicks
            if 'duration' in params:
                del params['duration']
        elif tool_type == 'longpress':
            params['duration'] = duration
            if 'clicks' in params:
                del params['clicks']
        elif tool_type == 'slide':
            params['duration'] = slide_duration
            if 'clicks' in params:
                del params['clicks']
        else: # click
             if 'duration' in params:
                del params['duration']
             if 'clicks' in params:
                del params['clicks']
        interaction_data['interaction_parameters'] = params

    return interactions


def process_folder(folder_name):
    if not folder_name:
        return [], "", "Please select a folder.", {}, None, {}

    base_folder_path = os.path.join("test_folder", folder_name)
    if not os.path.isdir(base_folder_path):
        return [], "", "Please provide a valid folder path.", {}, None, {}

    # Load interactions from the interactions.json file in the test_img directory
    interactions = {}
    interaction_file = os.path.join(base_folder_path, "test_img", "interactions.json")
    if os.path.isfile(interaction_file):
        with open(interaction_file, 'r') as f:
            try:
                interactions = json.load(f)
            except json.JSONDecodeError:
                pass # Ignore if file is empty or corrupt

    test_ids_dir = os.path.join(base_folder_path, "test_img")
    if not os.path.isdir(test_ids_dir):
        return [], "", f"'test_img' directory not found in '{folder_name}'.", {}, None, interactions

    subfolders = [f.path for f in os.scandir(test_ids_dir) if f.is_dir()]
    if not subfolders:
        return [], "", "The 'test_img' directory has no subfolders.", {}, None, interactions

    image_groups = {}
    for subfolder in subfolders:
        test_id = os.path.basename(subfolder)
        imgs_folder = os.path.join(subfolder, "imgs")
        if os.path.isdir(imgs_folder):
            images = sorted([os.path.join(imgs_folder, img) for img in os.listdir(imgs_folder)])
            if images:
                image_groups[test_id] = images

    if not image_groups:
        return [], "", "No subfolders with an 'imgs' directory found in 'test_img'.", {}, None, interactions

    test_ids = sorted(list(image_groups.keys()))
    first_test_id = test_ids[0]
    first_image_path = image_groups[first_test_id][0]
    with Image.open(first_image_path) as img:
        dims = img.size
    
    return image_groups[first_test_id], first_test_id, f"Displaying images for {first_test_id}", image_groups, dims, interactions

def create_app():
    with gr.Blocks() as app:
        gr.HTML("""<style>
        .gr-image { pointer-events: none; }
        </style>""")
        # States
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
                image_display = gr.Image(label="Image", interactive=True, type="pil", height=512, sources=[])
            with gr.Column(scale=1):
                test_id_dropdown = gr.Dropdown(label="Test ID", interactive=True)
                img_id_label = gr.Label(label="Image ID")
                with gr.Row():
                    prev_button = gr.Button("Previous")
                    next_button = gr.Button("Next")
                with gr.Group("Interaction Parameters"):
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
                return {}, "", 0, gr.update(choices=[], value=None), None, None, {}, folder_path, "", gr.update(interactive=False), gr.update(interactive=False)

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
                gr.update(value=tool_type),
                gr.update(value=clicks, visible=tool_type=='multiclick'),
                gr.update(value=duration, visible=tool_type=='longpress'),
                gr.update(value=slide_duration, visible=tool_type=='slide'),
                test_id,
                gr.update(interactive=not disable_buttons)
            )

        test_id_dropdown.change(
            fn=update_gallery,
            inputs=[test_id_dropdown, image_groups_state, interactions_state],
            outputs=[
                image_display, current_image_index_state, image_dimensions_state, 
                grounding_label, interactions_state, img_id_label, prev_button, next_button,
                tool_selector, multiclick_clicks, longpress_duration, slide_duration,
                current_test_id_state, export_button
            ],
        )

        def change_image(image_groups, test_id, current_index, direction, interactions, tool_type, clicks, duration, slide_duration):
            # First, update interactions for the *current* image
            interactions = update_and_get_interactions(interactions, test_id, current_index, image_groups, tool_type, clicks, duration, slide_duration)

            if not test_id or not image_groups:
                return gr.update(), gr.update(), gr.update(), gr.update(), interactions, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            images = image_groups.get(test_id, [])
            if not images:
                return gr.update(), gr.update(), gr.update(), gr.update(), interactions, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            new_index = current_index + direction

            if 0 <= new_index < len(images):
                image_path = images[new_index]
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
                img_label = f"{img_id} ({new_index + 1}/{len(images)})"
                
                prev_interactive = (new_index > 0) and not disable_buttons
                next_interactive = (new_index < len(images) - 1) and not disable_buttons
                export_interactive = not disable_buttons

                return (
                    display_image, new_index, dims, grounding_text, interactions, img_label, 
                    gr.update(interactive=prev_interactive), gr.update(interactive=next_interactive),
                    gr.update(value=tool_type),
                    gr.update(value=clicks, visible=tool_type=='multiclick'),
                    gr.update(value=duration, visible=tool_type=='longpress'),
                    gr.update(value=slide_duration, visible=tool_type=='slide'),
                    gr.update(interactive=export_interactive)
                )
            
            # If index is out of bounds, do not change anything
            return gr.update(), gr.update(), gr.update(), gr.update(), interactions, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        prev_button.click(
            fn=lambda g, t, i, inter, tool, c, d, sd: change_image(g, t, i, -1, inter, tool, c, d, sd),
            inputs=[
                image_groups_state,
                current_test_id_state,
                current_image_index_state,
                interactions_state,
                tool_selector,
                multiclick_clicks,
                longpress_duration,
                slide_duration
            ],
            outputs=[
                image_display, current_image_index_state, image_dimensions_state, grounding_label, 
                interactions_state, img_id_label, prev_button, next_button,
                tool_selector, multiclick_clicks, longpress_duration, slide_duration, export_button
            ],
        )

        next_button.click(
            fn=lambda g, t, i, inter, tool, c, d, sd: change_image(g, t, i, 1, inter, tool, c, d, sd),
            inputs=[
                image_groups_state,
                current_test_id_state,
                current_image_index_state,
                interactions_state,
                tool_selector,
                multiclick_clicks,
                longpress_duration,
                slide_duration
            ],
            outputs=[
                image_display, current_image_index_state, image_dimensions_state, grounding_label, 
                interactions_state, img_id_label, prev_button, next_button,
                tool_selector, multiclick_clicks, longpress_duration, slide_duration, export_button
            ],
        )

        def export_interactions(interactions, folder_path, test_id, index, image_groups, tool_type, clicks, duration, slide_duration):
            updated_interactions = update_and_get_interactions(interactions, test_id, index, image_groups, tool_type, clicks, duration, slide_duration)
            
            if not updated_interactions or not folder_path:
                gr.Warning("No interactions to export or folder path is not set.")
                return interactions
            
            output_path = os.path.join("test_folder", folder_path, "test_img", "interactions.json")
            try:
                with open(output_path, "w") as f:
                    json.dump(updated_interactions, f, indent=4)
                gr.Info(f"Interactions exported successfully to {output_path}")
            except Exception as e:
                gr.Error(f"Failed to export interactions: {e}")
            
            return updated_interactions

        export_button.click(
            fn=export_interactions,
            inputs=[
                interactions_state,
                folder_path_state,
                current_test_id_state,
                current_image_index_state,
                image_groups_state,
                tool_selector,
                multiclick_clicks,
                longpress_duration,
                slide_duration
            ],
            outputs=[interactions_state]
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()