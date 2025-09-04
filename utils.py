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
            images = sorted([os.path.join(imgs_folder, img) for img in os.listdir(imgs_folder) if not img.startswith('.')])
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