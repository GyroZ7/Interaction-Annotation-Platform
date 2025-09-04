import gradio as gr
from annotation_tab import annotation_tab
from calculate_tab import calculate_tab

def create_app():
    with gr.Blocks() as app:
        gr.HTML("""<style>
        .gr-image { pointer-events: none; }
        </style>""")
        
        with gr.Tabs() as main_tabs:
            annotation_tab()
            calculate_tab()
            
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()