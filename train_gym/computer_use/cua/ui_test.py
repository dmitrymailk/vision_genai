#!/usr/bin/env python3
"""
Simple example script for the Computer-Use Agent Gradio UI.

This script launches the advanced Gradio UI for the Computer-Use Agent
with full model selection and configuration options.
It can be run directly from the command line.
"""


# from utils import load_dotenv_files

# load_dotenv_files()
import os

os.environ["OPENAI_API_KEY"] = "qwe"
# os.environ["OPENAI_API_BASE"] = "http://172.17.0.1:1337/v1"
os.environ["OPENAI_API_BASE"] = "http://172.17.0.1:1234/v1"

# Import the create_gradio_ui function
from agent.ui.gradio.ui_components import create_gradio_ui

if __name__ == "__main__":
    print("Launching Computer-Use Agent Gradio UI with advanced features...")
    app = create_gradio_ui()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
