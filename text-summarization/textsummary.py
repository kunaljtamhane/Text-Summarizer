import torch
import gradio as gr
from transformers import pipeline

# Load the summarization pipeline using a Hugging Face model
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# Function to summarize input text
def summary(input_text):
    output = text_summary(input_text)
    return output[0]['summary_text']

# Close any previously running Gradio demos
gr.close_all()

# Create the Gradio interface
demo = gr.Interface(
    fn=summary,
    inputs=gr.Textbox(label="Input text to summarize", lines=6),
    outputs=gr.Textbox(label="Summarized text", lines=4),
    title="Text to Summary",
    description="This application summarizes text for you."
)

demo.launch()
