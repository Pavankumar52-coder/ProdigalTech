from transformers import pipeline
import gradio as gr

# Load pre-trained models
def load_model(task):
    if task == "text-generation":
        return pipeline(task, model="gpt2")
    elif task == "question-answering":
        return pipeline(task, model="distilbert-base-cased-distilled-squad")
    elif task == "summarization":
        return pipeline(task, model="facebook/bart-large-cnn")

generator = load_model("text-generation")
qa_model = load_model("question-answering")
summarization_model = load_model("summarization")

# Response generation
def generate_response(prompt, task_type, context=None):
    if not prompt.strip():
        return "Please provide a valid prompt."
    if task_type == "General Text Generation":
        return generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
    elif task_type == "Q&A":
        if not context or not context.strip():
            return "Please provide a context for the question."
        return qa_model(question=prompt, context=context)["answer"]
    elif task_type == "Summarization":
        return summarization_model(prompt, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
    return "Invalid task type selected."

# Gradio app interface
task_types = ["General Text Generation", "Q&A", "Summarization"]
ui = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Enter your prompt (or question for Q&A)"),
        gr.Radio(task_types, label="Select Task Type"),
        gr.Textbox(label="Provide context for Q&A (leave blank for other tasks)"),  # No 'optional' parameter
    ],
    outputs="text",
    title="LLM Tool",
    description="Perform text generation, Q&A, and summarization with Hugging Face Transformers."
)

# Launch the app
if __name__ == "__main__":
    ui.launch()
