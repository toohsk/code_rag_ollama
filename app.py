import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama API endpoint
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

def get_available_models():
    """Get a list of available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            return ["Failed to fetch models"]
    except Exception as e:
        return [f"Error: {str(e)}"]

def generate_response(prompt, model, temperature=0.7, max_tokens=1024):
    """Generate a response from Ollama"""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False
        }
        
        response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload)
        
        if response.status_code == 200:
            return response.json().get("response", "No response generated")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def chat_with_ollama(message, history, model, temperature, max_tokens):
    """Chat with Ollama model and return the response"""
    # Prepare conversation history in a format Ollama can understand
    conversation = "\n".join([f"User: {user}\nAssistant: {assistant}" for user, assistant in history])
    
    # Add the current message
    if conversation:
        conversation += f"\nUser: {message}"
    else:
        conversation = f"User: {message}"
    
    # Generate response
    response = generate_response(conversation, model, temperature, max_tokens)
    
    return response

# Create Gradio interface
with gr.Blocks(title="Ollama Chat Interface") as demo:
    gr.Markdown("# Ollama Chat Interface")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Message", placeholder="Type your message here...", lines=3)
            clear = gr.Button("Clear Conversation")
            
        with gr.Column(scale=1):
            models = get_available_models()
            model_dropdown = gr.Dropdown(choices=models, value=models[0] if models else None, label="Model")
            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(minimum=64, maximum=4096, value=1024, step=64, label="Max Tokens")
    
    def respond(message, chat_history, model, temperature, max_tokens):
        bot_message = chat_with_ollama(message, chat_history, model, temperature, max_tokens)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot, model_dropdown, temperature, max_tokens], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the app
if __name__ == "__main__":
    demo.launch()
