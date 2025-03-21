# Ollama Chat Interface with Gradio

This application provides a web interface for interacting with Ollama language models using Gradio.

## Prerequisites

- Python 3.8+
- Ollama installed and running locally (or accessible via network)

## Setup

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install gradio requests python-dotenv
```

2. Make sure Ollama is running:

```bash
ollama serve
```

3. (Optional) Create a `.env` file to configure the Ollama API URL:

```
OLLAMA_API_URL=http://localhost:11434
```

## Usage

1. Run the application (make sure your virtual environment is activated):

```bash
source .venv/bin/activate  # If not already activated
python app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

3. Select a model from the dropdown, adjust parameters if needed, and start chatting!

## Features

- Chat with any model available in your Ollama installation
- Adjust temperature and max tokens for response generation
- Clear conversation history with a single click

## Troubleshooting

- If no models appear in the dropdown, ensure Ollama is running and accessible
- Check that you have at least one model pulled in Ollama (`ollama list`)
- To pull a model: `ollama pull modelname`
