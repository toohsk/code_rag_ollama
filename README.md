# Code RAG with Ollama

This application provides a Retrieval-Augmented Generation (RAG) system for code repositories using Ollama language models and Gradio. It allows you to load a local code repository, index it, and then ask questions about the code.

## Prerequisites

- Python 3.11+
- Ollama installed and running locally (or accessible via network)

## Setup

### Installing Dependencies

#### Option 1: Using Poetry (Recommended)

1. Make sure you have Poetry installed. If not, follow the instructions at [Python Poetry's official website](https://python-poetry.org/docs/#installation).

2. Configure Poetry to create the virtual environment inside the project directory:

```bash
poetry config virtualenvs.in-project true
```

3. Install dependencies:

```bash
poetry install --no-root
```

This will create a `.venv` directory inside your project and install all dependencies there.

#### Option 2: Using pip and venv

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install gradio requests python-dotenv langchain langchain-community faiss-cpu sentence-transformers pygments unstructured
```

### Setting Up Ollama

1. Make sure Ollama is installed and running:

```bash
ollama serve
```

2. Download a recommended model. For best results with code-related questions, we recommend using Gemma 3 models:

```bash
# For a balance of performance and resource usage
ollama pull gemma3:4b

# For better performance if you have sufficient resources
ollama pull gemma3:12b
```

3. (Optional) Create a `.env` file to configure the Ollama API URL:

```
OLLAMA_API_URL=http://localhost:11434
```

## Usage

1. Run the application:

   **If using Poetry:**

   ```bash
   poetry run python app.py
   ```

   Or activate the Poetry shell first and then run:

   ```bash
   poetry shell
   python app.py
   ```

   **If using venv:**

   ```bash
   source .venv/bin/activate  # If not already activated
   python app.py
   ```

2. Open your browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

3. Load a code repository:
   - Go to the "Repository Loader" tab
   - Enter the full path to your repository or select from common directories
   - Click "Browse Repositories" to see available repositories in the selected directory
   - Select a repository from the dropdown (if available)
   - Click "Load Repository" to index the code

4. Ask questions about the code:
   - Go to the "Code Q&A" tab
   - Type your question in the input box
   - View the answer and the relevant code snippets that were used to generate the answer

## Features

- Load and index local code repositories
- Browse available repositories in common directories
- Ask questions about code and get contextual answers
- View relevant code snippets used to generate answers
- Adjust model parameters (temperature, max tokens)
- Support for multiple programming languages
- Clear conversation history with a single click

## Supported File Types

The application supports a wide range of file extensions including:

- Python (.py)
- JavaScript/TypeScript (.js, .jsx, .ts, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- And many more (see the CODE_EXTENSIONS list in the code)

## Troubleshooting

- If no models appear in the dropdown, ensure Ollama is running and accessible
- Check that you have at least one model pulled in Ollama (`ollama list`)
- To pull a model: `ollama pull modelname`
- If you encounter memory issues when loading large repositories, try using a smaller repository or adjust the chunk size in the code
