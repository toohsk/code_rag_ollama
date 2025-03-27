import os
import json
import shutil
import tempfile
import requests
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
import concurrent.futures
import time

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()

# Ollama API endpoint
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

# Global variable to store the vector store
vector_store = None
repo_path = None
code_files = []

# File extensions to consider as code files
CODE_EXTENSIONS = [
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rb", ".php", ".swift", ".kt", ".rs", ".scala", ".sh",
    ".html", ".css", ".scss", ".sass", ".less", ".json", ".xml", ".yaml", ".yml",
    ".md", ".sql", ".graphql", ".proto", ".dart", ".lua", ".r", ".pl", ".ex", ".exs"
]

# Files and directories to ignore
IGNORE_PATTERNS = [
    ".git", "__pycache__", "node_modules", "venv", ".venv", "env", ".env",
    "dist", "build", ".idea", ".vscode", ".DS_Store", "*.pyc", "*.pyo", "*.pyd",
    "*.so", "*.dylib", "*.dll", "*.exe", "*.bin", "*.obj", "*.o"
]

def get_available_models():
    """Get a list of available models from Ollama with retry logic"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                if model_names:
                    print(f"Successfully fetched {len(model_names)} models from Ollama")
                    return model_names
                else:
                    print("No models found in Ollama, returning default models")
                    return ["llama3", "codellama", "mistral", "deepseek-coder"]  # Default models
            else:
                print(f"Error response from Ollama: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return ["llama3", "codellama", "mistral", "deepseek-coder"]  # Default models
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error to Ollama (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Max retries reached, returning default models")
                return ["llama3", "codellama", "mistral", "deepseek-coder"]  # Default models
        except Exception as e:
            print(f"Error fetching models from Ollama: {str(e)}")
            return ["llama3", "codellama", "mistral", "deepseek-coder"]  # Default models

def generate_response(prompt, model, temperature=0.7, max_tokens=1024):
    """Generate a response from Ollama with retry logic"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    # Validate and prepare the payload
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False
        }
    except ValueError as e:
        return f"Error in parameters: {str(e)}"
    
    # Try to connect to Ollama with retries
    for attempt in range(max_retries):
        try:
            print(f"Sending request to Ollama (attempt {attempt+1}/{max_retries})")
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate", 
                json=payload,
                timeout=30  # Longer timeout for generation
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "No response generated")
                print(f"Successfully generated response ({len(result)} chars)")
                return result
            else:
                print(f"Error response from Ollama: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return f"Error: Unable to generate response after {max_retries} attempts. Status code: {response.status_code}"
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error to Ollama (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return f"Error: Unable to connect to Ollama after {max_retries} attempts. Please make sure Ollama is running at {OLLAMA_API_URL}."
        except requests.exceptions.Timeout:
            print(f"Timeout error (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"Retrying with a longer timeout...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return "Error: Request to Ollama timed out. The model might be too large or the server is overloaded."
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return f"Error: {str(e)}"

def is_code_file(file_path):
    """Check if a file is a code file based on its extension"""
    return any(file_path.endswith(ext) for ext in CODE_EXTENSIONS)

def should_ignore(path):
    """Check if a path should be ignored"""
    path_str = str(path)
    return any(pattern in path_str for pattern in IGNORE_PATTERNS)

def find_code_files(directory):
    """Find all code files in a directory recursively"""
    print(f"Searching for code files in: {directory}")
    code_files = []
    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]
        
        for file in files:
            file_path = os.path.join(root, file)
            if is_code_file(file_path) and not should_ignore(file_path):
                code_files.append(file_path)
    
    print(f"Found {len(code_files)} code files")
    return code_files

def load_and_process_code_files(directory):
    """Load and process code files from a directory"""
    global code_files
    code_files = find_code_files(directory)
    
    documents = []
    for file_path in code_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                relative_path = os.path.relpath(file_path, directory)
                documents.append(Document(
                    page_content=content,
                    metadata={"source": relative_path}
                ))
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if len(documents) == 0:
        print("No documents were loaded. Check file permissions and encoding.")
    
    return documents

def create_vector_store(documents):
    """Create a vector store from documents"""
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split documents into {len(chunks)} chunks")
    
    # Create embeddings with a better model for code understanding
    # Options include:
    # - "BAAI/bge-small-en-v1.5" (better quality, still fast)
    # - "Xenova/code-llama-instruct-7b" (specialized for code)
    # - "microsoft/codebert-base" (specialized for code)
    # - "nomic-ai/nomic-embed-text-v1" (high quality general embeddings)
    try:
        print("Creating embeddings with improved model...")
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        print(f"Error loading preferred embedding model: {str(e)}")
        print("Falling back to default embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    print("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store created with {len(chunks)} chunks")
    
    return vector_store

def analyze_codebase_for_optimal_params(documents):
    """Analyze codebase to determine optimal vector store parameters"""
    print("Analyzing codebase for optimal parameters...")
    
    # Calculate statistics about the codebase
    total_files = len(documents)
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_file_size = total_chars / total_files if total_files > 0 else 0
    max_file_size = max(len(doc.page_content) for doc in documents) if documents else 0
    min_file_size = min(len(doc.page_content) for doc in documents) if documents else 0
    
    # Count code lines and comments
    total_lines = 0
    comment_lines = 0
    code_lines = 0
    for doc in documents:
        lines = doc.page_content.split('\n')
        total_lines += len(lines)
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line.startswith('//') or line.startswith('/*') or line.startswith('*'):
                comment_lines += 1
            elif line and not line.startswith('"""') and not line.startswith("'''"):
                code_lines += 1
    
    # Count file types
    file_extensions = {}
    for file_path in code_files:
        ext = os.path.splitext(file_path)[1]
        if ext in file_extensions:
            file_extensions[ext] += 1
        else:
            file_extensions[ext] = 1
    
    # Determine if it's mostly code or documentation
    doc_ratio = comment_lines / total_lines if total_lines > 0 else 0
    is_doc_heavy = doc_ratio > 0.3  # If more than 30% is comments/docs
    
    # Determine language complexity based on file extensions
    complex_langs = ['.py', '.java', '.cpp', '.js', '.ts', '.go', '.rs']
    simple_langs = ['.html', '.css', '.md', '.txt', '.json', '.yaml', '.yml']
    
    complex_files = sum(file_extensions.get(ext, 0) for ext in complex_langs)
    simple_files = sum(file_extensions.get(ext, 0) for ext in simple_langs)
    
    is_complex_codebase = complex_files > simple_files
    
    # Determine optimal parameters based on analysis
    if is_doc_heavy:
        # Documentation-heavy codebases benefit from larger chunks
        optimal_chunk_size = min(3500, max(1500, int(avg_file_size / 3)))
        optimal_chunk_overlap = min(400, max(150, int(optimal_chunk_size / 5)))
    elif is_complex_codebase:
        # Complex code benefits from smaller chunks for precision
        optimal_chunk_size = min(2000, max(1000, int(avg_file_size / 5)))
        optimal_chunk_overlap = min(300, max(100, int(optimal_chunk_size / 4)))
    else:
        # Balanced approach for mixed codebases
        optimal_chunk_size = min(2500, max(1200, int(avg_file_size / 4)))
        optimal_chunk_overlap = min(350, max(120, int(optimal_chunk_size / 5)))
    
    # Determine optimal top_k based on codebase size
    if total_files < 10:
        optimal_top_k = 3
    elif total_files < 50:
        optimal_top_k = 5
    elif total_files < 200:
        optimal_top_k = 8
    else:
        optimal_top_k = 10
    
    # Ensure parameters are within valid ranges
    optimal_chunk_size = max(500, min(5000, optimal_chunk_size))
    optimal_chunk_overlap = max(0, min(500, optimal_chunk_overlap))
    optimal_top_k = max(1, min(20, optimal_top_k))
    
    print(f"Analysis complete. Optimal parameters determined:")
    print(f"- Chunk Size: {optimal_chunk_size}")
    print(f"- Chunk Overlap: {optimal_chunk_overlap}")
    print(f"- Top K Results: {optimal_top_k}")
    
    # Return analysis results and optimal parameters
    return {
        "chunk_size": optimal_chunk_size,
        "chunk_overlap": optimal_chunk_overlap,
        "top_k": optimal_top_k,
        "analysis": {
            "total_files": total_files,
            "total_chars": total_chars,
            "avg_file_size": avg_file_size,
            "max_file_size": max_file_size,
            "min_file_size": min_file_size,
            "total_lines": total_lines,
            "comment_lines": comment_lines,
            "code_lines": code_lines,
            "doc_ratio": doc_ratio,
            "is_doc_heavy": is_doc_heavy,
            "is_complex_codebase": is_complex_codebase,
            "file_extensions": file_extensions
        }
    }

def process_repository(repo_directory, embedding_model_name="BAAI/bge-small-en-v1.5", chunk_size=2000, chunk_overlap=200, top_k=5, auto_tune=False, progress=None):
    """Process a repository and create a vector store with configurable parameters"""
    global vector_store, repo_path, top_k_results
    repo_path = repo_directory
    top_k_results = top_k
    
    print(f"Processing repository: {repo_directory}")
    print(f"Using embedding model: {embedding_model_name}")
    
    # Load and process code files
    documents = load_and_process_code_files(repo_directory)
    
    print(f"Loaded {len(documents)} documents, creating vector store...")
    
    # If auto-tune is enabled, analyze the codebase to determine optimal parameters
    if auto_tune and documents:
        print("Auto-tuning enabled, analyzing codebase...")
        optimal_params = analyze_codebase_for_optimal_params(documents)
        chunk_size = optimal_params["chunk_size"]
        chunk_overlap = optimal_params["chunk_overlap"]
        top_k = optimal_params["top_k"]
        top_k_results = top_k
        
        print(f"Using auto-tuned parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={top_k}")
    else:
        print(f"Using manual parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={top_k}")
    
    # Create vector store
    if len(documents) > 0:
        # Split the documents into chunks with user-defined parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"Split documents into {len(chunks)} chunks using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        # Create embeddings with the selected model
        try:
            print(f"Creating embeddings with model: {embedding_model_name}")
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        except Exception as e:
            print(f"Error loading embedding model {embedding_model_name}: {str(e)}")
            print("Falling back to default embedding model...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store
        print("Building FAISS index...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"Vector store created with {len(chunks)} chunks")
        
        # Return detailed information about the indexing process
        return f"Repository processed successfully:\n" \
               f"- {len(documents)} files indexed\n" \
               f"- {len(chunks)} chunks created\n" \
               f"- Using {embedding_model_name} embedding model\n" \
               f"- Chunk size: {chunk_size} characters\n" \
               f"- Chunk overlap: {chunk_overlap} characters\n" \
               f"- Will retrieve top {top_k} results per query"
    else:
        print("No documents found to create vector store")
        return "No code files found in the repository. Please check the path and try again."


def get_relevant_code_snippets(query):
    """Get relevant code snippets for a query using the configured top_k parameter"""
    global vector_store, top_k_results
    if vector_store is None:
        return []
    
    # Use the globally configured top_k_results parameter
    k = top_k_results if top_k_results is not None else 5
    print(f"Retrieving top {k} code snippets for query: {query}")
    
    # Search for relevant documents
    docs = vector_store.similarity_search(query, k=k)
    
    # Format the results
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content
        results.append({
            "source": source,
            "content": content
        })
    
    print(f"Retrieved {len(results)} code snippets")
    return results

def format_code_snippets(snippets):
    """Format code snippets for display"""
    if not snippets:
        return "No relevant code snippets found."
    
    formatted_text = ""
    for i, snippet in enumerate(snippets):
        formatted_text += f"### File: {snippet['source']}\n\n```\n{snippet['content']}\n```\n\n"
    
    return formatted_text

def answer_code_question(question, model, temperature, max_tokens):
    """Answer a code question using RAG"""
    global vector_store, repo_path
    
    if vector_store is None:
        return "Please load a repository first."
    
    # Get relevant code snippets
    snippets = get_relevant_code_snippets(question)
    formatted_snippets = format_code_snippets(snippets)
    
    # Create prompt
    prompt = f"""
    You are a helpful coding assistant. Use the following code snippets to answer the question.
    
    CODE SNIPPETS:
    {formatted_snippets}
    
    QUESTION: {question}
    
    Please provide a detailed and accurate answer based on the code snippets. If the information in the snippets is not sufficient to answer the question, say so.
    """
    
    # Generate response
    response = generate_response(prompt, model, temperature, max_tokens)
    
    return response, formatted_snippets

def list_repositories(directory):
    """List potential repositories in a directory"""
    repos = []
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it has a .git directory or other common repo indicators
                if os.path.exists(os.path.join(item_path, '.git')) or \
                   os.path.exists(os.path.join(item_path, 'package.json')) or \
                   os.path.exists(os.path.join(item_path, 'setup.py')) or \
                   os.path.exists(os.path.join(item_path, 'Cargo.toml')):
                    repos.append(item)
    except Exception as e:
        print(f"Error listing repositories: {str(e)}")
    
    return repos

def check_ollama_server():
    """Check if the Ollama server is running and return status message"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            return True, "Ollama server is running."
        else:
            return False, f"Ollama server returned unexpected status: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to Ollama server at {OLLAMA_API_URL}. Please make sure it's running."
    except Exception as e:
        return False, f"Error checking Ollama server: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Code RAG with Ollama") as demo:
    gr.Markdown("# Code RAG with Ollama")
    
    # Check Ollama server status
    server_status, status_message = check_ollama_server()
    if not server_status:
        gr.Markdown(f"⚠️ **{status_message}**\n\nTo start Ollama server, run: `ollama serve` in a terminal.")
    
    gr.Markdown("Load a local code repository and ask questions about it.")
    
    with gr.Tab("Repository Loader"):
        with gr.Row():
            with gr.Column(scale=3):
                # Simple path selection
                common_paths = gr.Dropdown(
                    label="Common Folders", 
                    choices=[
                        os.path.expanduser("~"),  # Home directory
                        os.path.expanduser("~/git"),
                        os.path.expanduser("~/workspace"),
                        os.path.expanduser("~/repository")
                    ]
                )
                
                # Repository path input
                repo_path_input = gr.Textbox(label="Repository Path", placeholder="Enter the full path to your repository")
                
                # Add embedding model selection
                embedding_model = gr.Dropdown(
                    label="Embedding Model",
                    choices=[
                        "BAAI/bge-small-en-v1.5",  # Better quality, still fast
                        "all-MiniLM-L6-v2",      # Default, fast but less accurate
                        "microsoft/codebert-base", # Specialized for code
                        "nomic-ai/nomic-embed-text-v1" # High quality general embeddings
                    ],
                    value="BAAI/bge-small-en-v1.5",
                    info="Select the embedding model to use for code retrieval"
                )
                
                # Add vector configuration parameters
                with gr.Accordion("Vector Configuration", open=False):
                    with gr.Row():
                        auto_tune = gr.Checkbox(
                            label="Auto-tune Parameters", 
                            value=False,
                            info="Automatically detect optimal parameters based on your codebase"
                        )
                        auto_tune_btn = gr.Button("Analyze & Tune", visible=False)
                    
                    with gr.Group(visible=True) as manual_params:
                        chunk_size = gr.Slider(
                            minimum=500, 
                            maximum=5000, 
                            value=2000, 
                            step=100, 
                            label="Chunk Size",
                            info="Size of text chunks in characters. Smaller chunks are more precise but may lose context."
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0, 
                            maximum=500, 
                            value=200, 
                            step=50, 
                            label="Chunk Overlap",
                            info="Overlap between chunks in characters. Higher overlap helps maintain context between chunks."
                        )
                        top_k_results = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Top K Results",
                            info="Number of most relevant code snippets to retrieve for each query."
                        )
                    
                    # Function to toggle auto-tune button visibility
                    def toggle_auto_tune(checked):
                        return gr.Button(visible=checked), gr.Group(visible=not checked)
                    
                    # Function to analyze codebase and update UI with optimal parameters
                    def analyze_codebase(repo_path):
                        if not repo_path or not os.path.exists(repo_path):
                            return "Invalid repository path. Please enter a valid path.", None, None, None
                        
                        print(f"Analyzing codebase at: {repo_path}")
                        try:
                            # Load documents without creating vector store
                            documents = load_and_process_code_files(repo_path)
                            
                            if not documents:
                                return "No documents found in the repository. Please check the path.", None, None, None
                            
                            # Analyze codebase to determine optimal parameters
                            optimal_params = analyze_codebase_for_optimal_params(documents)
                            
                            # Format analysis results
                            analysis = optimal_params["analysis"]
                            result = f"## Codebase Analysis Results\n\n"
                            result += f"**Repository:** {os.path.basename(repo_path)}\n\n"
                            result += f"**Files Analyzed:** {analysis['total_files']}\n\n"
                            result += f"**Total Lines:** {analysis['total_lines']}\n\n"
                            result += f"**Code/Comment Ratio:** {(1-analysis['doc_ratio']):.2f} / {analysis['doc_ratio']:.2f}\n\n"
                            
                            result += f"**Optimal Parameters Detected:**\n\n"
                            result += f"- Chunk Size: **{optimal_params['chunk_size']}**\n\n"
                            result += f"- Chunk Overlap: **{optimal_params['chunk_overlap']}**\n\n"
                            result += f"- Top K Results: **{optimal_params['top_k']}**\n\n"
                            
                            result += f"**Codebase Characteristics:**\n\n"
                            result += f"- {'Documentation-heavy' if analysis['is_doc_heavy'] else 'Code-heavy'} codebase\n\n"
                            result += f"- {'Complex' if analysis['is_complex_codebase'] else 'Simple'} language mix\n\n"
                            
                            result += f"**File Types:**\n\n"
                            for ext, count in sorted(analysis['file_extensions'].items(), key=lambda x: x[1], reverse=True)[:10]:
                                result += f"- {ext}: {count}\n\n"
                            
                            # Return the analysis result and optimal parameters
                            return result, optimal_params['chunk_size'], optimal_params['chunk_overlap'], optimal_params['top_k']
                            
                        except Exception as e:
                            print(f"Error analyzing codebase: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            return f"Error analyzing codebase: {str(e)}", None, None, None
                    
                    auto_tune.change(toggle_auto_tune, auto_tune, [auto_tune_btn, manual_params])
                    # We need to move this after repo_info is defined
                    # Will connect it later
                
                def update_repo_path(path):
                    # Simply return the selected path
                    return path
                
                common_paths.change(update_repo_path, common_paths, repo_path_input)
                
                with gr.Row():
                    load_repo_btn = gr.Button("Load Repository", variant="primary")
                
                # Simple function to update repository path when selecting from dropdown
                def update_repo_path(selected_path):
                    if selected_path:
                        return selected_path
                    return ""
                
                # Connect the dropdown to update the repository path
                common_paths.change(update_repo_path, common_paths, repo_path_input)
            
            with gr.Column(scale=2):
                repo_info = gr.Markdown("No repository loaded.")
                
                # Now connect the auto-tune button after repo_info is defined
                auto_tune_btn.click(
                    fn=analyze_codebase,
                    inputs=repo_path_input,
                    outputs=[repo_info, chunk_size, chunk_overlap, top_k_results]
                )
    
    with gr.Tab("Code Q&A"):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, type="messages")
                question_input = gr.Textbox(label="Question", placeholder="Ask a question about the code...")
                clear_btn = gr.Button("Clear Conversation")
            
            with gr.Column(scale=2):
                models = get_available_models()
                model_dropdown = gr.Dropdown(
                    choices=models, 
                    value=models[0] if models else None, 
                    label="Model"
                )
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                max_tokens = gr.Slider(minimum=64, maximum=4096, value=2048, step=64, label="Max Tokens")
                
                # Add code snippets display
                code_snippets = gr.Markdown(label="Relevant Code Snippets")
    
    # Define functions for the interface
    def load_repository(repo_path, embedding_model_name, chunk_size, chunk_overlap, top_k, auto_tune=False):
        print(f"Load repository function called with path: {repo_path}")
        print(f"Using embedding model: {embedding_model_name}")
        print(f"Auto-tune: {auto_tune}")
        if not auto_tune:
            print(f"Vector configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={top_k}")
        
        if not repo_path or not os.path.exists(repo_path):
            print(f"Invalid repository path: {repo_path}")
            return "Invalid repository path. Please enter a valid path."
        
        try:
            print(f"Repository path exists, processing repository...")
            # Process repository with all the configuration parameters
            result = process_repository(
                repo_path, 
                embedding_model_name, 
                chunk_size, 
                chunk_overlap, 
                top_k,
                auto_tune
            )
            print(f"Process repository returned: {result}")
            
            # The result from process_repository now contains all the information we need
            # Format it as markdown for display
            info = f"## Repository: {os.path.basename(repo_path)}\n\n"
            info += f"**Path:** {repo_path}\n\n"
            
            if auto_tune:
                info += "**Auto-tuning:** Enabled - Parameters were automatically optimized for this codebase\n\n"
            
            info += result.replace("\n", "\n\n")
            
            print(f"Returning repository info: {info[:100]}...")
            return info
        except Exception as e:
            print(f"Exception in load_repository: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error loading repository: {str(e)}"

    
    def ask_question(question, history, model, temperature, max_tokens):
        if not question.strip():
            return None, history
        
        # Add user message to history
        history.append({"role": "user", "content": question})
        
        # Get answer and code snippets
        answer, snippets = answer_code_question(question, model, temperature, max_tokens)
        
        # Add assistant message to history
        history.append({"role": "assistant", "content": answer})
        
        return "", history, snippets
    
    # Connect UI elements to functions
    load_repo_btn.click(
        fn=load_repository,
        inputs=[repo_path_input, embedding_model, chunk_size, chunk_overlap, top_k_results, auto_tune],
        outputs=repo_info,
        api_name="load_repository"
    )
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot, model_dropdown, temperature, max_tokens],
        outputs=[question_input, chatbot, code_snippets],
        api_name="ask_question"
    )
    clear_btn.click(
        fn=lambda: ([], ""),
        inputs=None,
        outputs=[chatbot, code_snippets],
        api_name="clear_conversation"
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
