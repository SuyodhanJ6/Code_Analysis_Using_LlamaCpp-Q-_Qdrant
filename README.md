# Code_Analysis_Using_LlamaCpp(Q)

Advanced code analysis tool using CodeLlama-13b-instruct.Q4_K_M.gguf model via LlamaCpp and Qdrant vector database.

## Project Architecture

Below is a flowchart representing the architecture of this RAG application:

![image](https://raw.githubusercontent.com/SuyodhanJ6/Code_Analysis_Using_LlamaCpp-Q-_Qdrant/main/flowchart/Screenshot%20from%202024-09-25%2000-14-30.png)

## Setup Guide

Follow these steps to set up and run the project:

### Prerequisites

- Anaconda or Miniconda
- Git
- C++ compiler (g++ or clang)

### Installation

1. **Install system dependencies:**

   ```bash
   sudo apt-get update
   sudo apt-get install build-essential g++ clang
   ```

2. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/Code_Analysis_Using_LlamaCpp-Q.git
   cd Code_Analysis_Using_LlamaCpp-Q
   ```

3. **Create and activate Conda environment:**

   ```bash
   conda create -p ./venv python=3.10
   conda activate ./venv
   ```

4. **Install Python dependencies:**

   ```bash
   pip install openai tiktoken langchain langchain-community GitPython gpt4all llama-cpp-python==0.2.77 langchain-qdrant
   ```

5. **Download the CodeLlama model:**

   ```bash
   wget https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf
   ```

### Project Setup

Create a Python script named `setup_project.py` with the following content:

```python
import os
from git import Repo, GitCommandError
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryMemory
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Clone or update repository
repo_url = "https://github.com/SuyodhanJ6/LLM-workshop-2024"
repo_path = "./suyo_repo"

try:
    if os.path.exists(repo_path):
        print(f"Updating repository in {repo_path}...")
        repo = Repo(repo_path)
        origin = repo.remotes.origin
        origin.pull()
    else:
        print(f"Cloning repository to {repo_path}...")
        repo = Repo.clone_from(repo_url, repo_path)
    print("Repository cloned/updated successfully.")

    # Load and process documents
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    print(f"Number of documents loaded: {len(documents)}")

    # Split documents
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = python_splitter.split_documents(documents)

    # Set up embeddings
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embeddings = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)

    # Set up Qdrant vector store
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="autogen_collection1",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="autogen_collection1",
        embedding=embeddings,
    )
    vector_store.add_documents(texts)

    # Set up retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3},
    )

    # Set up LLM
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="codellama-13b-instruct.Q4_K_M.gguf",
        n_ctx=5000,
        n_gpu_layers=1,
        n_batch=512,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=True,
    )

    # Set up ConversationalRetrievalChain
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    print("Setup complete. You can now use the 'qa' object for code analysis.")

except GitCommandError as e:
    print(f"Git error occurred: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
```

### Running the Project

1. **Run the setup script:**

   ```bash
   python setup_project.py
   ```

2. **Using the Code Analyzer:**

   After running the setup script, you can use the `qa` object in Python to analyze code. For example:

   ```python
   question = "How to prepare a dataset for supervised instruction finetuning?"
   response = qa.invoke({"question": question})
   print(response['answer'])
   ```

## Troubleshooting

- If you encounter issues with LlamaCpp, ensure you have the correct version (0.2.77) installed.
- Make sure the CodeLlama model file is in the correct location and has the right permissions.

