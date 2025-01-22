# PDF Text Extractor

PDF Text Extractor is a web application that allows users to upload PDF files, extract text, and interact with the content using a chat interface powered by OpenAI's GPT-3.5-turbo. The application also supports document indexing and querying using a Retrieval-Augmented Generation (RAG) approach.

## Features

- **PDF Upload and Text Extraction**: Upload PDF files and extract text, tables, and images.
- **Chat Assistant**: Interact with the extracted content using a chat interface.
- **Document Indexing**: Index documents for efficient querying.
- **History Management**: View and manage the history of processed PDFs.

## Prerequisites

- Python 3.9 or later
- Node.js and npm (for frontend dependencies, if applicable)
- Docker (optional, for containerized deployment)

## Installation

### Running Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   flask run
   ```

5. **Access the application**:
   Open your web browser and go to `http://localhost:5000`.

### Running with Docker

1. **Build the Docker image**:
   ```bash
   docker build -t pdf-text-extractor .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 5000:5000 pdf-text-extractor
   ```

3. **Access the application**:
   Open your web browser and go to `http://localhost:5000`.

## Configuration

- **API Keys**: The application requires an OpenAI API key for the chat assistant feature. You can set this key in the application interface or store it in a `.env` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.