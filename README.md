# PDF Chat API with Flask, Pinecone, and OpenAI

This project provides a REST API that allows users to upload PDF documents and interact with them using natural language questions. The API leverages machine learning models to understand and answer questions based on the content of the uploaded PDF files. It uses **Flask** as the web framework, **Pinecone** for vector storage, and **OpenAI**'s language model for question answering. The application simulates a teacher-like response style, providing interactive and structured feedback to users.

## Features

- **PDF Upload**: Upload a PDF document to the server.
- **Question Answering**: Ask questions related to the uploaded PDF, and receive responses as if from a knowledgeable teacher.
- **Interactive Feedback**: The system is designed to provide structured, conversational, and supportive feedback, making it ideal for educational purposes.

## Project Structure

- `app.py`: The main application file containing the Flask API and endpoints for PDF upload and question answering.
- `requirements.txt`: Lists all the dependencies needed for the project.
- `.gitignore`: Specifies files and folders to be ignored by Git.
- `uploads/`: Directory where uploaded PDF files are stored (not tracked by Git).

## Endpoints

### 1. `/v1/pdf` (POST)

- **Description**: Endpoint for uploading PDF documents.
- **Request**: Multipart form-data containing the PDF file.
- **Response**:
  - `pdf_id`: Unique ID assigned to the uploaded PDF.
  - `index_name`: Name of the Pinecone index created for the PDF.

#### Example cURL Request

```bash
curl -X POST -F "file=@path/to/yourfile.pdf" localhost/v1/pdf
```

### 2. `/v1/chat/<pdf_id>` (POST)

- **Description**: Endpoint for asking questions about the uploaded PDF.
- **Request**: JSON payload with a message field containing the question.
- **Response**:
  - `pdf_id`: ID of the PDF being queried.
  - `query`: Original question.
  - `answer`: Response from the AI, structured like a teacher's answer.
  - `status`: Indicates success or error.

#### Example cURL Request

```bash
curl -X POST -H "Content-Type: application/json" -d '{"message": "What is supervised learning?"}' localhost/v1/chat/1
```

## Usage 

- **Upload a PDF**: using the `/v1/pdf` endpoint.
- **Ask Question**: related to the PDF content through the `/v1/chat/<pdf_id>` endpoint.
The system will respond with answers in a teacher-like, structured manner, providing interactive feedback.




