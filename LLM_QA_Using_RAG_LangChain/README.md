# LLM QA Using RAG and LangChain

This project provides a chatbot assistant capable of answering user queries based on uploaded PDF or text files. It leverages Retrieval-Augmented Generation (RAG) and LangChain for document retrieval and question answering, tailored to specific tasks depending on user requirements.

## Features

- **File-Based Question Answering**: The chatbot allows users to upload PDF or text files and generates responses based on the content.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with a language model to improve the relevance and accuracy of responses.
- **LangChain Integration**: Uses LangChain for document splitting, vector storage, and query processing, making it efficient and modular.
- **Conversational Memory**: The chatbot retains conversation history, allowing for context-aware responses to follow-up questions.
- **Customizable Language Models**: Integrates the `lmsys/vicuna-7b-v1.5` model by default, with flexibility to switch to other models.

## Workflow

1. **File Upload**: Users upload a PDF or text document containing the information they wish to query.
2. **Document Processing**: The system splits the document into smaller chunks, processes them, and stores them in a vector database.
3. **Query Processing**: Users input queries, and the system retrieves relevant document parts to generate a coherent answer using the language model.
4. **Follow-up Questions**: The chatbot remembers previous interactions and answers follow-up questions with context.

## Use Cases

- Extracting information from reports, research papers, or technical documents.
- Domain-specific assistance based on uploaded datasets.
- Conversational Q&A for custom datasets provided by users.

---

This README provides a concise overview of the chatbot's capabilities and workflow while focusing on its core functionalities and use cases.
