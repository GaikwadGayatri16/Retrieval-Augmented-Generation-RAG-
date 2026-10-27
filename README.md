# Practical No. 06 — Retrieval-Augmented Generation (RAG) Using a Pre-trained LLM

A document question-answering application that uses Retrieval-Augmented Generation (RAG) to answer questions from domain-specific documents. This project also compares RAG-generated answers with responses from a pre-trained Large Language Model (LLM) without retrieval.

## Student Details

- **Student Name:** Gayatri Pandharinath Gaikwad
- **PRN:** 202502110002
- **Practical No.:** 06
- **Title:** Retrieval-Augmented Generation (RAG) Using a Pre-trained LLM

## 1. Objective

To build a Retrieval-Augmented Generation (RAG) application for answering questions from a domain-specific document collection. The application implements document retrieval and answer generation using a pre-trained LLM and compares the generated responses with those obtained using the LLM without retrieval.

## 2. Project Overview

The project demonstrates how a RAG system combines document retrieval with a language model to generate context-aware answers.

The application uses fictional company-policy documents covering:

- Leave policies
- Remote work and information security
- Expense and travel policies

When a user asks a question, the system retrieves relevant document chunks and supplies them to the LLM as context. The generated answer is then compared with the response produced by the same LLM without retrieved context.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Google Colab | Notebook execution |
| Sentence Transformers | Text embedding generation |
| `all-MiniLM-L6-v2` | Embedding model |
| ChromaDB | Vector storage and similarity search |
| Hugging Face Transformers | Loading and running the LLM |
| `google/flan-t5-large` | Pre-trained text-generation model |
| Pandas | Evaluation data processing |
| Matplotlib | Evaluation visualization |
| pypdf | PDF text extraction |

## 4. System Architecture

```text
Domain-Specific Documents
           |
           v
   Document Loading
           |
           v
   Text Preprocessing
           |
           v
    Text Chunking
           |
           v
  Embedding Generation
           |
           v
  ChromaDB Vector Store
           |
           v
      User Query
           |
           v
   Query Embedding
           |
           v
  Similarity-Based Search
           |
           v
   Retrieve Top-k Chunks
           |
           v
    Context + User Query
           |
           v
    Pre-trained LLM
           |
           v
      RAG Answer
           |
           v
 Compare with LLM-only Answer
```

## 5. Implementation Details

### 5.1 Document Processing

The application loads domain-specific documents and extracts their textual content. Supported document formats include TXT, Markdown, and PDF.

Text preprocessing includes whitespace normalization and cleaning unnecessary line breaks or page-number lines.

### 5.2 Text Chunking

The documents are divided into smaller, overlapping chunks to make retrieval more effective.

Default configuration:

- Chunk size: 500 characters
- Chunk overlap: 80 characters

Chunking allows the system to retrieve relevant sections rather than passing entire documents to the language model.

### 5.3 Embedding Generation

The `sentence-transformers/all-MiniLM-L6-v2` model converts text chunks into numerical vectors called embeddings.

These embeddings represent semantic information about the text and enable similarity-based document retrieval.

### 5.4 Vector Database

ChromaDB stores the document chunks, embeddings, and associated metadata.

When a user submits a question, the application searches the vector database and retrieves the four most relevant chunks by default.

### 5.5 Retrieval-Augmented Generation

The RAG pipeline combines the user's question with the retrieved document context and sends the combined prompt to the pre-trained LLM.

The prompt instructs the model to answer using the supplied context and acknowledge when the required information is not available in the documents.

### 5.6 LLM-only Response

For comparison, the application sends the same question to the language model without retrieving document context.

This baseline demonstrates the difference between general language-model responses and responses supported by a specific document collection.

## 6. Project Structure

The notebook generates the following project structure in the Google Colab runtime:

```text
rag_project/
├── config.py
├── ingest.py
├── llm.py
├── rag.py
├── evaluate.py
├── data/
│   ├── docs/
│   │   ├── leave_policy.txt
│   │   ├── remote_work_and_security.txt
│   │   └── expense_and_travel.txt
│   └── eval_questions.json
├── chroma_db/
└── results/
    ├── eval_results.csv
    └── comparison.md
```

The `chroma_db/` directory stores the indexed vector data, while the `results/` directory contains evaluation outputs.

**Note:** These files are generated in the Colab runtime. Include the notebook and the necessary source and data files in your GitHub repository if you want others to reproduce the project.

## 7. Installation and Execution

### Option A: Google Colab

1. Open Google Colab.
2. Upload the `RAG_Practical_06.ipynb` notebook.
3. Select a Python runtime.
4. Execute the notebook cells from top to bottom.
5. Wait for the dependencies and pre-trained model to load.
6. Verify that the documents are processed and indexed.
7. Test the application with sample questions.
8. Run the evaluation cells to generate the comparison results and chart.

The first model-loading run may require an internet connection and sufficient memory.

### Option B: Running the Generated Python Files

Install the required dependencies:

```bash
pip install sentence-transformers chromadb pypdf tabulate openai transformers torch pandas matplotlib
```

After the notebook has generated the project files, navigate to the project directory and run:

```bash
python ingest.py
python evaluate.py
```

The default model is `google/flan-t5-large`, which is downloaded through Hugging Face Transformers when needed.

**Important:** The standalone commands require the generated Python files to be saved locally. Running the notebook in Colab does not automatically create these files on your computer or upload them to GitHub.

## 8. Sample Questions

The application evaluates questions such as:

- How many unused annual leave days can be carried forward?
- What is the nightly hotel limit in metro cities?
- Which corporate VPN must employees use outside the office?
- Within how many days must expense claims be submitted?
- What is the maximum client entertainment spend per person per event?

An unrelated question, such as “What is AI?”, can also be used to test whether the RAG system avoids answering questions unsupported by the supplied documents.

## 9. Evaluation Methodology

The notebook evaluates 12 test questions using expected answers, keywords, and expected source documents.

The following metrics are calculated:

- **Answer Accuracy:** Checks whether the expected keyword or figure appears in the generated answer.
- **Retrieval Hit Rate:** Checks whether the expected source document appears in the retrieved results.
- **Grounding Score:** Estimates lexical overlap between the generated answer and retrieved context.
- **Hallucination Rate:** Estimates unsupported or incorrect responses using the evaluation script's rules.

These metrics provide a basic evaluation of the system. They do not replace manual assessment or a more comprehensive semantic evaluation.

## 10. Results and Comparison

The saved notebook reports the following results for the 12-question test set.

| Evaluation Metric | LLM-only | RAG |
|---|---:|---:|
| Answer Accuracy | 8% | 100% |
| Hallucination Rate | 92% | 0% |
| Retrieval Hit Rate | N/A | 100% |
| Average Grounding Score | N/A | 0.92 |

### Observations

1. The RAG pipeline retrieves relevant information from the supplied policy documents.
2. The LLM-only baseline does not have access to the supplied document context.
3. RAG performs better on the selected questions about fictional company policies.
4. The system is designed to acknowledge when information is unavailable in the retrieved context.
5. The evaluation chart illustrates the difference between the two approaches.

**Note:** These figures are the notebook's recorded results, not a guarantee of general performance. The accuracy calculation is keyword-based, and the grounding and hallucination metrics use simplified rules. Results may vary with different documents, questions, models, and evaluation methods.

## 11. Advantages of RAG

- Provides answers based on a specified document collection.
- Reduces dependence on the LLM's pretrained knowledge for domain-specific questions.
- Allows documents to be updated without retraining the language model.
- Makes relevant source passages available for inspection.
- Supports domain-specific question answering.
- Provides a useful baseline for comparing retrieval-assisted and retrieval-free generation.

## 12. Limitations

- Answer quality depends on document quality and retrieval relevance.
- Incorrect or incomplete source documents can lead to incorrect answers.
- Keyword-based evaluation may not detect every incorrect answer.
- Lexical grounding scores do not fully measure semantic correctness.
- The default language model may require considerable memory and processing time.
- The example uses fictional company policies and is intended for demonstration.

## 13. Future Enhancements

Potential improvements include:

- Adding a web interface using Streamlit or Flask.
- Displaying source-document citations alongside answers.
- Improving retrieval with reranking and similarity thresholds.
- Expanding the evaluation dataset.
- Introducing semantic evaluation and human assessment.
- Supporting additional document formats.
- Adding access controls for confidential documents.

## 14. Conclusion

This practical demonstrates the complete Retrieval-Augmented Generation workflow, including document processing, text chunking, embedding generation, vector database creation, similarity-based retrieval, context-aware answer generation, and comparison with an LLM-only baseline.

The notebook's recorded results show that RAG performs better than the no-retrieval baseline on the selected domain-specific questions. The project illustrates how retrieval can help language models provide answers grounded in a supplied document collection.

## 15. GitHub Repository

**Repository URL:** [PASTE YOUR GITHUB REPOSITORY LINK HERE]

The repository should contain the completed notebook and all required source and data files for reproducing the experiment.

---

**End of README**
