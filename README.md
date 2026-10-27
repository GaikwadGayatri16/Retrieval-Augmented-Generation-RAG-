
Practical No. 06 --- Retrieval-Augmented Generation (RAG) with a Pre-trained LLM
A document question-answering application that retrieves relevant
passages from a domain-specific collection and uses a pre-trained
language model to generate context-grounded answers. The notebook also
compares RAG responses with responses from the LLM without retrieval.

Student Details
Student: Gayatri Pandharinath Gaikwad
PRN: 202502110002
Practical: No. 06 --- Retrieval-Augmented Generation (RAG) using
a Pre-trained LLM
1. Objective
Build a Retrieval-Augmented Generation (RAG) application for answering
questions from a domain-specific document collection. Implement document
retrieval and answer generation using a pre-trained Large Language Model
(LLM), then compare the generated responses with those produced by the
LLM without retrieval.

2. Project Overview
The project uses a fictional company handbook covering leave, remote
work and information security, and expense and travel policies. These
documents contain specific policy values that can be used to test
whether retrieval helps the model answer accurately.

Pipeline
Domain documents (.txt / .md / .pdf)
              |
              v
       Load and clean text
              |
              v
    Chunk text with overlap
              |
              v
 Generate sentence embeddings
              |
              v
     Store chunks in ChromaDB
              |
       User asks a question
              |
              v
     Embed the user question
              |
              v
 Retrieve top-k relevant chunks
              |
              v
 Add retrieved context to prompt
              |
              v
      Pre-trained LLM answer
              |
              v
 Compare RAG with LLM-only answer
3. Technologies Used
Python --- application and evaluation logic.
Google Colab --- notebook-based execution environment.
Sentence Transformers --- generates text embeddings.
sentence-transformers/all-MiniLM-L6-v2 --- embedding model.
ChromaDB --- persistent vector database and similarity search.
google/flan-t5-large --- default pre-trained text-generation
model.
Transformers --- loads and runs the Hugging Face model.
pypdf --- extracts text from PDF files.
Pandas --- stores and summarizes evaluation results.
Matplotlib --- visualizes the RAG vs. LLM-only accuracy
comparison.
The code also includes an optional OpenAI-compatible API configuration,
but the notebook's selected default is the local Hugging Face model.

4. Project Structure
The notebook creates the project under /content/rag_project in Google
Colab.

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
├── chroma_db/                 # created when documents are indexed
└── results/
    ├── eval_results.csv       # created by evaluation
    └── comparison.md          # created by evaluation
The submitted notebook contains the code that writes these project files
into the Colab runtime. If publishing the project as a standalone GitHub
repository, include the source files and data files as well as the
notebook, rather than assuming that files created inside a Colab session
are automatically stored in GitHub.

5. How It Works
5.1 Document loading and cleaning
The ingestion module supports .txt, .md, and .pdf documents. It
extracts the text and normalizes whitespace, joins hyphenated line
breaks, and removes lines that appear to contain page numbers.

5.2 Chunking
Long documents are divided into smaller text chunks. The default
configuration uses:

Chunk size: 500 characters
Chunk overlap: 80 characters
The overlap helps preserve context when information falls near a chunk
boundary.

5.3 Embedding generation
The embedding model sentence-transformers/all-MiniLM-L6-v2 converts
each chunk into a numeric vector. The same model encodes a user's
question so the question can be compared with document chunks in the
same embedding space.

5.4 Vector database and retrieval
ChromaDB stores the chunk text, embeddings, source filename, and chunk
ID. At question time, the application searches for the top 4 most
similar chunks by default. The retrieved text and source metadata are
returned for inspection.

5.5 Answer generation with retrieval
The RAG prompt tells the model to answer using only the retrieved
context and to state that it does not know based on the documents if the
answer is not present. This design aims to reduce unsupported answers.

5.6 LLM-only baseline
The baseline sends the question to the same configured LLM without
adding retrieved document context. Comparing both methods helps
illustrate how retrieval can improve answers about information specific
to the supplied documents.

6. Running the Notebook
Option A --- Google Colab (recommended for this practical)
Upload RAG_Practical_06.ipynb to Google Colab.
Select a Python runtime. A GPU runtime may reduce model-loading or
generation time; CPU can also work if sufficient memory is
available.
Run the notebook cells from top to bottom.
Wait for the dependencies and models to load.
Confirm that document indexing completes and the ChromaDB collection
is created.
Run the sample questions and the custom-question cell.
Run the evaluation cell and inspect the generated summary table and
chart.
Save/download the notebook with its outputs.
Option B --- Run the generated project scripts
After the notebook has created the project files, the core commands from
the project directory are:

pip install sentence-transformers chromadb pypdf tabulate openai transformers torch pandas matplotlib
python ingest.py
python evaluate.py
The default configuration expects the pre-trained model
google/flan-t5-large to be available through Hugging Face
Transformers. The first run downloads model files and therefore requires
internet access. The model can require substantial memory; use a smaller
model such as google/flan-t5-base if your environment cannot load the
default model, and rerun the evaluation if you change models.

To run rag.py interactively, create a Python script or notebook that
imports RAGPipeline, initializes it after indexing, and calls
answer_rag(question) and answer_plain(question).

7. Example Questions
The notebook tests questions such as:

How many unused annual leave days can be carried forward?
What is the nightly hotel limit in metro cities?
What is the name of the corporate VPN employees must use outside the
office?
Within how many days must expense claims be submitted?
What is the maximum client entertainment spend per person per event?
It also includes an unrelated question such as "What is AI?" to
demonstrate the RAG prompt's instruction to avoid answering beyond the
supplied documents.

8. Evaluation Method
The notebook evaluates 12 questions using a JSON file containing
each question, its expected reference, expected keywords, and expected
source document.

The evaluation script calculates:

Answer accuracy: whether the expected keyword(s) or figure(s)
appear in the generated answer.
Retrieval hit rate: whether the expected source document appears
among the retrieved top-k chunks.
Grounding score: the proportion of answer tokens also found in
the retrieved context. This is a simple lexical proxy, not a
semantic or human evaluation.
Hallucination rate: the proportion of answers marked incorrect
that do not contain one of the script's uncertainty phrases.
These are lightweight, rule-based metrics. They are useful for a
practical demonstration but do not replace manual review or a more
comprehensive evaluation.

9. Recorded Results
The saved notebook reports the following results for its 12-question
test set:

Metric LLM-only (no retrieval) RAG

Answer accuracy 8% 100%
Hallucination rate 92% 0%
Retrieval hit rate (top-k) N/A 100%
Average grounding in context N/A 0.92

The notebook's accuracy check is keyword/figure based. Therefore, the
values above describe this particular test set and evaluation method;
they should not be interpreted as proof that RAG will always be 100%
accurate. The results may change with different documents, questions,
models, chunking settings, or runtime versions.

10. Sample Observations
The RAG answer to the annual-leave carry-forward question uses
information retrieved from the leave policy.
The hotel-limit question retrieves the expense and travel policy.
For an unrelated question, the RAG prompt encourages the model to
say that the answer is not available in the supplied documents.
In this test set, the LLM-only baseline often lacks the specific
fictional policy details, while the RAG pipeline can retrieve them
from the document collection.
11. Configuration
The main settings are in config.py:

Setting Default Purpose

RAG_CHUNK_SIZE 500 Maximum target chunk
length in characters

RAG_CHUNK_OVERLAP 80 Overlap between chunks

RAG_EMBED_MODEL sentence-transformers/all-MiniLM-L6-v2 Embedding model

RAG_TOP_K 4 Number of chunks
retrieved

LLM_PROVIDER hf Uses Hugging Face by
default

HF_MODEL google/flan-t5-large Default generation
model

RAG_MAX_NEW_TOKENS 200 Generation length limit
12. Limitations and Future Improvements
The example corpus contains fictional company policies and is
intended for demonstration.
Keyword-based accuracy may not fully capture whether an answer is
semantically correct.
The grounding score is based on token overlap and is only a rough
indicator.
Retrieval can return weakly related chunks; similarity thresholds
and better retrieval strategies could help.
The project could be improved with source citations in final
answers, a user interface, larger and more varied evaluation data,
semantic evaluation, and manual review by a human evaluator.
For real company documents, access control, privacy, and protection
of confidential information should be considered before indexing.
13. Conclusion
This practical implements the core RAG workflow: document loading and
cleaning, chunking, embedding generation, vector indexing,
similarity-based retrieval, context-aware answer generation, and
comparison with an LLM-only baseline. In the notebook's recorded
evaluation, RAG outperformed the no-retrieval baseline on the selected
company-policy questions.

GitHub Repository
Repository URL: [Add your GitHub repository URL here]

Before sharing, ensure the repository contains the notebook and any
project files you intend others to run. Do not commit API keys, access
tokens, private documents, or other secrets.

