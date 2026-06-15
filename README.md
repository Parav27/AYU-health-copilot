# AYU - AI Powered Health Intelligence Platform

AYU is an AI-powered health report analysis platform that helps users understand complex medical reports in simple language.

Users can upload a medical report PDF, extract biomarker values, identify abnormal findings, receive AI-generated health summaries, and ask report-aware health questions through a conversational interface.

---

## Problem Statement

Medical reports are often difficult for non-medical users to understand.

Most people receive lab reports containing dozens of biomarkers, reference ranges, abbreviations, and medical terminology without clear explanations.

AYU bridges this gap by converting complex medical reports into understandable health insights and enabling users to ask questions about their reports in natural language.

---

## Features

### Medical Report Analysis

- Upload PDF medical reports
- Automatic text extraction from reports
- Biomarker extraction and categorization
- Detection of abnormal values
- Plain-English health summaries
- Educational health notes

### AI-Powered Health Assistant

- Report-aware conversational chat
- Answers questions using uploaded report data
- Educational explanations of biomarkers
- Context-aware responses

### Retrieval-Augmented Generation (RAG)

- ChromaDB vector database
- Semantic search over curated medical knowledge
- Grounded responses using retrieved context
- Reduced hallucinations compared to standard chatbot approaches

### Safety Features

- Educational information only
- No diagnosis
- No prescriptions
- No treatment recommendations
- Encourages consultation with healthcare professionals

---

## Tech Stack

### Frontend

- Next.js
- TypeScript
- Tailwind CSS

### Backend

- FastAPI
- Pydantic
- PyMuPDF

### AI & RAG

- Groq LLM
- ChromaDB
- LangChain
- Sentence Transformers
- all-MiniLM-L6-v2 embeddings

---

## Architecture

```text
User
 │
 ▼
Next.js Frontend
 │
 ▼
FastAPI Backend
 │
 ├── PDF Extraction (PyMuPDF)
 │
 ├── Groq Report Analysis
 │
 ├── Biomarker Extraction
 │
 ├── Report Context Memory
 │
 └── ChromaDB Retriever
          │
          ▼
      Medical Knowledge Base
          │
          ▼
      Groq Answer Generation
```

---

## How It Works

### Report Analysis Pipeline

```text
PDF Upload
    ↓
Text Extraction
    ↓
Groq Analysis
    ↓
Structured Biomarker Extraction
    ↓
Health Summary Generation
    ↓
Frontend Visualization
```

### Chat Pipeline

```text
User Question
      ↓
Retrieve Relevant Knowledge
      ↓
Combine Report Context
      ↓
Groq Generation
      ↓
Context-Aware Answer
```

---

## Example Questions

After uploading a report, users can ask:

- What is my HbA1c value?
- Which biomarkers are abnormal?
- Why is my LDL cholesterol high?
- What does elevated triglyceride mean?
- Explain my blood report in simple language.

---

## Project Highlights

- Full-stack AI application
- Medical PDF processing
- Retrieval-Augmented Generation (RAG)
- Vector database integration
- Report-aware conversational AI
- Structured data extraction
- Production-style API architecture

---

## Challenges Solved

One major challenge was handling inconsistent LLM outputs.

The AI model occasionally returned values such as:

```json
{
  "report_type": "comprehensive_metabolic_panel",
  "status": "borderline_high"
}
```

These values failed schema validation and caused biomarker extraction to be discarded.

A normalization layer was implemented before validation to map AI outputs into valid schema values, preserving biomarker extraction while maintaining strict validation rules.

---

## Future Scope

- Multi-report comparison
- Historical trend analysis
- Health dashboard
- User authentication
- Cloud deployment
- Fitness tracker integration
- Personalized health insights

---

## Disclaimer

AYU is an educational platform and does not provide medical diagnosis, treatment, or professional healthcare advice.

Users should always consult qualified healthcare professionals regarding medical decisions.

---

## Author

Built by Bai as a full-stack AI healthcare project using FastAPI, Next.js, Groq, ChromaDB, and Retrieval-Augmented Generation.