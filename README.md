# EvidencePilot — Research, Grounded in Evidence.

<p align="center">
  <strong>Better questions. Stronger evidence. More transparent research.</strong>
</p>

<p align="center">
  <a href="https://evidencepilot.streamlit.app/"><strong>🚀 Launch EvidencePilot</strong></a>
  &nbsp; • &nbsp;
  <a href="https://github.com/AravindSAsok/EvidencePilot">GitHub Repository</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Google Gemini"/>
  <img src="https://img.shields.io/badge/AI-Assisted_Research-111827?style=for-the-badge" alt="AI-assisted research"/>
</p>

---

## 🔍 Overview

**EvidencePilot is an AI-powered research assistant designed to help users explore PDF documents, discover relevant evidence, generate source-grounded answers, and evaluate the reliability of research outputs.**

Instead of treating AI-generated answers as unquestionable facts, EvidencePilot emphasizes evidence inspection, citation awareness, claim auditing, and human review.

Built with Python, Streamlit, and Google's Gemini models, the application brings document question-answering, research assistance, and retrieval evaluation together in one interactive workspace.

Whether you're exploring a technical paper, reviewing a report, or investigating a document's claims, EvidencePilot aims to make the research process more transparent and evidence-driven.

## ✨ Key Features

### 📄 1. PDF-Based Research

- Upload text-based PDF documents for analysis.
- Extract document text and organize it into searchable passages.
- Explore source material without manually searching every page.
- Ask questions grounded in the uploaded document.

### 🔎 2. Intelligent Evidence Retrieval

- **Keyword search:** Find passages using direct textual matches.
- **Semantic search:** Retrieve passages based on conceptual relevance.
- Identify potentially useful evidence before generating an answer.
- Inspect retrieved passages to understand the basis of a response.

### 💬 3. Evidence-Grounded Question Answering

- Ask natural-language questions about an uploaded document.
- Generate AI-assisted answers using retrieved evidence.
- Display page references where available.
- Inspect the passages used to support a response.

EvidencePilot is designed to make the connection between an answer and its underlying source material easier to examine.

### 🧪 4. Answer Audit

Evaluate an existing research answer against the retrieved evidence.

The audit workflow is designed to identify:

- Supported claims
- Partially supported claims
- Unsupported claims
- Unclear claims
- Missing or questionable page citations
- Potential conflicts between claims and evidence

The purpose is to help users identify statements that deserve closer inspection, not to treat an automated verdict as definitive proof.

### 🤖 5. Research Agent

A dedicated research workflow for investigating questions using available document evidence.

The Research Agent helps organize evidence and produce structured research outputs. Its results should be reviewed for factual accuracy, citation validity, unsupported claims, and gaps in the available material.

### 🛡️ 6. Research Audit

A separate quality-assurance workflow for examining research outputs and their relationship to supporting evidence.

It helps surface potential evidence gaps, questionable claims, and citation issues that may need further investigation.

### 👤 7. Human Review

A dedicated review workflow that supports human oversight of AI-assisted research.

Automated analysis can help prioritize what needs attention, but the final interpretation of important claims should remain subject to human judgment.

### 📊 8. Retrieval Evaluation Lab

Evaluate retrieval quality using manually identified supporting pages.

The evaluation workflow can compare keyword and semantic retrieval using metrics such as:

- **Hit@K:** Whether relevant evidence appears within the top K retrieved results.
- **Page Recall@K:** The proportion of known relevant pages retrieved within the top K results.
- **MRR@K:** How highly the first relevant result is ranked.

These metrics help assess retrieval performance on a defined evaluation set rather than relying on subjective impressions alone.

---

## 🧠 Design Philosophy

EvidencePilot is built around a simple principle:

> An AI-generated answer is only as useful as the evidence supporting it — and the user's ability to inspect that evidence.

The project emphasizes four complementary practices:

| Practice | Purpose |
|---|---|
| Retrieval | Find potentially relevant source passages. |
| Grounding | Generate answers using retrieved evidence. |
| Auditing | Examine claims, supporting passages, and citations. |
| Human oversight | Review uncertainty and make informed judgments. |

These practices improve transparency, but they do not eliminate hallucinations or guarantee that every claim is correct.

## 🏗️ Technology Stack

| Technology | Role |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| Google Gemini | AI-assisted answer generation and research workflows |
| PDF text extraction | Document ingestion and processing |
| Keyword retrieval | Lexical matching against document passages |
| Semantic retrieval | Similarity-based evidence discovery |
| Streamlit Community Cloud | Public application hosting |

## 🖥️ Application Pages

EvidencePilot organizes its workflows into separate pages:

- **Main Workspace** — Upload a PDF, ask questions, and inspect evidence.
- **Answer Audit** — Evaluate claims against retrieved passages.
- **Human Review** — Support human examination of AI-assisted results.
- **Research Agent** — Investigate questions and organize research outputs.
- **Research Audit** — Review research outputs for evidence and citation issues.

The main workspace also includes a Retrieval Evaluation Lab for comparing retrieval approaches.

## 🚀 Try the Live Application

**No local installation is required to try the hosted version.**

👉 **[Open EvidencePilot](https://evidencepilot.streamlit.app/)**

A typical workflow:

1. Open EvidencePilot.
2. Upload a supported, text-based PDF.
3. Ask a question about the document.
4. Inspect the retrieved evidence and available page references.
5. Review the answer and use the audit workflows when additional verification is needed.
6. Evaluate retrieval performance using known supporting pages when appropriate.

**Note:** Scanned PDFs may require OCR before their contents can be searched. OCR is not currently included in the documented workflow.

## ⚙️ Run Locally

### Prerequisites

- Python 3.10 or a compatible Python version supported by the project's dependencies.
- Git.
- A Google Gemini API key.
- The dependencies listed in `requirements.txt`.

### 1. Clone the repository

```bash
git clone https://github.com/AravindSAsok/EvidencePilot.git
cd EvidencePilot
```

### 2. Create and activate a virtual environment

**Windows:**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Create a local file at:

`.streamlit/secrets.toml`

Add the following configuration:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

Replace the placeholder with your own API key.

**Security:** Never commit your API key or `.streamlit/secrets.toml` to GitHub. Keep local secrets excluded through `.gitignore`.

### 5. Start EvidencePilot

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## ☁️ Deployment

EvidencePilot is hosted on Streamlit Community Cloud.

The deployment uses the GitHub repository as its source, with `app.py` as the main entry point. Required credentials should be configured through the platform's secrets settings rather than committed to source control.

To deploy your own instance, see the [official Streamlit Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

## ⚠️ Limitations and Responsible Use

EvidencePilot is an AI-assisted research tool, not an authoritative source of truth.

- **Hallucinations:** AI-generated answers and audits may contain incorrect statements.
- **Citation accuracy:** A valid page reference does not automatically prove that the cited passage supports the claim.
- **Evidence coverage:** Relevant passages may be missed during retrieval.
- **Evaluation scope:** Retrieval metrics depend on the quality and coverage of manually identified relevant pages.
- **PDF compatibility:** Scanned documents may require OCR, which is not currently included.
- **Privacy:** Avoid uploading confidential, sensitive, or personally identifiable information unless you have verified the applicable data-handling and privacy requirements. Requests to external AI services may involve processing document excerpts by those services.

Always verify important claims against the original document. Human review is particularly important for academic, legal, medical, financial, and other high-stakes research.

## 🛣️ Future Improvements

Potential directions for further development include:

- OCR support for scanned PDFs.
- More robust citation-to-claim verification.
- Expanded retrieval benchmarks and evaluation datasets.
- Improved handling of conflicting evidence.
- More comprehensive document comparison workflows.
- Additional transparency around uncertainty and evidence gaps.

These are potential improvements, not claims about currently implemented functionality.

## 👨‍💻 Author

**Aravind S Asok**  
Backend Engineer | AI Engineer

- 🌐 Portfolio: [aravindsasok.netlify.app](https://aravindsasok.netlify.app/)
- 💼 LinkedIn: [linkedin.com/in/aravind-asok](https://linkedin.com/in/aravind-asok)
- 🐙 GitHub: [github.com/AravindSAsok](https://github.com/AravindSAsok)
- 🚀 Live Project: [EvidencePilot](https://evidencepilot.streamlit.app/)

## 📄 License

No license has been specified in this repository yet. Please check the repository for licensing terms before redistributing or reusing the source code.

---

<p align="center">
  <strong>EvidencePilot</strong><br/>
  Research, Grounded in Evidence.
</p>
