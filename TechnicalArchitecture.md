# Technical Architecture

## 1. Project Infrastructure

The Dell Social Impact AI Datathon uses the NVIDIA GB10 platform integrated into the Dell Pro Max workstation to support local AI development and experimentation.

The GB10 features NVIDIA Grace Blackwell architecture and 128 GB of unified memory shared between the CPU and GPU.

This hardware provides the computing resources needed to experiment with large language models and AI-powered applications.

## 2. Development Environment

The workshop introduced the following development tools and technologies:

- **Linux:** Operating system environment and command-line operations.
- **SSH:** Secure remote access to the GB10 workstation.
- **Python:** Primary programming language for AI development and data processing.
- **Python Virtual Environments:** Dependency isolation and project environment management.
- **Jupyter Notebooks:** Interactive development, experimentation, and data analysis.
- **Git:** Source control and version tracking.
- **GitHub:** Project collaboration and code documentation.

## 3. AI Infrastructure and Model Deployment

The GB10 workstation supports local execution of large language models.

The workshop covered several model-serving technologies:

- LM Studio
- vLLM
- Quen
- Jupyter Lab
- Streamlit

These tools provide different approaches to running and interacting with local AI models.

## 4. Large Language Model Fundamentals

The workshop introduced foundational LLM concepts:

- Tokens and tokenization
- Context windows
- Temperature and model response variability
- Prompt engineering
- Local model inference
- API-based model interaction

These concepts inform how the project retrieves information, processes prompts, and generates responses.

## 5. API Integration

The project explores communication with a locally hosted or remotely accessible LLM through an API endpoint.

The workshop introduced OpenAI-compatible API calls, allowing Python applications to communicate with supported model servers using a standardized client interface.

Implementation details, including the endpoint configuration, model identifier, and request workflow, will be documented separately.

## 6. Community Data Integration

The project's data pipeline will incorporate publicly available San Antonio municipal datasets.

Planned development stages:

1. Identify relevant public datasets.
2. Retrieve and organize source data.
3. Process and prepare data for AI retrieval.
4. Integrate relevant data with the LLM workflow.
5. Generate responses and insights based on the available information.
6. Evaluate the usefulness and accuracy of the resulting insights.

## 7. System Architecture

Public San Antonio Data → Data Collection → Data Processing → LLM Integration → AI-Generated Insights → Community Impact Application

## 8. Security and Governance Considerations

The project considers:

- Secure remote server access.
- API credential protection.
- Data provenance and source verification.
- Responsible use of public datasets.
- Accuracy and reliability of AI-generated responses.
- Privacy and responsible AI governance.
