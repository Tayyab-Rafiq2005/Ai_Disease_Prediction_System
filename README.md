# AI Disease Prediction System & Academic AI Portfolio

An end-to-end medical predictive analysis suite paired with a rigorous 15-part Artificial Intelligence curriculum covering classical machine learning, deep neural networks, clustering, and modern Generative AI (Retrieval-Augmented Generation).

---

## 🛠️ Tech Stack

### Core Technologies
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Generative AI & Vector Engines (Lab 15)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0052CC?style=for-the-badge&logo=meta&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-black?style=for-the-badge&logo=ollama&logoColor=white)

---

## 📌 Repository Overview

This repository houses two primary components:
1. **The Disease Prediction System (Production-ready Application)**: A machine learning-driven clinical engine capable of diagnosing ailments based on symptoms using supervised learning techniques.
2. **AI Lab Manuals Curriculum (Labs 1–15)**: A comprehensive educational journey tracking the evolution of AI. It spans search strategies, deep artificial neural networks (DNNs), convolutional neural networks (CNNs), K-Means & Agglomerative clustering, and concludes with a localized **Retrieval-Augmented Generation (RAG) chatbot** built using LangChain, Streamlit, and Llama 3 (via Ollama).

---

## 🚀 Key Features

* **Intelligent Clinical Diagnostics**: The core Disease Prediction System uses optimized multi-class classification algorithms to map patient symptoms to precise disease classifications.
* **Interactive UI & Widgets**: Uses `ipywidgets` inside Jupyter Notebooks to construct responsive symptom selectors and prediction panels.
* **Diverse Lab Experiments Suite**:
  * **Labs 1–5**: Search algorithms, agent environments, and foundational heuristics.
  * **Labs 6 & 9**: Deep Neural Networks (DNN) and Convolutional Neural Networks (CNN) for computer vision and complex classifications.
  * **Labs 10 & 11**: Unsupervised grouping using K-Means and Agglomerative Hierarchical Clustering.
  * **Lab 15 (RAG System)**: An on-premise, privacy-first chatbot that parses documents, indexes them in a local FAISS vector store, and queries them using local LLMs.

---

## 📂 Project Directory Structure

```directory
.
├── Ai-Project_Report/
│   └── Project-Report.pdf                 # Comprehensive project documentation
├── DiseasePredictionSystem/
│   ├── Disease_Prediction.ipynb           # Main clinical diagnosis system
│   └── requirements.txt                   # Classical ML dependencies
├── Lab_Manuals(AI)/                       # Complete course progression folder
│   ├── lab1/ to lab14/                    # Foundational AI labs & assignments (Jupyter/Python)
│   └── lab15/                             # Next-Gen RAG chatbot suite
│       ├── app.py                         # Streamlit front-end
│       ├── localama.py                    # Local Ollama helper interface
│       └── requirements.txt               # LLM/Generative AI dependencies
└── labs 1-15.pdf                          # Master compilation of syllabus details
```

---

## 🔧 Installation & Environment Setup

This project features two separate dependencies scopes depending on what you wish to run: **Classical ML Diagnostics** or **Generative AI Chatbots**. 

We recommend managing environments with `virtualenv` or `conda`.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Tayyab-Rafiq2005/Ai_Disease_Prediction_System.git
cd Ai_Disease_Prediction_System
```

---

### Option A: Running the Disease Prediction System (Classical ML)

1. Create and activate a clean Python virtual environment:
   ```bash
   python -m venv disease-env
   source disease-env/bin/activate  # On Windows: disease-env\Scripts\activate
   ```

2. Install the necessary system requirements:
   ```bash
   pip install -r DiseasePredictionSystem/requirements.txt
   ```

3. Launch the prediction notebook:
   ```bash
   jupyter notebook DiseasePredictionSystem/Disease_Prediction.ipynb
   ```

---

### Option B: Running the Lab 15 RAG Chatbot (Generative AI)

1. Create and activate a clean Python virtual environment:
   ```bash
   python -m venv rag-env
   source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r "Lab_Manuals(AI)/lab15/requirements.txt"
   ```

3. Install and run **Ollama** on your local machine ([Download Ollama](https://ollama.com)):
   Ensure Ollama is running, and pull your target local model (e.g., Llama 3):
   ```bash
   ollama pull llama3
   ```

4. Launch the Streamlit application:
   ```bash
   cd "Lab_Manuals(AI)/lab15"
   streamlit run app.py
   ```

---

## 🖥️ Usage & Execution Examples

### Core Disease Classifier

Inside `Disease_Prediction.ipynb`, load symptom metrics via interactive drop-down menus:

```python
# Select symptoms interactively
import ipywidgets as widgets
from IPython.display import display

symptom_1 = widgets.Dropdown(options=['itching', 'skin_rash', 'nodal_skin_eruptions', ...])
display(symptom_1)

# The pipeline processes inputs and executes prediction
prediction = model.predict([selected_features])
print(f"Diagnosed Medical Condition: {prediction[0]}")
```

### Local RAG Querying (Lab 15)

The vector generation pipeline uses local FAISS integrations to answer contextual documents:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize local Vector Index from document chunks
db = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = db.as_retriever(search_kwargs={"k": 3})

# Query local corpus
docs = retriever.get_relevant_documents("What are the core systems of AI disease prediction?")
```

---

## 🤝 Contributing

Contributions to improve diagnostic performance, expand the lab exercises, or optimize vector retrieval are welcome. 

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

This repository is distributed under the **MIT License**. For detailed terms, see the [LICENSE](LICENSE) file (if present) or standard MIT guidelines.