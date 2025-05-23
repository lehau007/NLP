# NLP Training Tasks

This repository encompasses a series of Natural Language Processing (NLP) training tasks, focusing on various models and techniques for Vietnamese language processing.

## 📁 Project Structure

The repository is organized into the following directories:

- **`BERT/`**: Implementations and experiments using BERT models.
- **`BiLSTM_CNN_NER/`**: BiLSTM-CNN architecture for Named Entity Recognition tasks.
- **`SVM/`**: Support Vector Machine models for text classification.
- **`IELTS_Generation/`**: Fine-tuning GPT-2 models for generating IELTS-style essays.

## 🛠️ Technologies Used

- **Programming Languages**: Python
- **Libraries**:
  - [PyTorch](https://pytorch.org/)
  - [Transformers](https://huggingface.co/transformers/)
  - [scikit-learn](https://scikit-learn.org/)
  - [spaCy](https://spacy.io/)
  - [NLTK](https://www.nltk.org/)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/lehau007/NLP.git
cd NLP
```

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, manually install the necessary libraries:

```bash
pip install torch transformers scikit-learn spacy nltk
```

### 4. Download Necessary NLP Models

For certain tasks, you may need to download additional NLP models:

- **spaCy Vietnamese Model**:

```bash
python -m spacy download vi_core_news_sm
```

- **NLTK Data**:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🧪 Running the Models

Navigate to the desired directory and follow the instructions provided in the respective `README.md` or script files. For example:

```bash
cd BERT
python main.py
```

> Ensure that you have the necessary datasets and configurations set up as per the instructions in each directory.
