# Text Classification App with XLM-RoBERTa

This repository contains a FastAPI application for text classification using a pre-trained XLM-RoBERTa model. The model is trained on English data but can predict text in any language. The project includes:

- **`app.py`**: A FastAPI application for predicting the label of a single sentence.
- **`train_notebook.ipynb`**: A Jupyter notebook for training the XLM-RoBERTa model on your own dataset.

## Project Overview

### FastAPI Application (`app.py`)

The `app.py` file provides an interface for predicting the classification of a single sentence. The model uses XLM-RoBERTa, which has been trained on English data and can classify text in multiple languages.

#### Features:
- **Predict Single Sentence**: Enter a sentence to receive a classification label and confidence score.
- **Color-Coded Labels**: Labels are color-coded for easy identification.

### Training Notebook (`train_notebook.ipynb`)

The Jupyter notebook demonstrates how to train the XLM-RoBERTa model using a dataset. The notebook includes steps for loading data, training the model, and saving the trained model for deployment.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   '''
2. **Install Dependencies**
    ```bash
   pip install fastapi uvicorn transformers torch pandas jupyter

3.Running the FastAPI Application

Run the following command in your terminal:

    '''bash
        uvicorn app:app --reload


## Training the Model
Run the notebook 



The notebook contains detailed instructions for preparing your dataset, training the model, and saving the trained model.



