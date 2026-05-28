# Sentiment Analysis of Movie Reviews using NLP

## Project Overview

This project builds a full Natural Language Processing (NLP) pipeline for sentiment analysis of movie reviews.The system classifies reviews as either **positive** or **negative** by using traditional machine learning algorithms along with the latest deep learning models based on transformers.

This notebook explains everything from getting and cleaning the data to training the model, evaluating the results, comparing the different methods, and trying them out in a real-world situation.


---

#  Main Features

- IMDB Movie Review Dataset (50,000 reviews)

- Text preprocessing and cleaning

- TF-IDF feature extraction

- Progressive model evaluation (1000 → 10000 → 25000)

- Naive Bayes classifier

- Support Vector Machine (SVM)

- Feed-forward Neural Network

- DistilBERT transformer model

- Confusion matrices and visualizations

- Execution-time comparison

- Real-world sentiment prediction examples

- Error analysis and limitations discussion

- Reproducible workflow with fixed random seeds

---

#  Technologies and Libraries Used

| Technology / Library  | The Purpose                           |
| --------------------- | --------------------------------- |
| Python                | Main programming language         |
| Google Colab GPU      | Faster DistilBERT inference       |
| Hugging Face datasets | Loading IMDB dataset              |
| scikit-learn          | TF-IDF, Naive Bayes, SVM, metrics |
| TensorFlow / Keras    | Neural Network implementation     |
| Transformers          | DistilBERT sentiment analysis     |
| Matplotlib / Seaborn  | Visualizations                    |
| NLTK                  | Stopwords and preprocessing       |
| WordCloud             | Word cloud visualization          |

---

#  Dataset

The project uses the publicly available:

**Stanford Large Movie Review Dataset (IMDB)**

* 25,000 training reviews
* 25,000 testing reviews
* Balanced positive and negative labels

Dataset source:

[https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)

---

#  Project Workflow

The NLP pipeline goes through the following stages:

1. Import libraries
2. Load IMDB dataset
3. Explore dataset distribution
4. Preprocess text
5. Generate progressive subsets
6. Create word clouds
7. Convert text using TF-IDF
8. Train classical ML models
9. Train neural network
10. Evaluate DistilBERT
11. Compare model performance
12. Test on unseen movie reviews
13. Perform error analysis
14. Discuss limitations and future work

---

#  Models Used

| Model                       | Type                         |
| --------------------------- | ---------------------------- |
| Naive Bayes                 | Traditional Machine Learning |
| Linear SVM                  | Traditional Machine Learning |
| Feed-forward Neural Network | Deep Learning                |
| DistilBERT                  | Transformer-based NLP        |

---

#  Final Results

| Model          | Accuracy | Precision | Recall | F1-score |
| -------------- | -------- | --------- | ------ | -------- |
| DistilBERT     | 0.8907   | 0.9146    | 0.8619 | 0.8875   |
| SVM            | 0.8619   | 0.8676    | 0.8541 | 0.8608   |
| Neural Network | 0.8663   | 0.8974    | 0.8271 | 0.8608   |
| Naive Bayes    | 0.8410   | 0.8552    | 0.8210 | 0.8377   |

### Best Performing Model

Thanks to its contextual understanding of language, DistilBERT was the one in the lead overall.

---

#  Screenshots of my notebook outcomes

**Dataset Distribution**  
![dataset_distribution]

**Word Clouds**  
![wordClouds]

**SVM Confusion Matrix**  
![sVMconfusionmatrix]

**DistilBERT Confusion Matrix**  
![distilBERTconfusionmatrix]

**Final Model Comparison Graph**  
![finalmodelcomparisongraph]

**Real-World Prediction Results**  
![applicationtesting]


---

#  Installation

## 1. Clone the repository

```bash
git clone https://github.com/BrailleNshuti/Project-NLP-DLBAIPNLP01-Sentiment-Analysis-of-Movie-Reviews
```

## 2. Open the notebook

Open the notebook in:

* Google Colab (it is recommended as it offers GPU)
* Jupyter Notebook
* VS Code

---

#  Running the Project

Run the notebook cells sequentially from top to bottom.

The notebook includes:

* Data loading
* Text preprocessing
* TF-IDF vectorization
* Model training
* Model evaluation
* Visualization
* Real-world testing

---

# Reproducibility

Random seeds settings were done for NumPy, TensorFlow, and Python random modules in order to make the experiment reproducible and the results consistent across different runs.

# Limitations

* Instead of fine-tuning directly on the IMDB dataset, DistilBERT was only pre-trained on the SST-2 dataset.
* The neural network is working on sparse TF-IDF features, whereas the state-of-the-art way is to use contextual embeddings.
* Transformer models require substantial computational resources, compared to traditional machine learning models.

# Future Improvements

The following are some of the ways this project could be improved or extended:
* Fine-tuning DistilBERT model directly on the IMDB dataset.
* Running hyperparameter optimization.
* Conducting cross-validation experiments.
* Using larger transformer models.
* Deploying the application on the web using Streamlit or Flask.
* Expanding to multi-class sentiment analysis.


---

#  Repository Structure

```
Project NLP/
│
├── README.md
├── .gitignore
├── requirements.txt
├── PROJECT_NLP.ipynb
└── screenshots/
   ├── dataset_distribution.png
   ├── wordclouds.png
   ├── svm_confusion_matrix.png
   ├── distilbert_confusion_matrix.png
   ├── final_model_comparison.png
   └── application_testing.png
```

---

#  Reflection

This project shows a hands-on implementation of Natural Language Processing with various techniques starting from traditional machine learning up to transformer-based deep learning. Besides, it draws attention to the factors that play a crucial role in building NLP systems like preprocessing, feature engineering, reproducibility, model comparison, and critical evaluation.


---

#  Author

**RUSINGIZA NSHUTI BRAILLE**

Applied Artificial Intelligence, B.Sc.

IU International University of Applied Sciences

---

#  Repository Link:
https://github.com/BrailleNshuti/Project-NLP-DLBAIPNLP01-Sentiment-Analysis-of-Movie-Reviews
