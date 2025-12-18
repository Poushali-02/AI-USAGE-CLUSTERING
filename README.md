# AI-USAGE-CLUSTERING

A machine learning project focused on clustering and analyzing AI usage patterns using unsupervised learning techniques.
Web application live at https://ai-usage-clustering.onrender.com/
## 📋 Overview

This project implements clustering algorithms to identify and analyze patterns in AI tool usage data. By grouping similar usage behaviors, the system helps understand how different users interact with AI systems, enabling better insights for optimization and personalization.

## 🎯 Objectives

- Analyze AI usage patterns across different user segments
- Identify clusters of similar usage behaviors
- Provide insights for improving AI tool adoption and user experience
- Enable data-driven decision-making for AI product development

## 🔧 Technologies & Tools

- **Python** - Primary programming language
- **Scikit-learn** - Machine learning algorithms and clustering
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## 🚀 Features

- **Data Preprocessing**: Clean and prepare AI usage data for analysis
- **Feature Engineering**: Extract meaningful features from usage patterns
- **Clustering Algorithms**: Implementation of multiple clustering techniques:
  - K-Means Clustering
  - Hierarchical Clustering
- **Visualization**: Interactive plots and charts for cluster analysis
- **Evaluation Metrics**: Silhouette score, Davies-Bouldin index, and elbow method

## 📊 Dataset
The Kaggle notebook is available [here](https://www.kaggle.com/code/poushal02/ai-assistant-usage)
The dataset used in this project is available [here](https://www.kaggle.com/datasets/ayeshasal89/ai-assistant-usage-in-student-life-synthetic)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Poushali-02/AI-USAGE-CLUSTERING.git
cd AI-USAGE-CLUSTERING

```
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

```
3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the python file
```bash
python app.py
```

## 📈 Methodology

1. **Data Collection**: Gather AI usage data from various sources
2. **Data Cleaning**: Handle missing values and outliers
3. **Feature Scaling**: Normalize features for better clustering performance
4. **Dimensionality Reduction**: Apply PCA if needed for high-dimensional data
5. **Clustering**: Apply multiple algorithms and compare results
6. **Validation**: Evaluate cluster quality using multiple metrics
7. **Interpretation**: Analyze cluster characteristics and insights

## 📊 Results

The clustering analysis provides:
- Identification of distinct user segments based on AI usage
- Behavioral patterns within each cluster
- Recommendations for targeted improvements
- Insights for personalization strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Project Structure

AI-USAGE-CLUSTERING/
- ├── static/
- ├── templates/ 
- ├── .gitignore
- ├── app.py
- ├── model.pkl # model made on kaggle
- ├── requirements.txt # Project dependencies
- └── README.md # Project documentation

## 📚 Key Concepts

- **Unsupervised Learning**: Machine learning without labeled data
- **Cluster Analysis**: Grouping similar data points together
- **Feature Engineering**: Creating meaningful variables from raw data
- **Evaluation Metrics**: Assessing clustering quality
---

⭐ If you find this project useful, please consider giving it a star!
