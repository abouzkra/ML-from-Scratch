# ML-from-Scratch
My own library of python notebooks where i go through the most important ML models/concepts implemented from scratch using `Numpy`.

# Motivation

Machine Learning is a vast field, everyone that studies it gets hit with the enormous amount of fragmented resources and the 'black-box' nature of modern high-level libraries. This project was built to narrow that gap.

The primary goals of this project are:

- **Demistification through implementation:** As the covered algorithms/concepts are bulit from scratch using `Numpy`, we bypass the abstraction layers of production-ready frameworks (e.g. `scikit-learn` or `PyTorch`) to expose the underlying logic (of course we're not going to implement production-level optimizations but we're going to go through some of them).
- **Unify the learning paht:** This project serves as a great end-to-end guide/roadmap, that gathers foundational concepts into one well ordered/documented reference.
- **A 'First Principles' approach:** This repository targets students/engineers like me, as it prioritizes clarity and mathematical coverage, ensuring that anyone who has to 'start somewhere' can build a robust AI/ML foundation.

# Notebook structure

To ensure consistency and clarity across all notebooks, every notebook should more or less follow this structure:

### 1.Prerequisites
- An overview of the necessary theoretical/technical abilities.

### 2. Introduction
- **What is the 'concept/alogrithm/model...'?**
- **What problem(s) does it solve?**
- **Real world applications**
- **When vs When not to use**

### 3. Concept & Mathematics
- **Mathematical Formulation**
- **Mathematical Derivation** (If applicable)
- **Implementation stategies:** discussion about the implementation approaches (e.g. iterative vs. vectorized/optimized)
- **Time/Space complexity**

### 4. Implementation from scratch using `Numpy`
- **Data Collection/Generation**
- **`Numpy`/ standard python implementation**: Class-based implementations, Docstrings, comments
- **Visualization:** Visualizing algorithm behaviour and performance through evaluation metrics or observable outputs

### 5. References

# Roadmap & Status

![](https://geps.dev/progress/25.0) 7/28 notebooks done

### 01_regression:

- [x] [01_regression](./notebooks/01_regression/01_regression.ipynb)
- [ ] 02_ridge-regression
- [ ] 03_lasso-regression
- [ ] 04_elastic-net

### 02_classification:

- [x] [01_Naive-Bayes](./notebooks/02_classification/01_Naive-Bayes.ipynb)
- [x] [02_KNN](./notebooks/02_classification/02_KNN.ipynb)
- [x] [03_Decision-trees](./notebooks/02_classification/03_Decision-trees.ipynb)
- [ ] 04_random-forests
- [ ] 05_SVM

### 03_clustering:

- [ ] 01_K-Means
- [ ] 02_DBSCAN
- [ ] 03_hierarchical-clustering

### 04_neural_networks:

- [ ] 01_perceptron
- [ ] 02_activation-functions
- [ ] 03_MLP-backprop
- [ ] 04_optimizers
- [ ] 05_layers
- [ ] 06_CNN-foundations
- [ ] 07_RNN-foundations

### 05_dimensionality_reduction:

- [ ] LDA
- [ ] PCA
- [ ] t-SNE

### 06_time_series:
- [ ] 01_TS-fundamentals
- [ ] 02_classical-forecasting
- [ ] 03_ML-for-TS

### 07_information_retrieval:
- [x] [01_TF-IDF](./notebooks/07_information-retrieval/01_TF-IDF.ipynb)
- [x] [02_BM25](./notebooks/07_information-retrieval/01_BM25.ipynb)

### miscellaneous:
- [ ] cosine-similarity.ipynb
