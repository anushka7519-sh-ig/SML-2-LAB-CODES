# SML-2 Lab Repository (ED408)

Python implementations for Statistical Machine Learning II, covering classical machine learning techniques and metaheuristic optimization.

## Dataset Adaptation (Module 1)

For the first module, per lab instructions, all algorithms are applied to a single, custom tabular dataset derived from the BSDS500 (Berkeley Segmentation Data Set) image archive. 

- Transformation: We converted an image segmentation task into a genuine binary classification problem. One row = one pixel, uniformly sampled from all 200 training images.
- Features: Color (`R, G, B, gray`), gradient/texture (`grad_mag, grad_dir, laplacian, local_std`), and spatial coordinates (`x_norm, y_norm`).
- Target: `is_edge` — labeled as 1 if a majority of the 6 human annotators marked the pixel as a boundary, otherwise 0.
- Balance: 24,000 rows, perfectly balanced (12,000 edge / 12,000 non-edge).

The exact same dataset (`data/bsds_features.csv`) is reused across all Module 1 questions.

## Structure

```text
data/bsds_features.csv                    # The derived tabular dataset
module_1/01_knn_scratch.py                # Q1 - k-NN from scratch
module_1/02_decision_tree.py              # Q2 - Decision Tree + viz
module_1/03_naive_bayes.py                # Q3 - Naive Bayes classification
module_1/04_logistic_regression.py        # Q4 - Logistic Regression from scratch
module_1/05_pca_tsne.py                   # Q5 - PCA & t-SNE reduction
module_1/06_outliers.py                   # Q6 - Outlier detection (z-score, IQR)
module_1/07_lr_vs_svm.py                  # Q7 - Metric comparison (LR vs SVM)
module_2/08_simulated_annealing.ipynb     # Q8 - SA (Numerical Example)
module_2/09_genetic_algorithm.ipynb       # Q9 - GA (Numerical Example)
results/figures/                          # Generated plots and visualizations
results/metrics/                          # Numeric outputs and classification reports


python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter
# Run a specific Python script:
python3 module_1/01_knn_scratch.py
# Or launch Jupyter for Module 2 notebooks:
jupyter notebook
