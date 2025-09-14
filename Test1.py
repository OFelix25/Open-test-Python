
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset
df = pd.read_excel("C:/Users/HP/Desktop/Open test 1/Breast Cancer Winscon.xlsx")

# Convert 'Class' to binary: 2 = benign, 4 = malignant
df['Class'] = df['Class'].map({2: 0, 4: 1})

# 1. First EDA (Raw Data)
print("Summary Statistics:")
summary_stats = df.describe().T
summary_stats['IQR'] = df.quantile(0.75) - df.quantile(0.25)
print(summary_stats)

# Histograms for each attribute
df.hist(figsize=(12,10), bins=20)
plt.suptitle('Histograms of Tumor Attributes')
plt.show()

# Boxplots comparing benign vs malignant
for col in df.columns.drop(['Sample_code_number','Class']):
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Class', y=col, data=df)
    plt.title(f'Boxplot of {col} by Tumor Class')
    plt.show()

# Scatter plots between top 2-3 features (based on domain knowledge)
top_features = ['Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape']
sns.pairplot(df, vars=top_features, hue='Class', palette='coolwarm')
plt.suptitle('Scatter Plots of Top Features', y=1.02)
plt.show()

# 2. Preprocessing
# Handle missing values
df['Bare_nuclei'] = pd.to_numeric(df['Bare_nuclei'], errors='coerce')
df['Bare_nuclei'] = df['Bare_nuclei'].fillna(df['Bare_nuclei'].median())

# Separate features and target
X = df.drop(columns=['Sample_code_number', 'Class'])
y = df['Class']

# Feature Engineering: create ratio feature
X['cell_ratio'] = X['Clump_thickness'] / (X['Single_epithelial_cell_size'] + 1)

# Normalize continuous features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Feature Engineering & Reduction
# Recursive Feature Elimination (RFE)
model = LogisticRegression(max_iter=10000)
n_features = min(10, X.shape[1])
rfe = RFE(model, n_features_to_select=n_features)
rfe.fit(X_scaled, y)
selected_features = X.columns[rfe.support_]
print("Selected features (RFE):", selected_features)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio (PCA):", pca.explained_variance_ratio_)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 4. Second EDA (Post-Processing)
# Feature importance using Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_scaled, y)
plt.figure(figsize=(12,6))
plot_tree(dt, feature_names=X.columns, class_names=['Benign','Malignant'], filled=True, max_depth=3)
plt.title('Decision Tree Feature Importance (Top Levels)')
plt.show()

# 5. ML Modeling
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
y_prob_nb = nb.predict_proba(X_test)[:, 1]

# k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# 6. Evaluation
# Import libraries (same as before + calibration)
from sklearn.calibration import calibration_curve

# 6. Evaluation (updated with calibration curve)
def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

# Naive Bayes
acc_nb = accuracy_score(y_test, y_pred_nb)
sens_nb, spec_nb = sensitivity_specificity(y_test, y_pred_nb)
roc_nb = roc_auc_score(y_test, y_prob_nb)
print(f"Naive Bayes -> Accuracy: {acc_nb:.3f}, Sensitivity: {sens_nb:.3f}, Specificity: {spec_nb:.3f}, ROC-AUC: {roc_nb:.3f}")

# Calibration curve for Naive Bayes
prob_true, prob_pred = calibration_curve(y_test, y_prob_nb, n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Naive Bayes')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curve (Naive Bayes)')
plt.legend()
plt.show()

# k-NN
acc_knn = accuracy_score(y_test, y_pred_knn)
sens_knn, spec_knn = sensitivity_specificity(y_test, y_pred_knn)
print(f"k-NN -> Accuracy: {acc_knn:.3f}, Sensitivity: {sens_knn:.3f}, Specificity: {spec_knn:.3f}")

# 7. Visualization of Dimensionality Reduction
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='coolwarm')
plt.title('PCA')

plt.subplot(2,2,2)
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='coolwarm')
plt.title('t-SNE')

plt.subplot(2,2,3)
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=y, palette='coolwarm')
plt.title('UMAP')

plt.tight_layout()
plt.show()

# 8. Interpretation
print("\nInterpretation:")
print("Features that best separate benign/malignant tumors can be inferred from RFE selected features and Decision Tree importance.")
print("False negatives in healthcare are critical: a malignant tumor misclassified as benign could delay treatment and endanger patient health.")
