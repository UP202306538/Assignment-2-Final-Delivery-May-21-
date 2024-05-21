import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import json

# Carregar o dataset
data = pd.read_csv('D:\\Escola\\FCUP\\EIACC\\EIACD_Assignment_2_2023_2024\\EIACD_Assignment_2_2023_2024\\hcc_dataset.csv')

# Substituir "?" por NaN
data.replace("?", np.nan, inplace=True)

# Definir colunas numéricas e categóricas manualmente
numeric_columns = [
    'Age', 'Grams_day', 'Packs_year', 'INR', 'AFP', 'Hemoglobin', 'MCV', 'Leucocytes',
    'Platelets', ' Albumin', 'Total_Bil', 'ALT', 'AST', 'GGT', 'ALP', 'TP', 'Creatinine',
    'Nodules', 'Major_Dim', 'Dir_Bil', 'Iron', 'Sat', 'Ferritin'
]
categorical_columns = [
    'Gender', 'Symptoms', 'Alcohol', 'HBsAg', 'HBeAg', 'HBcAb', 'HCVAb', 'Cirrhosis',
    'Endemic', 'Smoking', 'Diabetes', 'Obesity', 'Hemochro', 'AHT', 'CRI', 'HIV', 'NASH',
    'Varices', 'Spleno', 'PHT', 'PVT', 'Metastasis', 'Hallmark', 'PS', 'Encephalopathy', 'Ascites'
]

# Imputação de valores ausentes
# Imputação para colunas numéricas (usando a média)
numeric_imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

# Imputação para colunas categóricas (usando a moda)
for col in categorical_columns:
    if data[col].isna().sum() > 0:  # Apenas processar se houver NaNs
        mode = data[col].mode()[0]  # A moda é o valor mais frequente
        data[col] = data[col].fillna(mode)

# Exploratory Data Analysis (EDA)
print(data.head())

print("Informações do Dataset:")
print(data.info())

print("\nEstatísticas Descritivas:")
print(data.describe())

print("\nDistribuição das Classes:")
sns.countplot(x='Class', data=data)
plt.title('Distribuição das Classes')
plt.show()

# Separar features e target
X = data.drop('Class', axis=1)
y = data['Class']

# Pipeline de pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Aplicar o pipeline de pré-processamento nos dados
X_preprocessed = preprocessor.fit_transform(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_proba_dt = dt.predict_proba(X_test)[:, 1]

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_proba_knn = knn.predict_proba(X_test)[:, 1]

# Convertendo rótulos para valores binários
label_binarizer = LabelBinarizer()
y_test_bin = label_binarizer.fit_transform(y_test)

# Calculando a curva ROC usando os rótulos binários
fpr_dt, tpr_dt, _ = roc_curve(y_test_bin, y_proba_dt)
fpr_knn, tpr_knn, _ = roc_curve(y_test_bin, y_proba_knn)
roc_auc_dt = auc(fpr_dt, tpr_dt)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Avaliação dos Modelos
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

roc_auc_dt = roc_auc_score(y_test, y_proba_dt)

print(f"\nAUC-ROC para Decision Tree: {roc_auc_dt}")
print(f"AUC-ROC para KNN: {roc_auc_knn}")

# Plotar a curva ROC para KNN
plt.figure()
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='KNN ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot(fpr_dt, tpr_dt, color='red', lw=2, label='DT ROC curve (area = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Conclusões
print("\nConclusões:")
print("Os modelos de Decision Tree e KNN foram treinados e avaliados com base em métricas de classificação.")
print(f"O modelo de Decision Tree apresentou uma AUC-ROC de {roc_auc_dt:.2f}, enquanto o modelo KNN apresentou uma AUC-ROC de {roc_auc_knn:.2f}.")
print("As curvas ROC mostraram que ambos os modelos têm um desempenho razoável, mas há espaço para melhorias, como a otimização de hiperparâmetros e a utilização de técnicas avançadas de pré-processamento de dados.")

# Salvar resultados em um arquivo
results = {
    "Decision Tree": {
        "Classification Report": classification_report(y_test, y_pred_dt, output_dict=True),
        "Confusion Matrix": confusion_matrix(y_test, y_pred_dt).tolist(),
        "AUC-ROC": roc_auc_dt
    },
    "KNN": {
        "Classification Report": classification_report(y_test, y_pred_knn, output_dict=True),
        "Confusion Matrix": confusion_matrix(y_test, y_pred_knn).tolist(),
        "AUC-ROC": roc_auc_knn
    }
}