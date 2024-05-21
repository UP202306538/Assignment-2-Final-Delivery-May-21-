import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

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
numeric_imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

for col in categorical_columns:
    if data[col].isna().sum() > 0:
        mode = data[col].mode()[0]
        data[col] = data[col].fillna(mode)

# Separar features e target
X = data.drop('Class', axis=1)
y = data['Class']

# Pipeline de pré-processamento sem SelectKBest
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10))
        ]), numeric_columns),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_columns)
    ])

# Aplicar o pipeline de pré-processamento nos dados
preprocessor.fit(X)

# Salvar o pré-processador ajustado
joblib.dump(preprocessor, 'D:\\Escola\\FCUP\\pycharm\\pythonProject\\preprocessor.pkl')

# Seleção de Features após pré-processamento
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_preprocessed, y)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Otimização de hiperparâmetros para Decision Tree
param_grid_dt = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                              param_grid=param_grid_dt,
                              cv=5, n_jobs=-1, scoring='roc_auc')
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_

# Otimização de hiperparâmetros para KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(),
                               param_grid=param_grid_knn,
                               cv=5, n_jobs=-1, scoring='roc_auc')
grid_search_knn.fit(X_train, y_train)
best_knn = grid_search_knn.best_estimator_

# Salvar os modelos treinados
joblib.dump(best_dt, 'D:\\Escola\\FCUP\\pycharm\\pythonProject\\best_dt_model.pkl')
joblib.dump(best_knn, 'D:\\Escola\\FCUP\\pycharm\\pythonProject\\best_knn_model.pkl')