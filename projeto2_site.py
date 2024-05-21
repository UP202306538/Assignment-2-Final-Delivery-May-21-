import streamlit as st
import pandas as pd
import joblib
import time

# Carregar o pré-processador ajustado
preprocessor = joblib.load('D:\\Escola\\FCUP\\pycharm\\pythonProject\\preprocessor.pkl')

# Carregar o modelo de Decision Tree treinado
best_dt = joblib.load('D:\\Escola\\FCUP\\pycharm\\pythonProject\\best_dt_model.pkl')

# Carregar o modelo KNN treinado
best_knn = joblib.load('D:\\Escola\\FCUP\\pycharm\\pythonProject\\best_knn_model.pkl')

st.title('Aplicativo de Previsão de Diagnóstico de Hepatocarcinoma')

# Coletar dados de entrada do usuário
st.sidebar.header('Parâmetros do Paciente')
name = st.sidebar.text_input('Nome da Pessoa', value=None)
age = st.sidebar.number_input('Idade', min_value=0, max_value=120, value=None)
gender = st.sidebar.selectbox('Gênero', ['Male', 'Female'], index=None)
symptoms = st.sidebar.selectbox('Sintomas', ['Yes', 'No'], index=None)
alcohol = st.sidebar.selectbox('Alcoolico', ['Yes', 'No'], index=None)
hbsag = st.sidebar.selectbox('HBsAg', ['Yes', 'No'], index=None)
hbeag = st.sidebar.selectbox('HBeAg', ['Yes', 'No'], index=None)
hbcab = st.sidebar.selectbox('HBcAb', ['Yes', 'No'], index=None)
hcvab = st.sidebar.selectbox('HCVAb', ['Yes', 'No'], index=None)
cirrhosis = st.sidebar.selectbox('Cirrhosis', ['Yes', 'No'], index=None)
endemic = st.sidebar.selectbox('Endemic', ['Yes', 'No'], index=None)
smoking = st.sidebar.selectbox('Smoking', ['Yes', 'No'], index=None)
diabetes = st.sidebar.selectbox('Diabetes', ['Yes', 'No'], index=None)
obesity = st.sidebar.selectbox('Obesity', ['Yes', 'No'], index=None)
hemochro = st.sidebar.selectbox('Hemochro', ['Yes', 'No'], index=None)
aht = st.sidebar.selectbox('AHT', ['Yes', 'No'], index=None)
cri = st.sidebar.selectbox('CRI', ['Yes', 'No'], index=None)
hiv = st.sidebar.selectbox('HIV', ['Yes', 'No'], index=None)
nash = st.sidebar.selectbox('NASH', ['Yes', 'No'], index=None)
varices = st.sidebar.selectbox('Varices', ['Yes', 'No'], index=None)
spleno = st.sidebar.selectbox('Spleno', ['Yes', 'No'], index=None)
pht = st.sidebar.selectbox('PHT', ['Yes', 'No'], index=None)
pvt = st.sidebar.selectbox('PVT', ['Yes', 'No'], index=None)
metastasis = st.sidebar.selectbox('Metastasis', ['Yes', 'No'], index=None)
hallmark = st.sidebar.selectbox('Hallmark', ['Yes', 'No'], index=None)
grams_day = st.sidebar.number_input('Consumo de Álcool (gramas por dia)', min_value=0, value=None)
packs_year = st.sidebar.number_input('Packs Year', min_value=0, value=None)
ps = st.sidebar.selectbox('PS', ['Active', 'Ambulatory', 'Restricted', 'Selfcare', 'Disabled', 'None'])
encephalopathy = st.sidebar.selectbox('Encephalopathy', ['Grade I/II', 'Grade III/IV', 'None'])
ascites = st.sidebar.selectbox('Ascites', ['Mild', 'Moderate/Severe'])
inr = st.sidebar.number_input('INR', min_value=0.0, value=None)
afp = st.sidebar.number_input('AFP', min_value=0.0, value=None)
hemoglobin = st.sidebar.number_input('Hemoglobina (g/dL)', min_value=0.0, value=None)
mcv = st.sidebar.number_input('MCV (fl)', min_value=0.0, value=None)
leucocytes = st.sidebar.number_input('Leucócitos (10^9/L)', min_value=0.0, value=None)
platelets = st.sidebar.number_input('Plaquetas (10^9/L)', min_value=0.0, value=None)
albumin = st.sidebar.number_input('Albumina (g/dL)', min_value=0.0, value=None)
total_bil = st.sidebar.number_input('Bilirrubina Total', min_value=0.0, value=None, step=0.1)
alt = st.sidebar.number_input('ALT (U/L)', min_value=0.0, value=None)
ast = st.sidebar.number_input('AST (U/L)', min_value=0.0, value=None)
ggt = st.sidebar.number_input('GGT (U/L)', min_value=0.0, value=None)
alp = st.sidebar.number_input('ALP', min_value=0.0, value=None, step=0.1)
tp = st.sidebar.number_input('TP (U/L)', min_value=0.0, value=None)
creatinine = st.sidebar.number_input('Creatinina', min_value=0.0, value=None)
nodules = st.sidebar.number_input('Nodulos', min_value=0.0, value=None)
major_dim = st.sidebar.number_input('Dimensão Maior do Tumor (cm)', min_value=0.0, value=None)
dir_bil = st.sidebar.number_input('Bilirrubina Direta (mg/dL)', min_value=0.0, max_value=10.0, value=None, step=0.1)
iron = st.sidebar.number_input('Ferro (µg/dL)', min_value=0.0, value=None)
sat = st.sidebar.number_input('Sat (µg/dL)', min_value=0.0, value=None)
ferritin = st.sidebar.number_input('Ferritina (ng/mL)', min_value=0.0, value=None)

# Escolher modelo para fazer a previsão
selected_model = st.sidebar.radio("Escolha o modelo para fazer a previsão:", options=[None, 'Decision Tree', 'KNN'])

# Se o modelo selecionado for None, isso significa que nenhum modelo foi selecionado ainda
if selected_model is None:
    st.warning("Por favor, selecione um modelo para fazer a previsão.")
    prediction = None
else:
    # Criar um DataFrame a partir dos dados de entrada do usuário
    input_data = pd.DataFrame({
        'Name': [name],
        'Age': [age],
        'Gender': [gender],
        'Grams_day': [grams_day],
        'Symptoms' : [symptoms],
        'Alcohol' : [alcohol],
        'HBsAg': [hbsag],
        'HBeAg': [hbeag],
        'HBcAb': [hbcab],
        'HCVAb': [hcvab],
        'Cirrhosis': [cirrhosis],
        'Endemic': [endemic],
        'Smoking': [smoking],
        'Diabetes': [diabetes],
        'Obesity': [obesity],
        'Hemochro': [hemochro],
        'AHT': [aht],
        'CRI': [cri],
        'HIV': [hiv],
        'NASH': [nash],
        'Varices': [varices],
        'Spleno': [spleno],
        'PHT': [pht],
        'PVT': [pvt],
        'Metastasis': [metastasis],
        'Hallmark': [hallmark],
        'PS': [ps],
        'Encephalopathy': [encephalopathy],
        'Ascites': [ascites],
        'Platelets': [platelets],
        'MCV': [mcv],
        'Leucocytes': [leucocytes],
        'Hemoglobin': [hemoglobin],
        'Major_Dim': [major_dim],
        'Dir_Bil': [dir_bil],
        'Albumin': [albumin],
        'ALT': [alt],
        'AST': [ast],
        'GGT': [ggt],
        'AFP': [afp],
        'Creatinine': [creatinine],
        'INR': [inr],
        ' Albumin': [albumin],
        'Total_Bil': [total_bil],
        'TP': [tp],
        'Packs_year': [packs_year],
        'ALP': [alp],
        'Nodules': [nodules],
        'Iron': [iron],
        'Sat': [sat],
        'Ferritin': [ferritin]
    })

    # Aplicar o pipeline de pré-processamento nos dados de entrada do usuário
    X_preprocessed_input = preprocessor.transform(input_data)

    # Selecionar apenas as features relevantes
    X_selected = X_preprocessed_input[:, :10]  # Selecionar as 10 primeiras features após o pré-processamento

    # Fazer previsão usando o modelo selecionado
    if selected_model == 'Decision Tree':
        prediction = best_dt.predict(X_selected)
    elif selected_model == 'KNN':
        prediction = best_knn.predict(X_selected)
    else:
        prediction = None

# Exibir resultados
st.write("### Resultados do Diagnóstico")
st.write("Nome:", name)
if prediction == None:
    st.write("Diagnóstico Previsto: Nada a assinalar")
else:
    st.write("Diagnóstico Previsto:", prediction[0])

# Se o diagnóstico_data não estiver no estado da sessão, inicialize-o como um DataFrame vazio
# Registrar o diagnóstico
if 'diagnosis_data' not in st.session_state:
    st.session_state.diagnosis_data = pd.DataFrame(columns=['Name', 'Diagnosis', 'Model'])

if st.button('Registrar Diagnóstico'):
    diagnosis_entry = pd.DataFrame({'Name': [name], 'Diagnosis': [prediction[0]], 'Model': [selected_model]})
    st.session_state.diagnosis_data = pd.concat([st.session_state.diagnosis_data, diagnosis_entry], ignore_index=True)

# Exibir tabela com todos os diagnósticos registrados
st.write("### Registros de Diagnóstico")
st.write(st.session_state.diagnosis_data)

# Botão para resetar toda a tabela de diagnóstico
if st.button('Resetar Tabela de Diagnóstico'):
    # Resete toda a tabela de diagnóstico aqui
    st.session_state.diagnosis_data = pd.DataFrame(columns=['Name', 'Diagnosis', 'Model'])

if st.sidebar.button('Resetar Parâmetros'):
    # Resete os valores dos parâmetros e limpe os campos de entrada
    name = st.session_state.value = ""
