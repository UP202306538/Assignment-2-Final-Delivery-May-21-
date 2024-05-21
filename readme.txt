**Descrição**
-> Este projeto possui a aplicação do Streamlit que permite prever a possibilidade de o individuo morrer/sobreviver de acordo com o seu diagnóstico de hepatocarcinoma, utilizando os dois modelos de machine learning: Decision Tree e K-Nearest Neighbors (KNN). 
A aplicação permite que se insire os parâmetros do paciente, se selecione um modelo para previsão, e registrar os diagnósticos em uma tabela exibida na interface.

**Como Rodar o Programa**
Pré-requisitos: Python 3.8 ou superior instalado no sistema bem como as bibliotecas streamlit, pandas, numpy, scikit-learn(sklearn) e joblib (através do comando pip install).

Antes de executar a Aplicação:
	1. Descarregue a database pretendida (hcc_dataset.csv), bem como, os scrpits numa pasta de fácil acesso todos juntos. Ex: C:\Users\user****\Downloads\projeto2
	2. Abre o script proj2_dados.py num software Python IDE, como o Pyzo, e nas seguintes linhas (15, 98 e 99) coloque a path onde os ficheiros estão e serão guardados
		Ex:
		Linha 15 - data = pd.read_csv('C:\\Users\\user****\\Downloads\\projeto2\\hcc_dataset.csv') 
		Linha 63 - joblib.dump(preprocessor, 'C:\\Users\\user****\\Downloads\\projeto2\\preprocessor.pkl')
		Linha 98 - joblib.dump(best_dt, 'C:\\Users\\user****\\Downloads\\projeto2\\best_dt_model.pkl')
		Linha 99 - joblib.dump(best_knn, 'C:\\Users\\user****\\Downloads\\projeto2\\best_knn_model.pkl')
	3. Execute o scrpit para que guarde os ficheiros para posteriormente serem usados para o Website em Streamlit
	4. Abre o script proj2_site.py num software Python IDE, como o Pyzo, e nas seguintes linhas (6, 9 e 12) coloque a path onde os ficheiros estão guardados
		Ex:
		Linha 6 - preprocessor = joblib.load('C:\\Users\\user****\\Downloads\\projeto2\\preprocessor.pkl')
		Linha 9 - best_dt = joblib.load('C:\\Users\\user****\\Downloads\\projeto2\\best_dt_model.pkl')
		Linha 12 - best_knn = joblib.load('C:\\Users\\user****\\Downloads\\projeto2\\best_knn_model.pkl')

Executar a Aplicação:
	1. Abra um terminal ou prompt de comando e navegue até o diretório onde está todo o projeto.
	2. Execute o seguinte comando para iniciar a aplicação Streamlit: streamlit run projeto2_site.py
	3. Uma nova aba do navegador será aberta com a aplicação Streamlit em execução.

Como Usar a Aplicação:

1. **Inserir Dados do Paciente:**
   - Preencha os parâmetros do paciente no painel lateral esquerdo. 
   - Todos os campos são opcionais, mas quanto mais dados você fornecer, mais precisa será a previsão.

2. **Selecionar o Modelo de Previsão:**
   - Escolha entre os modelos "Decision Tree" ou "KNN" no painel lateral esquerdo.

3. **Fazer a Previsão:**
   - Clique no botão "Fazer Previsão" para gerar o diagnóstico com base nos parâmetros inseridos e no modelo selecionado.

4. **Registrar o Diagnóstico:**
   - Clique no botão "Registrar Diagnóstico" para adicionar o resultado do diagnóstico à tabela de registros.

5. **Resetar Parâmetros:**
   - Clique no botão "Resetar Parâmetros" para limpar todos os campos de entrada.

6. **Reiniciar Tabela:**
   - Clique no botão "Reiniciar Tabela" para limpar todos os registros da tabela de diagnósticos.

## Estrutura do Projeto
- `projeto2_1analise.py`: Código da primeira Análise de Dados e Treinamento de Modelos.
- `projeto2_2analise.py`: Código da segunda Análise de Dados e Treinamento de Modelos.
- `projeto2_suportesite.py`: Código de Análise de Dados e Treinamento de Modelos usado para o Website.
- `projeto2_site.py`: Código principal da aplicação Streamlit.
- `hcc_dataset.csv`: Dataset utilizado para treinar os modelos de machine learning.
- `preprocessor.pkl`: Arquivo salvo contendo o pipeline de pré-processamento ajustado.
- `best_dt_model.pkl`: Arquivo salvo contendo o modelo treinado de Decision Tree.
- `best_knn_model.pkl`: Arquivo salvo contendo o modelo treinado de KNN.

## Notas
- Certifique-se de que o arquivo `hcc_dataset.csv` esteja presente no diretório especificado em ambos scripts `proj2_dados.py` e `proj2_site.py`.
- Os arquivos `preprocessor.pkl`, `best_dt_model.pkl` e `best_knn_model.pkl` devem ser gerados previamente através do script de análise de dados e treinamento de modelos (não incluído neste arquivo).

## Contato
Para qualquer dúvida entre em contato através do email: 
João Carneiro up202306538@up.pt
Hugo Lameira up202306672@up.pt

