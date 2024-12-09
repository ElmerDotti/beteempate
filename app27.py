import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Configurar Streamlit
st.title("Predição de Empates com XGBoost")
st.write("""
Este aplicativo utiliza XGBoost para predizer empates em partidas de futebol, 
considerando odds superiores a 2.5. O treinamento utiliza dados históricos de julho a outubro de 2024, 
e o backtest utiliza os dados de novembro de 2024.
""")

# Função de login
def login():
    st.subheader("Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if username == "betempate1" and password == "betempate1":
            st.success("Login realizado com sucesso!")
            return True
        else:
            st.error("Usuário ou senha incorretos.")
    return False

if not login():
    st.stop()

# Função para carregar dados diretamente do Google Drive via link
@st.cache_data
def load_data_from_drive_via_chunks(drive_url, start_date, end_date, date_column="datetime", chunksize=10**6):
    """
    Carrega dados diretamente do Google Drive em chunks e filtra por período.

    Args:
        drive_url (str): URL de compartilhamento do Google Drive (ajustado para uso com pandas).
        start_date (str): Data inicial do intervalo no formato "YYYY-MM-DD".
        end_date (str): Data final do intervalo no formato "YYYY-MM-DD".
        date_column (str): Nome da coluna que contém datas.
        chunksize (int): Tamanho do chunk para leitura eficiente.

    Returns:
        pd.DataFrame: DataFrame contendo os dados filtrados.
    """
    try:
        # Ajustar o link para compatibilidade com pandas
        adjusted_url = drive_url.replace("view?usp=sharing", "export?format=csv")
        
        # Lista para armazenar os chunks filtrados
        filtered_chunks = []

        # Ler os dados em chunks e filtrar pelo intervalo de datas
        for chunk in pd.read_csv(adjusted_url, chunksize=chunksize, parse_dates=[date_column]):
            filtered_chunk = chunk[(chunk[date_column] >= start_date) & (chunk[date_column] <= end_date)]
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
            print(f"Chunk processado: {len(filtered_chunk)} linhas filtradas.")

        # Concatenar os chunks em um único DataFrame
        if filtered_chunks:
            return pd.concat(filtered_chunks, ignore_index=True)
        else:
            return pd.DataFrame()  # Retorna vazio se nenhum dado for encontrado
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

# URL do arquivo no Google Drive
google_drive_url = "https://drive.google.com/file/d/1gIUl9UphsgeYK_QWjr6QY9ac0mXWL3Kp/view?usp=sharing"

# Carregar dados para treinamento, teste e validação (julho a outubro de 2024)
training_data = load_data_from_drive_via_chunks(
    drive_url=google_drive_url,
    start_date="2024-07-01",
    end_date="2024-10-31",
    date_column="datetime"
)

# Carregar dados para backtest (novembro de 2024)
backtest_data = load_data_from_drive_via_chunks(
    drive_url=google_drive_url,
    start_date="2024-11-01",
    end_date="2024-11-30",
    date_column="datetime"
)

# Verificar se os dados foram carregados corretamente
if training_data.empty:
    st.error("Os dados para treinamento, teste e validação (julho a outubro de 2024) não foram encontrados.")
if backtest_data.empty:
    st.error("Os dados para backtest (novembro de 2024) não foram encontrados.")

if not training_data.empty and not backtest_data.empty:
    st.write("Dados carregados com sucesso! Pré-visualização dos dados de treinamento:")
    st.dataframe(training_data.head())

    st.write("Pré-visualização dos dados de backtest:")
    st.dataframe(backtest_data.head())

    # Colunas numéricas que serão utilizadas
    numeric_cols = ["kHomeWinOdd", "kDrawOdd", "kAwayWinOdd", "kTotals", 
                    "kOverOdd", "kUnderOdd", "kHandicapHome", "kHandicapHomeOdd", 
                    "kHandicapAway", "kHandicapAwayOdd"]

    # Função para calcular colunas lag
    def calculate_lags(dataframe, columns, max_lag):
        for col in columns:
            for lag in range(1, max_lag + 1):
                lag_col_name = f"{col}_lag_{lag}"
                dataframe[lag_col_name] = dataframe[col].shift(lag)
        return dataframe

    # Calcular lags nos dados de treinamento
    training_data = calculate_lags(training_data, numeric_cols, max_lag=3)

    # Garantir que não há valores ausentes nos dados de treinamento
    training_data.interpolate(method='linear', inplace=True)
    training_data.dropna(inplace=True)

    # Criar coluna de empate
    training_data['is_draw'] = (training_data['homeScore'] == training_data['awayScore']).astype(int)

    # Filtrar apenas jogos com odds superiores a 2.5
    training_data = training_data[training_data['kDrawOdd'] >= 2.5]

    # Dividir dados em entradas e saídas
    X_train = training_data[numeric_cols + [f'{col}_lag_{lag}' for col in numeric_cols for lag in range(1, 4)]]
    y_train = training_data['is_draw']

    # Escalar os dados
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Dividir para treino e validação
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # Definir hiperparâmetros para GridSearch
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    # Configurar o modelo base
    xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

    # Realizar GridSearch
    st.write("Executando GridSearch para otimizar hiperparâmetros...")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_final, y_train_final)

    best_model = grid_search.best_estimator_
    st.write("Hiperparâmetros ótimos encontrados:")
    st.write(grid_search.best_params_)

    # Preparar dados de backtest
    backtest_data = calculate_lags(backtest_data, numeric_cols, max_lag=3)
    backtest_data.dropna(inplace=True)

    # Adicionar a coluna 'is_draw' no conjunto de dados de backtest
    backtest_data['is_draw'] = (backtest_data['homeScore'] == backtest_data['awayScore']).astype(int)

    X_test = backtest_data[numeric_cols + [f'{col}_lag_{lag}' for col in numeric_cols for lag in range(1, 4)]]
    y_test = backtest_data['is_draw']

    X_test_scaled = scaler.transform(X_test)

    # Avaliar o modelo no backtest
    st.write("Realizando predições no conjunto de backtest...")
    test_preds = best_model.predict(X_test_scaled)
    test_preds_rounded = np.round(test_preds).astype(int)

    # Gerar matriz de resultados
    backtest_data['predicted_is_draw'] = test_preds_rounded
    backtest_data['bet_result'] = (backtest_data['predicted_is_draw'] == backtest_data['is_draw']).astype(int)
    bet_value = 3000  # Valor da aposta em reais
    backtest_data['profit'] = np.where(
        backtest_data['bet_result'] == 1,
        (bet_value * backtest_data['kDrawOdd']) - bet_value,  # Lucro ajustado conforme solicitado
        -bet_value  # Perda quando erra
    )

    # Calcular acurácia
    accuracy = accuracy_score(y_test, test_preds_rounded)
    st.write(f"Acurácia no backtest: {accuracy:.2f}")

    # Mostrar matriz de resultados
    st.write("Matriz de resultados do backtest:")
    st.dataframe(backtest_data[['datetime', 'kDrawOdd', 'is_draw', 'predicted_is_draw', 'profit']])

    # Cálculo de indicadores adicionais
    capital_inicial = 200000
    st.write(f"Capital Inicial: R${capital_inicial:,.2f}")

    # Plotar gráfico de retorno financeiro
    backtest_data['cumulative_profit'] = backtest_data['profit'].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(backtest_data['datetime'], backtest_data['cumulative_profit'], marker='o')
    plt.title('Retorno Financeiro Acumulado')
    plt.xlabel('Data')
    plt.ylabel('Retorno Acumulado (R$)')
    plt.grid()
    st.pyplot(plt)
