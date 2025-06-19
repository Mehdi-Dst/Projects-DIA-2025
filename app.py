# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from curl_cffi import requests  # Pour requêtes HTTP (utilisé par yfinance avec session personnalisée)
from keras.models import load_model
import streamlit as st  # Framework web interactif Python
from datetime import datetime, timedelta
import openai  # API OpenAI pour chatbot GPT
openai.api_key = ""  # Clé API OpenAI à renseigner


# --------------------------
# PART 1: STREAMLIT WEB APP
# --------------------------
st.title("Stock Price Prediction (Past & Future)")  # Titre principal de l'application web

# Sidebar pour entrer symboles boursiers et paramètres utilisateur
st.sidebar.header("User Input")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOLD, TSLA, PEPSI(PEP), PHIZER(pfe), SWBI):", "AAPL")  # Entrée du symbole boursier
future_days = st.sidebar.slider("Number of Days to Predict:", 1, 60, 30)  # Curseur pour choisir nb jours à prédire

# Message d'accueil et section chatbot dans la barre latérale
st.sidebar.markdown("---")
st.sidebar.header("Stock Market Chatbot 🤖")
st.sidebar.write("Hi! I’m your stock market assistant. Ask me anything about the stock or predictions!")

# Récupération des données boursières avec yfinance, via une session HTTP personnalisée
end_date = datetime.now().strftime('%Y-%m-%d')  # Date actuelle au format 'YYYY-MM-DD'
session = requests.Session(impersonate="chrome")  # Session avec user-agent Chrome
ticker = yf.Ticker(stock_symbol, session=session)  # Objet ticker yfinance
df = ticker.history(start="2010-01-01", end=end_date)  # Historique des données depuis 2010 jusqu'à aujourd'hui

# Vérification si données disponibles pour le symbole entré
if df.empty:
    st.error(f"No data found for stock symbol: {stock_symbol}. Please enter a valid symbol.")  # Message d'erreur utilisateur
    st.stop()  # Arrête l'exécution du script

# Fonction d'ajout d'indicateurs techniques dans le DataFrame
def add_technical_indicators(df):
    # Moyennes mobiles simples sur 100 et 200 jours
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calcul du RSI (Relative Strength Index) sur 14 jours
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calcul MACD (Moving Average Convergence Divergence) et sa ligne de signal
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna()  # Supprime lignes avec NaN résultant du calcul rolling

# Appliquer les indicateurs techniques
df = add_technical_indicators(df)

# Affichage d'un aperçu des données récentes dans la page principale
st.subheader(f"Historical Data (2010 - {end_date}) for {stock_symbol}")
st.write(df[['Close', 'MA100', 'MA200', 'RSI', 'MACD']].tail())

# Split des données en train/test (80% / 20%) en conservant un chevauchement de 100 jours pour les séquences
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size][['Close', 'MA100', 'RSI', 'MACD']]
test_data = df.iloc[train_size - 100:][['Close', 'MA100', 'RSI', 'MACD']]

# Normalisation des données (MinMaxScaler) : fit uniquement sur train, puis transform train et test
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

# Fonction pour créer des séquences glissantes pour LSTM (taille séquence par défaut 100)
def create_sequences(data, seq_length=100):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])  # Séquence d'entrée
        y.append(data[i, 0])  # Valeur cible = prix 'Close' au temps t
    return np.array(X), np.array(y)

# Préparation des séquences pour l'entraînement et le test
X_train, y_train = create_sequences(scaled_train)
X_test, y_test = create_sequences(scaled_test)

# Chargement du modèle pré-entrainé Keras (fichier .h5)
model = load_model('apple_predictor.h5', custom_objects={'mse': 'mean_squared_error'})


# --------------------------
# PART 2: FUTURE PREDICTIONS
# --------------------------
def predict_future(model, last_sequence, days=30):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Prédire le prochain prix sur la séquence courante (reshape pour batch 1)
        next_pred = model.predict(current_sequence.reshape(1, 100, 4))[0, 0]
        predictions.append(next_pred)
        
        # Mise à jour de la séquence: décaler à gauche, ajouter la nouvelle prédiction à la fin
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = next_pred  # Mettre à jour uniquement la colonne 'Close'
        
    # Compléter avec des zéros pour les autres colonnes
    predictions_padded = np.zeros((len(predictions), 4))
    predictions_padded[:, 0] = predictions
    
    # Inverse scaler pour revenir aux valeurs réelles de prix
    return scaler.inverse_transform(predictions_padded)[:, 0]

# Extraire la dernière séquence test pour lancer les prédictions futures
last_sequence = scaled_test[-100:]

# Calculer les prix futurs prédits selon le nombre de jours choisi
future_prices = predict_future(model, last_sequence, days=future_days)

# Générer les dates futures correspondantes
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days+1)]


# --------------------------
# PART 3: CHATBOT (OpenAI GPT)
# --------------------------

# Fonction d'interrogation du chatbot OpenAI GPT avec contexte financier
def ask_gpt(question, stock_symbol, last_price, future_price, future_days):
    context = f"""
    Tu es un assistant financier. L'utilisateur regarde l'action {stock_symbol}.
    Le dernier prix connu est de {last_price:.2f}$.
    La prédiction pour dans {future_days} jours est de {future_price:.2f}$.
    Réponds de façon claire, concise et professionnelle à la question : {question}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # ou "gpt-3.5-turbo" pour moins cher
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ],
            temperature=0.7  # Température modérée pour équilibre entre créativité et précision
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Une erreur s'est produite : {e}"

# Interface utilisateur pour le chatbot
user_input = st.sidebar.text_input("Pose-moi une question sur l'action ou la prédiction :")
if user_input:
    response = ask_gpt(user_input, stock_symbol, df['Close'].iloc[-1], future_prices[-1], future_days)
    st.sidebar.write(f"**Chatbot GPT:** {response}")

# Suggestion proactive en fonction de la tendance prédite
if future_prices[-1] > df['Close'].iloc[-1].item():
    st.sidebar.write(f"**Chatbot:** The stock is predicted to rise by {abs(future_prices[-1] - df['Close'].iloc[-1].item()):.2f}% in the next {future_days} days. 📈 Would you like to know more about why this is a good time to buy?")
else:
    st.sidebar.write(f"**Chatbot:** The stock is predicted to drop by {abs(future_prices[-1] - df['Close'].iloc[-1].item()):.2f}% in the next {future_days} days. 📉 Would you like to know more about why this might be a good time to sell?")


# --------------------------
# PART 4: VISUALIZATION
# --------------------------

# Création de 3 onglets pour afficher différentes visualisations
tab1, tab2, tab3 = st.tabs(["Price Charts", "Technical Indicators", "Predictions"])

# Onglet 1 : Graphiques des prix historiques
with tab1:
    st.subheader("Original Price")
    fig1, ax1 = plt.subplots(figsize=(14,7))
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.set_title(f"{stock_symbol} Original Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    st.pyplot(fig1)

# Onglet 1 (suite) : Prix avec moyenne mobile 100 jours
with tab1:
    st.subheader("Price with Moving Average (MA100)")
    fig2, ax2 = plt.subplots(figsize=(14,7))
    ax2.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax2.plot(df.index, df['MA100'], label='MA100', color='orange', linestyle='--')
    ax2.set_title(f"{stock_symbol} Price with MA100")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    st.pyplot(fig2)

# Onglet 1 (suite) : Prix avec moyennes mobiles 100 et 200 jours
with tab1:
    st.subheader("Price with Moving Averages (MA100 and MA200)")
    fig3, ax3 = plt.subplots(figsize=(14,7))
    ax3.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax3.plot(df.index, df['MA100'], label='MA100', color='orange', linestyle='--')
    ax3.plot(df.index, df['MA200'], label='MA200', color='green', linestyle='--')
    ax3.set_title(f"{stock_symbol} Price with MA100 and MA200")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price ($)")
    ax3.legend()
    st.pyplot(fig3)

# Onglet 2 : Indicateurs techniques RSI
with tab2:
    st.subheader("RSI (Relative Strength Index)")
    fig4, ax4 = plt.subplots(figsize=(14,7))
    ax4.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax4.axhline(70, color='red', linestyle='--', label='Overbought (70)')  # Zone surachetée
    ax4.axhline(30, color='green', linestyle='--', label='Oversold (30)')   # Zone survendue
    ax4.set_title(f"{stock_symbol} RSI")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("RSI")
    ax4.legend()
    st.pyplot(fig4)

# Onglet 2 : Indicateur MACD
with tab2:
    st.subheader("MACD (Moving Average Convergence Divergence)")
    fig5, ax5 = plt.subplots(figsize=(14,7))
    ax5.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax5.plot(df.index, df['Signal_Line'], label='Signal Line', color='orange', linestyle='--')
    ax5.set_title(f"{stock_symbol} MACD")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("MACD")
    ax5.legend()
    st.pyplot(fig5)

# Onglet 3 : Prix historiques + prédictions futures
with tab3:
    st.subheader("Historical and Predicted Prices")

    # Préparer prix historiques réels (inverse scaler avec padding)
    scaled_test_padded = np.zeros((len(scaled_test), 4))
    scaled_test_padded[:, 0] = scaled_test[:, 0]  # Valeurs 'Close' réelles

    actual_prices = scaler.inverse_transform(scaled_test_padded)[:, 0]

    # Index ajusté pour l'affichage (avec chevauchement 100 jours)
    adjusted_index = df.index[train_size - 100:]

    # Tracer les prix historiques et prédictions futures
    fig6, ax6 = plt.subplots(figsize=(14,7))
    ax6.plot(adjusted_index, actual_prices, 'b-', label="Historical Prices")
    ax6.plot(future_dates, future_prices, 'r--', label="Future Predictions")
    ax6.set_title(f"{stock_symbol} Stock Price Forecast")
    ax6.set_xlabel("Date")
    ax6.set_ylabel("Price ($)")
    ax6.legend()
    st.pyplot(fig6)

    # Tableau des prix futurs prédits
    st.subheader(f"Next {future_days}-Day Forecast for {stock_symbol}")
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": future_prices
    })
    st.write(future_df)

# Footer simple avec crédits
st.markdown("---")
st.markdown("**Final Year Project** | Created by Mehdi OUAFSSOU | [GitHub Repo](#)")
