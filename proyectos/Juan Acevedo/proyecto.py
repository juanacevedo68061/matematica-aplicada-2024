import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time
import os

# Obtener la ruta del directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el dataset desde el archivo CSV usando ruta relativa
file_path = os.path.join(script_dir, 'archive', 'test_data.csv')

df = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', 0)  # Ajuste automático del ancho según el contenido

# Inicializar el analizador de sentimientos de VADER
sia = SentimentIntensityAnalyzer()

# Función para calcular las puntuaciones
def calcular_puntuaciones(tweet):
    TweetPos = 0  # Inicializar la puntuación positiva
    TweetNeg = 0  # Inicializar la puntuación negativa
    
    # Dividir el tweet en palabras
    palabras = tweet.split()  
    
    for palabra in palabras:
        # Obtener las puntuaciones de VADER para cada palabra
        puntuaciones = sia.polarity_scores(palabra)
        μ_Pos = puntuaciones['pos']  # Puntuación positiva
        μ_Neg = puntuaciones['neg']  # Puntuación negativa
        
        # Sumar solo los valores positivos y negativos retornados
        if μ_Pos > 0:
            TweetPos += μ_Pos  # Sumar a TweetPos si es positivo
        if μ_Neg > 0:
            TweetNeg += μ_Neg  # Sumar a TweetNeg si es negativo
    
    return TweetPos, TweetNeg

# Aplicar la función a cada tweet en el DataFrame
df[['TweetPos', 'TweetNeg']] = df['sentence'].apply(calcular_puntuaciones).apply(pd.Series)

# Seccion 3.3.1 en adelante

# Calcular los valores mínimos y máximos globales
min_pos = df['TweetPos'].min()  # (14)
max_pos = df['TweetPos'].max()  
min_neg = df['TweetNeg'].min()  
max_neg = df['TweetNeg'].max()  

# Calcular el valor medio
mid_pos = (min_pos + max_pos) / 2  # (14)
mid_neg = (min_neg + max_neg) / 2  

# Crear las variables de entrada
tweet_pos = ctrl.Antecedent(np.arange(min_pos, max_pos + 0.1, 0.1), 'TweetPos')  
tweet_neg = ctrl.Antecedent(np.arange(min_neg, max_neg + 0.1, 0.1), 'TweetNeg')  

# Crear la variable de salida
output = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'SentimentScore')  # Valores van de 0 a 10 a paso de 0.1

# Definir las funciones de pertenencia para TweetPos
tweet_pos['Bajo'] = fuzz.trimf(tweet_pos.universe, [min_pos, min_pos, mid_pos])  # Bajo (13)
tweet_pos['Medio'] = fuzz.trimf(tweet_pos.universe, [min_pos, mid_pos, max_pos])  # Medio (13)
tweet_pos['Alto'] = fuzz.trimf(tweet_pos.universe, [mid_pos, max_pos, max_pos])  # Alto (13)

# Definir las funciones de pertenencia para TweetNeg
tweet_neg['Bajo'] = fuzz.trimf(tweet_neg.universe, [min_neg, min_neg, mid_neg])  # Bajo (13)
tweet_neg['Medio'] = fuzz.trimf(tweet_neg.universe, [min_neg, mid_neg, max_neg])  # Medio (13)
tweet_neg['Alto'] = fuzz.trimf(tweet_neg.universe, [mid_neg, max_neg, max_neg])  # Alto (13)

# Definir las funciones de pertenencia para la salida
output['Negativo'] = fuzz.trimf(output.universe, [0, 0, 5])  # Negativo (13)
output['Neutral'] = fuzz.trimf(output.universe, [0, 5, 10])  # Neutral (13)
output['Positivo'] = fuzz.trimf(output.universe, [5, 10, 10])  # Positivo (13)

# Definir las reglas basadas en la tabla proporcionada (Figura 4 y Tabla 2)
rules = []
rules.append(ctrl.Rule(tweet_pos['Bajo'] & tweet_neg['Bajo'], output['Neutral']))  # WR1 (15)
rules.append(ctrl.Rule(tweet_pos['Medio'] & tweet_neg['Bajo'], output['Positivo']))  # WR2 (16)
rules.append(ctrl.Rule(tweet_pos['Alto'] & tweet_neg['Bajo'], output['Positivo']))  # WR3 (17)
rules.append(ctrl.Rule(tweet_pos['Bajo'] & tweet_neg['Medio'], output['Negativo']))  # WR4 (18)
rules.append(ctrl.Rule(tweet_pos['Medio'] & tweet_neg['Medio'], output['Neutral']))  # WR5 (19)
rules.append(ctrl.Rule(tweet_pos['Alto'] & tweet_neg['Medio'], output['Positivo']))  # WR6 (20)
rules.append(ctrl.Rule(tweet_pos['Bajo'] & tweet_neg['Alto'], output['Negativo']))  # WR7 (21)
rules.append(ctrl.Rule(tweet_pos['Medio'] & tweet_neg['Alto'], output['Negativo']))  # WR8 (22)
rules.append(ctrl.Rule(tweet_pos['Alto'] & tweet_neg['Alto'], output['Neutral']))  # WR9 (23)

# Generancion de salidas y Benchmarks

# Crear el sistema de control
sentiment_ctrl = ctrl.ControlSystem(rules)  
sentiment_sim = ctrl.ControlSystemSimulation(sentiment_ctrl)

# Calcular el puntaje de sentimiento para cada tweet
sentiment_scores = []
execution_times = []
resultados = []
for index, row in df.iterrows():
    
    start_time = time.time()  # Iniciar temporizador

    sentiment_sim.input['TweetPos'] = row['TweetPos']  
    sentiment_sim.input['TweetNeg'] = row['TweetNeg']  
    sentiment_sim.compute()  # (24) al (31) Esto computa el puntaje de salida basado en las reglas
    z = sentiment_sim.output['SentimentScore']
    sentiment_scores.append(z) # Almacenar el puntaje de sentimiento

    execution_time = time.time() - start_time  # Calcular tiempo de ejecución
    execution_times.append(execution_time)  # Almacenar tiempo de ejecución

    # Almacenar resultados para el CSV nuevo
    resultados.append({
        'Oración Original': row['sentence'],
        'Label Original': row['sentiment'],
        'Puntaje Positivo': row['TweetPos'],
        'Puntaje Negativo': row['TweetNeg'],
        'Resultado de Inferencia': z, 
        'Tiempo de Ejecución': execution_time
    })

# Agregar los puntajes de sentimiento al DataFrame
df['SentimentScore'] = sentiment_scores

# Método de desfuzzificación del centroide
def classify_sentiment(score):
    if 0 < score < 3.3:  # (32)
        return 'Negativo'
    elif 3.3 < score < 6.7:  # (32)
        return 'Neutral'
    elif 6.7 < score < 10:  # (32)
        return 'Positivo'

# Agregar la clasificación al DataFrame
df['SentimentSet'] = df['SentimentScore'].apply(classify_sentiment)

# Crear un DataFrame con los resultados
benchmarks_df = pd.DataFrame(resultados)

# Guardar el DataFrame en un nuevo archivo CSV usando ruta relativa
output_file_path = os.path.join(script_dir, 'archive', 'benchmarks.csv')
benchmarks_df.to_csv(output_file_path, index=False)

# Contar la cantidad de positivos, negativos y neutrales
count_positive = (df['SentimentSet'] == 'Positivo').sum()
count_negative = (df['SentimentSet'] == 'Negativo').sum()
count_neutral = (df['SentimentSet'] == 'Neutral').sum()

# Calcular tiempo promedio Total
total_tweets = len(df)
average_execution_time = sum(execution_times) / total_tweets

# Mostrar resultados en la consola
print(benchmarks_df)
print(f"\nCantidad de tweets positivos: {count_positive}")
print(f"Cantidad de tweets negativos: {count_negative}")
print(f"Cantidad de tweets neutrales: {count_neutral}")
print(f'Tiempo de Ejecución Promedio Total: {average_execution_time:.4f} segundos')
