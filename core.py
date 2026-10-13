import pandas as pd
data = pd.read_csv("processed_dataset.csv", sep=",", engine="python")

'''Ejercicio 1'''

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import contractions
import emoji

# Descargamos los recursos necesarios de NLTK
nltk.download("punkt", quiet=True)   # Tokenización
nltk.download("stopwords", quiet=True)   # Stopwords en inglés

# Definimos las expresión regulares
url_pattern = re.compile(r"http[s]?://\S+")
html_pattern = re.compile(r"<.*?>")

# Creamos un conjunto de stopwords en inglés para filtrar palabras comunes
stop_words = set(stopwords.words("english"))


# Función para convertir emojis a texto descriptivo
def convert_emojis_to_text(text):
    return emoji.demojize(text)


def preprocess_post(text: str):
    # Paso 1: Convertir todo el texto a minúsculas y resolver contracciones (como "I'm" a "I am")
    text = contractions.fix(text.lower())

    # Paso 2: Eliminar signos de puntuación y espacios extra
    text = text.translate(str.maketrans("", "", string.punctuation)).strip()

    # Paso 3: Eliminar URLs
    text = url_pattern.sub("", text)
    
    # Paso 4: Eliminar etiquetas HTML
    text = html_pattern.sub("", text)

    # Paso 5: Convertir los emojis en el texto a su representación textual (ej. ":smile:" en lugar de "😊")
    text = convert_emojis_to_text(text)

    # Paso 6: Tokenizar el texto, eliminar palabras que no sean alfabéticas y que sean stopwords
    sin_stopwords = [word for word in word_tokenize(text) if word.isalpha() and word not in stop_words]

    # Paso 7: Unir las palabras procesadas en un único string, separado por espacios
    clean_text = " ".join(sin_stopwords)

    return clean_text   # Devolver el texto limpio y preprocesado




'''Ejercicio 2'''

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

data = data.dropna(subset=["clean_post", "subreddit"]).reset_index(drop=True)

# 1. División de datos en entrenamiento y prueba (70% - 30%)
X_train, X_test, y_train, y_test = train_test_split(data["clean_post"], data["subreddit"], test_size=0.3, random_state=42)

# 2. Vectorización TF-IDF
# Convierte los textos en matrices numéricas donde cada valor representa la importancia relativa de cada término en el documento
vectorizer_tfidf = TfidfVectorizer(max_df=0.95, min_df=5)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# 3. Codificar las etiquetas
# Convierte las etiquetas categóricas en valores numéricos.
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 4. Entrenamiento del modelo
# Se utiliza un clasificador Random Forest, que combina múltiples árboles de decisión.
classifier = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1, bootstrap=True, max_depth=50, min_samples_leaf=1, min_samples_split=2, max_features="sqrt")
classifier.fit(X_train_tfidf, y_train_encoded)

# 5. Predicción y evaluación
# y_pred = classifier.predict(X_test_tfidf)
# f1 = f1_score(y_test_encoded, y_pred, average="weighted")
# print(f"F1 Score: {f1}")
# print(classification_report(y_test_encoded, y_pred, zero_division=0))


# # Obtener la importancia de las características del clasificador
# feature_importances = classifier.feature_importances_

# # Obtener los nombres de las características
# feature_names = vectorizer_tfidf.get_feature_names_out()

# # Ordenar las características por importancia
# sorted_indices = np.argsort(feature_importances)[::-1]

# # Seleccionar las 20 características más importantes
# top_n = 20
# top_features = [feature_names[i] for i in sorted_indices[:top_n]]
# top_importances = feature_importances[sorted_indices[:top_n]]

# # Graficar la importancia de las características
# plt.figure(figsize=(10, 8))
# plt.barh(top_features, top_importances, color="lightblue", align="center", edgecolor="blue")
# plt.xlabel("Importancia de la característica")
# plt.title("Top 20 características más importantes")
# plt.gca().invert_yaxis()
# plt.show()


def classify_subreddit(text):
    '''Función para clasificar un texto de entrada en un subreddit predefinido.'''

    # 1. Preprocesar y vectorizar el texto
    # Convierte el texto de entrada en una representación numérica utilizando
    # el vectorizador TF-IDF ajustado durante el entrenamiento.
    text_vectorized = vectorizer_tfidf.transform([text])
    
    # 2. Realizar predicción
    # Utiliza el modelo de clasificación entrenado para predecir la etiqueta del texto.
    prediction = classifier.predict(text_vectorized)
    
    # 3. Decodificar la etiqueta predicha
    # Convierte la predicción numérica de vuelta a la etiqueta categórica original.
    subreddit = label_encoder.inverse_transform(prediction)

    return subreddit[0]




'''Ejercicio 3'''

### Función 1
reddit_exp = re.compile(r"\/r\/[a-z]+")

def find_subreddit_mentions(text: str) -> list:
    """Encuentra y devuelve todas las menciones de subreddits en el texto dado."""
    
    subreddit = reddit_exp.findall(text)
    return subreddit


### Función 2
url_exp = re.compile(r"http[s]?\:\/\/\S+")

def url_extraction(text: str) -> list:
    """Extrae todas las URLs del texto dado."""
    
    urls = url_exp.findall(text)
    return urls


### Función 3
phone_exp = re.compile(r"\+?[0-9]{9,10}|\([0-9]{3}\)\s?[0-9]{3}\-[0-9]{3,4}|[0-9]{3}[\-\s\_][0-9]{3}[\-\s\_][0-9]{3,4}")

def phone_number_extraction(text: str) -> list:
    """Extrae todos los números de teléfono del texto dado."""

    phones = phone_exp.findall(text)
    return phones


### Función 4
import spacy
import dateparser

def extract_dates(text):
    """Función para extraer y analizar fechas de un texto dado utilizando spaCy y dateparser.

    - return: Una lista de tuplas con la fecha original y la fecha procesada.
    """
    
    # Cargar el modelo de lenguaje de spaCy para inglés
    nlp = spacy.load("en_core_web_sm")
    
    # Procesar el texto con el modelo de spaCy para extraer las entidades
    doc = nlp(text)
    
    dates = []
    
    # Iterar sobre las entidades identificadas por spaCy
    for ent in doc.ents:
        # Comprobar si la entidad es una fecha o una hora
        if ent.label_ == "DATE" or ent.label_ == "TIME":
            # Usar dateparser para analizar la fecha y convertirla a un formato estandarizado
            date = dateparser.parse(ent.text, settings={"DATE_ORDER": "DMY"})
            # Añadir la fecha original y la fecha procesada a la lista
            dates.append((ent.text, date))
    
    return dates


### Función 5
html_exp = re.compile(r"<.*?>")   # Detectar etiquetas HTML individuales
html_snippet_exp = re.compile(r"^\s*<.*>.*</.*>\s*$")   # Detectar fragmentos HTML completos

# Expresiones regulares para detectar sintaxis Python
python_keywords = re.compile(r"\b(def|class|import|from|return|if|elif|else|for|while|try|except|with|as|lambda|print)\b")
python_structures = re.compile(r"[{}:()=\[\]]")   # Caracteres comunes en Python
python_comments = re.compile(r"^\s*#")   # Comentarios en Python


csvs = ["stackoverflow_code_dataset.csv", "github_code_dataset.csv", "github_code_dataset_2.csv"]
new_data = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(new_data["text"])
y = new_data["label"]

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.3, random_state=42)

classifier_2 = RandomForestClassifier(random_state=42, n_estimators=1000, n_jobs=-1, bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features="sqrt")
classifier_2.fit(X_train_2, y_train_2)



def is_python_code(line):
    """Detecta si una línea tiene características de código Python."""

    if python_keywords.search(line) or python_structures.search(line) or python_comments.search(line):
        return True
    return False


def predict_code(linea):
    """Función de predicción que clasifica si una línea es código o no utilizando el modelo entrenado."""
    
    text_features = vectorizer.transform([linea])
    prediction = classifier_2.predict(text_features)

    if prediction[0] == "code" and is_python_code(linea):   # Confirmar que sea código Python
        return "code"
    return "non-code"


def code_extraction(texto: str) -> dict:
    """
    Extrae etiquetas HTML, fragmentos HTML completos y fragmentos de código de un texto.
    
    - texto: El texto a analizar.
    - return: Diccionario con 'etiquetas_html', 'fragmentos_html' y 'fragmentos_codigo'.
    """
    fragmentos_codigo = []
    fragmentos_html = []
    etiquetas_html = []

    # Procesar el texto línea por línea
    for linea in texto.splitlines():
        linea = linea.strip()
        if linea:   # Ignorar líneas vacías
            # Extraer etiquetas HTML individuales
            etiquetas = html_exp.findall(linea)
            if etiquetas:
                etiquetas_html.extend(etiquetas)
            
            # Detectar fragmentos HTML completos
            if html_snippet_exp.match(linea):
                fragmentos_html.append(linea)
            else:
                # Identificar si la línea es código
                if predict_code(linea) == "code":
                    fragmentos_codigo.append(linea)

    # Se devuelven los resultados en un diccionario
    return {
        "etiquetas_html": etiquetas_html if etiquetas_html else None,
        "fragmentos_html": fragmentos_html if fragmentos_html else None,
        "fragmentos_codigo": fragmentos_codigo if fragmentos_codigo else None
    }




'''Ejercicio 4'''

from nltk.tokenize import word_tokenize

# Crear vocabularios únicos
positive_vocabulary = set()   # Conjunto vacío para las palabras positivas
negative_vocabulary = set()   # Conjunto vacío para las palabras negativas

# Iteramos sobre las reseñas positivas
for review in data[data["sentiment"] == "POSITIVE"]["clean_post"]:
    words = set(review.lower().split())   # Convertimos a minúsculas y dividimos en palabras
    positive_vocabulary.update(words)   # Actualizamos el vocabulario positivo

# Iteramos sobre las reseñas negativas
for review in data[data["sentiment"] == "NEGATIVE"]["clean_post"]:
    words = set(review.lower().split())   
    negative_vocabulary.update(words)   # Actualizamos el vocabulario negativo


# Clasificación basada en vocabularios únicos
def unique_word_based_sentiment(text):
    try:
        tokens = set(word_tokenize(text.lower()))   # Convertimos el texto a minúsculas y tokenizamos
        
        # Calculamos coincidencias
        pos_matches = len(tokens.intersection(positive_vocabulary))
        neg_matches = len(tokens.intersection(negative_vocabulary))
        
        # Devolvemos la etiqueta en el formato correcto
        return "POSITIVE" if pos_matches > neg_matches else "NEGATIVE"

    except Exception as e:
        print(f"Error de procesamiento: {e}")
        return "NEGATIVE"   # Default return value


# Función principal para análisis de sentimiento
def sentiment_analysis(text: str) -> str:
    """
    Realiza análisis de sentimiento basado en vocabularios únicos.

    Args:
        text (str): Texto para analizar.

    Returns:
        str: Predicción del sentimiento ("POSITIVE" o "NEGATIVE").
    """
    return unique_word_based_sentiment(text)




'''Ejercicio 5'''

import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Definir stop words en español
stop_words = stopwords.words("spanish")

def post_summarisation(text: str, ngram_range=(1, 2), summary_length=3) -> str:
    """
    Genera un resumen del texto dado usando n-gramas y puntuación basada en CountVectorizer y TfidfVectorizer.
    
    Args:
        text (str): El texto de entrada para resumir.
        ngram_range (tuple): Rango de n-gramas a usar, por ejemplo, (1, 1) para unigramas o (1, 2) para bigramas.
        summary_length (int): Número de oraciones en el resumen.

    Returns:
        str: Resumen generado del texto.
    """

    # Tokenizar el texto en oraciones
    sentences = sent_tokenize(text)
    
    # Crear los vectorizadores para n-gramas usando stopwords
    count_vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    tfidf_vec = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    
    try:
        # Ajustar y transformar el texto
        count_matrix = count_vec.fit_transform(sentences)
        tfidf_matrix = tfidf_vec.fit_transform(sentences)
        
        # Calcular las puntuaciones de las oraciones
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            count_score = np.sum(count_matrix[i].toarray())
            tfidf_score = np.sum(tfidf_matrix[i].toarray())
            
            # Diferentes ponderaciones según el tipo de n-grama
            if ngram_range[1] == 1:   # Unigramas
                sentence_scores[sentence] = 0.7 * count_score + 0.3 * tfidf_score
            elif ngram_range[1] == 2:   # Bigramas
                sentence_scores[sentence] = 0.4 * count_score + 0.6 * tfidf_score
            else:   # Trigramas
                sentence_scores[sentence] = 0.2 * count_score + 0.8 * tfidf_score
            
            # Añadir factor de posición
            position_weight = 1 / (i + 1)
            sentence_scores[sentence] *= (0.8 + 0.2 * position_weight)
        
        # Seleccionar las mejores oraciones
        summary_sentences = sorted(sentence_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:summary_length]
        
        # Mantener el orden original de las oraciones
        ordered_summary = [sent for sent in sentences 
                         if sent in [s[0] for s in summary_sentences]]
        
        return " ".join(ordered_summary)
    
    except ValueError as e:
        return f"Error al procesar el texto: {str(e)}"




'''Ejercicio 6'''

import gensim.downloader as api
from scipy.spatial.distance import cdist

# Cargar el modelo preentrenado Word2Vec de Gensim (en inglés)
model = api.load("word2vec-google-news-300")


def texts_distance(text1: str, text2: str) -> float:
    """
    Calcula la distancia entre dos textos utilizando Word2Vec y la métrica Euclidiana.
    
    Args:
        text1 (str): Primer texto.
        text2 (str): Segundo texto.

    Returns:
        float: Distancia mínima entre los textos. Devuelve 'inf' si no hay palabras válidas.
    """

    # Tokenizar los textos y obtener sus vectores promedio
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    # Filtrar palabras que no están en el vocabulario del modelo
    words1 = [word for word in words1 if word in model.key_to_index]
    words2 = [word for word in words2 if word in model.key_to_index]
    
    if not words1 or not words2:
        return float("inf")   # Si algún texto no tiene palabras válidas en el vocabulario
    
    # Obtener las representaciones de las palabras
    vectors1 = [model[word] for word in words1]
    vectors2 = [model[word] for word in words2]
    
    # Calcular la distancia entre los puntos
    distance_matrix = cdist(vectors1, vectors2, metric="euclidean")
    
    # El coste mínimo de mover las palabras
    return distance_matrix.min()