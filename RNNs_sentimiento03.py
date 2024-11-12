import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from RNNs_sentimiento01 import RNNode

class BidirectionalRNN_ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(BidirectionalRNN_ANN, self).__init__()

        # Definimos la capa RNN1 (->)
        self.rnn1 = RNNode(input_size,hidden_size)
        # Definimos la capa RNN2 (<-)
        self.rnn2 = RNNode(input_size,hidden_size)

        # Definimos la estructura ANN que toma la concatenación de las salidas de las capas RNN
        
        # Añadiendo más capas ANN para el procesamiento posterior
        self.fc1 = nn.Linear(2*hidden_size, 2*hidden_size)  # Nueva capa lineal
        self.relu1 = nn.ReLU()  # Función de activación ReLU
        self.fc2 = nn.Linear(2*hidden_size, hidden_size)  # Otra capa lineal, reduce dimensiones
        self.relu2 = nn.ReLU()  # Otra ReLU
        self.f_sent = nn.Linear(hidden_size, output_size)  # Capa final ajustada después de nuevas capas

    
    def forward(self, secuencia ):
        # Recorremos la secuencia de entrada en orden
        h1,h2 = None, None
        for x_input in secuencia:
            h1 = self.rnn1(x_input,h1)
        # Recorremos la secuencia de entrada en orden inverso
        for x_input in reversed(secuencia):
            h2 = self.rnn2(x_input,h2)
        
        # Concatenamos las salidas de las capas RNN
        h = torch.cat((h1,h2),1)

        # Pasamos la concatenación por la capa ANN
        x = self.fc1(h)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
       
        output = self.f_sent(x)
        output = torch.tanh(output)  # Manteniendo la función de activación tanh para la salida
        output = output.squeeze()  # Eliminando dimensiones extra

        return output


class SentimentBidirectional_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, Preprocessor, dropout=0.5, lr=0.001):
        super(SentimentBidirectional_RNN, self).__init__()
        self.Preprocessor = Preprocessor
        self.model = BidirectionalRNN_ANN(input_size, hidden_size, output_size)
        self.loss = nn.MSELoss()
        self.dropout = nn.Dropout(dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, y_train, epoch=10, batch_size=32):
        loss_list = []
        for e in range(epoch):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                self.optimizer.zero_grad()
                outputs = []
                for secuencia in X_batch:
                    output = self.model(secuencia)
                    outputs.append(output)

                outputs = torch.stack(outputs).squeeze()
                loss = self.loss(outputs, torch.tensor(y_batch))
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                #
            loss_list.append(total_loss)
            print(f'Epoch {e+1} - Loss: {total_loss}')
        self.loss_list = loss_list

    def plot_loss(self):
        plt.plot(self.loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            total_loss = 0
            outputs = []
            for secuencia in X_test:
                output = self.model(secuencia)
                outputs.append(output)
            outputs = torch.stack(outputs).squeeze()
            total_loss = self.loss(outputs, torch.tensor(y_test)).item()
            mse = total_loss / len(y_test)
            print(f'MSE: {mse}')
        return mse
    
    def predict_sentence_embeding(self, texto):
        secuencia = self.Preprocessor.get_embedding_parrafo(texto)
        output = self.model(secuencia)
        return output.item()

    def predict(self, texto):
        secuencia = self.Preprocessor.get_embedding_texto(texto)
        output = self.model(secuencia)
        return output.item()

# Creamos una clase para preprocesar los datos
class Preprocessor:    
    def __init__(self, path: str, embedding_type='glove') -> None:
        # Atributo (str) path de la clase
        self.path = path
        # Leemos el archivo
        self.text = self.read_path(self.path)
        # Creamos un diccionario de parrafos y sentimientos
        self.data = self.create_data(self.text)
        #print(self.data)
        # Creamos un dataframe
        self.df = self.create_df(self.data)
        # Creamos un df de palabras y sentimientos
        self.df_words = self.create_df_words(self.df)
        
        # Cargar modelo de embeddings
        self.embedding_type = embedding_type
        if embedding_type == 'glove':
            print("Cargando GloVe...")
            # extraemos el archivo pickle
            self.embedding_model = pickle.load(open("glove_model.pkl", 'rb'))
            self.embedding_dim = 100
        elif embedding_type == 'word2vec':
            print("Cargando Word2Vec...")
            self.embedding_model = pickle.load(open("word2vec_model.pkl", 'rb'))
            self.embedding_dim = 300
        else:
            raise ValueError("Tipo de embedding no soportado. Usa 'glove' o 'word2vec'.")

        # Creamos una lista de tensores de secuencias y sus targets
        self.X, self.y = self.create_X_y_words(self.df_words)
    
    def get_embedding_texto(self, texto: str) -> list:
        ''' Método que obtiene los embeddings de un texto '''
        texto = re.findall(r'\w+', texto.lower())
        embeddings = [self.embedding_model[word] for word in texto if word in self.embedding_model]
        # Convertimos la lista de embeddings a un tensor
        if embeddings:
                # convertimos la lista de embeddings a un tensor
            embeddings = [torch.tensor(embedding) for embedding in embeddings]
        else:
            embeddings = np.zeros(self.embedding_dim)  # Vector de ceros si no se encuentran palabras válidas

        return embeddings
    
    def get_embedding_parrafo(self, parrafo: str) -> torch.Tensor:
        ''' Método que obtiene el embedding promedio de un párrafo '''
        parrafo = re.findall(r'\w+', parrafo.lower())
        embeddings = [self.embedding_model[word] for word in parrafo if word in self.embedding_model]
        if embeddings:
            # Calculamos el promedio de los embeddings
            paragraph_embedding = torch.tensor(embeddings).mean(dim=0)
            # Reshape del tensor
            paragraph_embedding = paragraph_embedding.view(1,-1)
        else:
            paragraph_embedding = np.zeros(self.embedding_dim)
        
        return paragraph_embedding
    
    def create_X_y_text(self, df: str) -> torch.Tensor:
        X = []
        y = []
        for parrafo,sentimiento in zip(df['parrafos'],df['sentimientos']):
            embeddings = [self.embedding_model[word] for word in parrafo if word in self.embedding_model]
            if embeddings:
                # Calculamos el promedio de los embeddings
                paragraph_embedding = torch.tensor(embeddings).mean(dim=0)
                # Reshape del tensor
                paragraph_embedding = paragraph_embedding.view(1,-1)
            else:
                paragraph_embedding = np.zeros(self.embedding_dim)  # Vector de ceros si no se encuentran palabras válidas
            X.append(paragraph_embedding)
            # Convertimos el sentimiento a tensor
            sentimiento = torch.tensor(sentimiento)
            y.append(sentimiento)
        
        # Convertimos las listas a tensores
        X = torch.stack(X)
        
        return X,y
    
    def create_X_y_words(self, df: pd.DataFrame) -> list:
        ''' Método que crea una lista de embeddings de secuencias '''
        X = []
        y = []
        for parrafo,sentimiento in zip(df['parrafos'],df['sentimientos']):
            embeddings = [self.embedding_model[word] for word in parrafo if word in self.embedding_model]
            if embeddings:
                # convertimos la lista de embeddings a un tensor
                paragraph_embedding = [torch.tensor(embedding) for embedding in embeddings]
            else:
                paragraph_embedding = np.zeros(self.embedding_dim)  # Vector de ceros si no se encuentran palabras válidas
            X.append(paragraph_embedding)
            # Convertimos el sentimiento a tensor
            sentimiento = torch.tensor(sentimiento)
            y.append(sentimiento)
    
        return X, y
    
    def create_df_words(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Método que crea un dataframe de palabras y sentimientos '''
        df_words = df.copy()
        df_words['parrafos'] = df['parrafos'].apply(lambda x: re.findall(r'\w+', x.lower()))
        return df_words
    
    def create_df(self, data: dict) -> pd.DataFrame:
        ''' Método que crea un dataframe a partir de un diccionario de parrafos y sentimientos '''
        df = pd.DataFrame(data)
        return df
    
    def create_data(self, text: str) -> dict:
        ''' Método que crea un diccionario de parrafos y sentimientos '''
        parrafos = re.findall(r'Parrafo_\d+:\n(.*?)\nsentimiento_\d+:\n', text, re.DOTALL)
        sentimientos = re.findall(r'sentimiento_\d+:\n(.*?)(?:\n|$)', text, re.DOTALL)
        # Convertimos los sentimientos a float
        sentimientos = [float(sentimiento) for sentimiento in sentimientos]
        data = {'parrafos': parrafos, 'sentimientos': sentimientos }
        return data
    
    def read_path(self, path: str) -> str:
        ''' Método que lee un archivo de texto y retorna el texto '''
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    

def evaluate_model_bidirectional_plotly(model, sentences, target_sentiment):
    # Obtener los sentimientos predichos
    predicted_sentiment = [model.predict_sentence_embeding(sentence) for sentence in sentences]

    # Imprimir la oracion, el sentimiento esperado y el sentimiento predicho
    for i, (sentence, target, predicted) in enumerate(zip(sentences, target_sentiment, predicted_sentiment)):
        print(f"Oración {i+1}: {sentence}")
        print(f"Sentimiento esperado: {target:.2f}")
        print(f"Sentimiento predicho: {predicted:.2f}")
        print()
    
    # Crear etiquetas para el eje x
    x_labels = [f"Oración {i+1}" for i in range(len(sentences))]
    
    # Crear el gráfico de barras agrupadas
    fig = go.Figure()

    # Sentimiento esperado
    fig.add_trace(go.Bar(
        x=x_labels,
        y=target_sentiment,
        name='Sentimiento esperado',
        marker=dict(color='rgba(55, 128, 191, 0.7)'),
    ))

    # Sentimiento predicho
    fig.add_trace(go.Bar(
        x=x_labels,
        y=predicted_sentiment,
        name='Sentimiento predicho',
        marker=dict(color='rgba(219, 64, 82, 0.7)'),
    ))

    # Personalizar el diseño del gráfico
    fig.update_layout(
        title="Comparación de Sentimiento Esperado vs. Sentimiento Predicho",
        xaxis_title="Oraciones",
        yaxis_title="Sentimiento",
        barmode='group',
        xaxis_tickangle=-45,
        template="plotly_white",
    )

    # Mostrar el gráfico
    fig.show()
