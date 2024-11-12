import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import pickle

class RNNode(nn.Module):
    ''' Clase que representa un nodo de una RNN
    params:
    - input_size(int): tamaño del embeding de la caracteristica de entrada
    - hiden_size(int): tamaño del vector del estado oculto el cual va retener información de la secuencia
    '''
    def __init__(self,input_size:int,hidden_size:int) -> None :
        # Inicializa la clase base nn.Module
        super(RNNode,self).__init__()

        # Atributos (int) de la clase
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Atributos (nn.Linear [ capa lineal ]) de la clase
        #-------------------------------------------------------------#
        # Capa Lineal [entradas: input_size] -> [salidas : hidden_size]
        self.ih = nn.Linear(input_size,hidden_size)

        # Capa Lineal [entradas: hidden_size] -> [salidas : hidden_size]
        self.hh = nn.Linear(hidden_size,hidden_size)
    
    def forward(self,x,h_anterior=None):
        # Evaluamos en el caso que no exista un h
        if h_anterior is None:
            # Inicializa h como un tensor de zeros (1, hidden_size)
            #print(f'x: {x}')
            h_anterior = torch.zeros( 1 , self.hidden_size )
            
            #print(f'H_0 (size): {h_anterior.size()}')
            #print(f'H_0 : \n {h_anterior}')
        
        h = torch.tanh( self.ih(x) + self.hh(h_anterior) )
        
        return h

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

class RNN_ANN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(RNN_ANN2, self).__init__()

        # Atributos (tipo RNNode)
        self.cell = RNNode(input_size=input_size, hidden_size=hidden_size)

        # Añadiendo más capas ANN para el procesamiento posterior
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Nueva capa lineal
        self.relu1 = nn.ReLU()  # Función de activación ReLU
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Otra capa lineal, reduce dimensiones
        self.relu2 = nn.ReLU()  # Otra ReLU
        self.f_sent = nn.Linear(hidden_size // 2, output_size)  # Capa final ajustada después de nuevas capas

    def forward(self, secuencia, h=None):
        for x_input in secuencia:
            h = self.cell(x_input, h)

        # Pasando el estado oculto final a través de las capas adicionales
        x = self.fc1(h)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        output = self.f_sent(x)
        output = torch.tanh(output)  # Manteniendo la función de activación tanh para la salida
        output = output.squeeze()  # Eliminando dimensiones extra

        return output


class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, Preprocessor, dropout=0.5, lr=0.001):
        super(SentimentRNN, self).__init__()
        self.Preprocessor = Preprocessor
        self.model = RNN_ANN2(input_size, hidden_size, output_size)
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

    def predict(self, texto):
        secuencia = self.Preprocessor.get_embedding_texto(texto)
        output = self.model(secuencia)
        return output.item()