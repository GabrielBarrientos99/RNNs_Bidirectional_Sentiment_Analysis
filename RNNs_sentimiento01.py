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

class RNN_ANN(nn.Module):
    def __init__(self, input_size, hidden_size , output_size=1):
        super(RNN_ANN,self).__init__()

        # Atributos (tipo NNode)
        self.cell = RNNode(input_size=input_size,hidden_size=hidden_size)
        # Atributo ANN simple, Capa que determina el sentimiento
        self.f_sent = nn.Linear(hidden_size,output_size)
    
    def forward(self, secuencia ,h = None):
        ''' Metodo que genera un tensor h_t usando la clase Node y se le pasa un h_anterior

        params:
        - secuencia list[ x_input ] : representa una matriz de n_inputs con tamaño x_input
        - h_inicial: representa el tensor oculto inicial de la capa oculta
        '''

        for x_input in secuencia:
            h = self.cell(x_input, h)
        
        #print(f'Output shape before tanh: {h.shape}')
        output = self.f_sent(h)
        #print(f'Output shape before tanh: {output.shape}')
        output = torch.tanh(output)
        #print(f'Output shape after tanh: {output.shape}')
        output = output.squeeze()
        #print(f'Output shape after squeeze: {output.shape}')
        return output


class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, Preprocessor):
        super(SentimentRNN,self).__init__()
        # Atributos - preprocesador
        self.Preprocessor = Preprocessor

        # Atributos - modelo (tipo RNN_ANN)
        self.model= RNN_ANN(input_size,hidden_size,output_size)
        # Definimos la funcion de perdida
        self.loss = nn.MSELoss()
        # Definimos el optimizador (le pasamos los parametros del modelo)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self,X_train, y_train,epoch=10):
        ''' Metodo que entrena el modelo con un conjunto de datos de entrenamiento
        preprocesador como un conjunto de secuencias y sus targets
        '''
        # ALmacenamos la perdida total de cada epoca en una lista
        loss_list = []
        for e in range(epoch):
            # Almacenamos la perdida total
            total_loss = 0
            for secuencia,target in zip(X_train,y_train):
                #print(f'Secuencia: {secuencia}')
                #print(f'tamaño de la secuencia: {len(secuencia)}')
                # Inicializamos el gradiente
                self.optimizer.zero_grad()

                # Pasamos la secuencia a la red
                output = self.model(secuencia)

                # Imprimimos el output y el target
                #print(f'Output: {output}, Target: {target}')
                
                # Calculamos la perdida
                loss = self.loss(output,target)

                
                # Calculamos el gradiente
                loss.backward()

                # Actualizamos los pesos
                self.optimizer.step()

                # Acumulamos la perdida
                total_loss += loss.item()

            # Almacenamos la perdida total de la epoca
            loss_list.append(total_loss)
            print(f'Epoch {e+1} - Loss: {total_loss}')
        
        # Almacenamos los errores en un atributo
        self.loss_list = loss_list
    
    def plot_loss(self):
        ''' Metodo que grafica la perdida del modelo
        '''
        
        plt.plot(self.loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def evaluate(self,X_test,y_test):
        ''' Metodo que evalua el modelo con un conjunto de datos de test
        preprocesador como un conjunto de secuencias y sus targets
        '''
        correct = 0
        total = 0

        # Desactivamos el calculo de gradientes
        with torch.no_grad():
            # Almacenamos la perdida total
            total_loss = 0
            for secuencia,target in zip(X_test,y_test):
                output = self.model(secuencia)
                # Calculamos el loss
                loss = self.loss(output,target)
                # Acumulamos la perdida
                total_loss += loss.item()
            
            # Calculamos el error cuadratico medio
            mse = total_loss/len(X_test)
            print(f'MSE: {mse}')
        
        return mse
    
    def predict(self,texto):
        ''' Metodo que predice el sentimiento de un texto
        '''
        # Preprocesamos el texto
        secuencia = self.Preprocessor.get_embedding_texto(texto)
        # Pasamos la secuencia por la red
        output = self.model(secuencia)
        # Retornamos el valor de salida
        return output.item()

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