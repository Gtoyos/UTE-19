from torch import nn
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dtt
from datetime import datetime as datet
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multiprocessing import Pool
DATADIR = 'E:/Data/ECD-UY'
EWH_FILE_DIR = DATADIR + '/ewh/consumption_data_customers.csv'
CUSTOMERS_DATA_FILE = DATADIR + '/ewh/consumption_data_customers.csv'
THREADS = 2

def reader(m):
  df_customers = pd.read_csv(CUSTOMERS_DATA_FILE, skiprows=range(1,2))
  #df_customers = pd.read_csv(CUSTOMERS_DATA_FILE)
  df_customers["datetime_str"] = pd.to_datetime(
      df_customers.datetime,
      unit='s',
      utc=True,
      cache=True
  ).dt.tz_convert('America/Montevideo')

  ids = df_customers["id"].unique()
  fv = pd.read_csv(DATADIR + '/thc/consumption_data_2019'+m+'.csv')
  fv = fv[fv["id"].map(lambda x: x in ids)]
  print(fv)
  return fv

def readfiles():


  # Caracterización
  # EWH tiene una frecuencia de 1 minuto, mientras que THC tiene una frecuencia de 15 min.

  # Load customers data EWH.
  #CUSTOMERS_DATA_FILE = DATADIR + '/consumption_data_customers.csv'
  #CUSTOMERS_DATA_FILE = DATADIR + '/fake/ewh.csv'
  # Load customers data THC.
  #CUSTOMERS_DATA_FILE2 = DATADIR + '/consumption_data_202009.csv'
  #CUSTOMERS_DATA_FILE2 = DATADIR + '/thc/consumption_data_201907.csv'
  CUSTOMERS_DATA_FILE2 = DATADIR + '/fake/thc.csv'

  #EWH
  df_customers = pd.read_csv(CUSTOMERS_DATA_FILE, skiprows=range(1,2))
  #df_customers = pd.read_csv(CUSTOMERS_DATA_FILE)
  df_customers["datetime_str"] = pd.to_datetime(
      df_customers.datetime,
      unit='s',
      utc=True,
      cache=True
  ).dt.tz_convert('America/Montevideo')

  ids = df_customers["id"].unique()
  #THC
  #ewh_dates = ["07","08","09","10","11","12"]
  ewh_dates = ["07","08"]
  pool = Pool(THREADS)
  frames = pool.map(reader,ewh_dates)
  pool.close()
  pool.join()
  df_customers2 = pd.concat(frames)

  #df_customers2 = pd.read_csv(CUSTOMERS_DATA_FILE2, skiprows=range(1,2), nrows=10000000)
  df_customers2["datetime_str"] = pd.to_datetime(
      df_customers2.datetime,
      unit='s',
      utc=True,
      cache=True
  ).dt.tz_convert('America/Montevideo')
  return df_customers,df_customers2

def processfiles(df_customers,df_customers2):
  def create_timeseries(df):
    df['day'] = df['datetime'].apply(lambda x: datet.utcfromtimestamp(int(x)).strftime('%Y-%m-%d'))
    df['hour'] = df['datetime'].apply(lambda x: datet.utcfromtimestamp(int(x)).strftime('%H:%M:%S'))
    return df

  df_customers = create_timeseries(df_customers)
  grouped_df = df_customers.groupby(['id','day'])
  df_timeseries = pd.DataFrame([], columns = ['ID', 'DATE','EWH_TIMESERIE'])
  for key,item in grouped_df:
    a_group = grouped_df.get_group(key)
    df_timeseries.loc[len(df_timeseries)] = [key[0],key[1],a_group[['hour','power']].to_numpy()]


  df_app = create_timeseries(df_customers2)
  grouped_df_app = df_app.groupby(['id','day'])
  df_timeseries_app =  pd.DataFrame([], columns = ['ID', 'DATE','THC_TIMESERIE'])
  for key,item in grouped_df_app:
    a_group_app = grouped_df_app.get_group(key)
    df_timeseries_app.loc[len(df_timeseries_app)] = [key[0],key[1],a_group_app[['hour','value']].to_numpy()]

  df_ewh = df_timeseries
  df_thc = df_timeseries_app

  # A partir de aqui tengo las dos series ordenadas. Aplico granularidad:
  def granularidad(x,sr):
    ts = None
    if(sr=="EWH"):
      ts = x.EWH_TIMESERIE
    elif(sr=="THC"):
      ts = x.THC_TIMESERIE
    counter = {}
    power = {}
    for v in ts:
      h = v[0].split(":")[0]
      if h not in counter:
        counter[h] = 1
        power[h] = v[1]
      else:
        counter[h] = counter[h] + 1
        power[h] = power[h] + v[1]
    r = []
    for k,v in power.items():
      i = []
      r.append([k,v/(counter[k]+0.0)])
    if(sr=="EWH"):
      x.EWH_TIMESERIE = r
    elif(sr=="THC"):
      x.THC_TIMESERIE = r
    return x

  gewh = df_ewh.apply(granularidad,axis=1,sr="EWH")
  gthc =df_thc.apply(granularidad,axis=1,sr="THC")

  print(gewh.head())
  print(gthc.head())

  # Junto los dataframes que comparten fecha y customer ID

  cc = gewh.merge(gthc,on=["ID","DATE"],how="inner")

  #genero resultado
  calefonON = []
  calefonOFF = []
  def foo(x):
    index_ewh = x.EWH_TIMESERIE[0][0]
    index_thc = x.THC_TIMESERIE[0][0]
    for y in x.EWH_TIMESERIE:
      print(y)
      print(x.THC_TIMESERIE)
      try: # En el caso de que 
        if(float(y[1])>0):
          calefonON.append([
            dtt.date.fromisoformat(x.DATE).month,
            dtt.date.fromisoformat(x.DATE).weekday(),
            x.THC_TIMESERIE[int(y[0])-int(index_thc)][1]])
        else:
          calefonOFF.append([x.DATE.split("-")[1],x.DATE.split("-")[2],x.THC_TIMESERIE[int(y[0])-int(index_thc)][1]])
      except:
        pass
  #resu = [foo(date,ewh,thc) for date,ewh,thc in zip(cc['DATE'],cc['EWH_TIMESERIE'],cc['THC_TIMESERIE'])]
  print(cc)
  cc.apply(foo,axis=1)
  print(calefonON)
  print(len(calefonON))
  return calefonON,calefonOFF
# RED NEURONAL
def clasificador(calefonON,calefonOFF):
  data = { 'data': np.concatenate((calefonON,calefonOFF)), 'target': np.concatenate((np.ones(len(calefonON)),np.zeros(len(calefonOFF)))), 'target_names': np.array(['False', 'True'], dtype='<U9')}
  x = data['data']
  y = data['target']




  sns.countplot(x = 'target', data=data)


  # Dividimos el conjunto de datos para el entrenamiento y el test de la red
  test_size = 0.2
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,shuffle=True)

  # Escalamos la entrada para que esté nomalizada
  sc = StandardScaler()
  x_train = sc.fit_transform(x_train)
  x_test = sc.fit_transform(x_test)

  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  print(device)

  # Creamos una clase para Dataset los datos de forma automática
  class dataset(Dataset):
    def __init__(self,x,y):
      self.x = torch.tensor(x,dtype=torch.float32)
      self.x = self.x.to(device)
      self.y = torch.tensor(y,dtype=torch.float32)
      self.y = self.y.to(device)
      #self.length = self.x.shape[0]
  
    def __getitem__(self,idx):
      return self.x[idx],self.y[idx]

    def __len__(self):
      return len(self.y)

  #---
  class Net(nn.Module):

    def __init__(self,input_shape):
      super().__init__()

      # Definimos la arquitectura de la red
      self.fc1 = nn.Linear(input_shape,720)
      self.fc2 = nn.Linear(720,360)
      self.fc3 = nn.Linear(360,180)
      self.fc4 = nn.Linear(180,90)
      self.fc5 = nn.Linear(90,1)
      
    # Definimos como opera la red (una red feed-forward)
    def forward(self,x):
      x = torch.relu(self.fc1(x))
      x = torch.relu(self.fc2(x))
      x = torch.sigmoid(self.fc3(x))
      x = torch.relu(self.fc4(x))
      x = torch.sigmoid(self.fc5(x))
      return x

  net = Net(input_shape=x.shape[1])
  print(net)

  #----

  # Principales hiperparámetros
  learning_rate = 0.01
  epochs = 500

  # Model , Optimizer, Loss
  model = Net(input_shape=x.shape[1])
  optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

  loss_fn = nn.BCELoss()

  #----

  # El tamaño del batch (cuantas muestras se van a leer en paralelo para actulizar la red)
  batch_size = 64

  # Creamos los conjuntos de datos
  trainset = dataset(x_train,y_train)
  testset = dataset(x_test,y_test)

  trainset_size = len(trainset)
  number_of_batches = int(trainset_size/batch_size) + 1

  # Creamos los lectores de datos de cada conjunto de datos
  trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
  testloader = DataLoader(testset,batch_size=batch_size,shuffle=True)

  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  print(device)

  # Creamos una clase para Dataset los datos de forma automática
  class dataset(Dataset):
    def __init__(self,x,y):
      self.x = torch.tensor(x,dtype=torch.float32)
      self.x = self.x.to(device)
      self.y = torch.tensor(y,dtype=torch.float32)
      self.y = self.y.to(device)
      #self.length = self.x.shape[0]
  
    def __getitem__(self,idx):
      return self.x[idx],self.y[idx]

    def __len__(self):
      return len(self.y)

  #---
  class Net(nn.Module):

    def __init__(self,input_shape):
      super().__init__()

      # Definimos la arquitectura de la red
      self.fc1 = nn.Linear(input_shape,720)
      self.fc2 = nn.Linear(720,360)
      self.fc3 = nn.Linear(360,180)
      self.fc4 = nn.Linear(180,90)
      self.fc5 = nn.Linear(90,1)
      
    # Definimos como opera la red (una red feed-forward)
    def forward(self,x):
      x = torch.relu(self.fc1(x))
      x = torch.relu(self.fc2(x))
      x = torch.sigmoid(self.fc3(x))
      x = torch.relu(self.fc4(x))
      x = torch.sigmoid(self.fc5(x))
      return x

  net = Net(input_shape=x.shape[1])
  print(net)

  #----

  # Principales hiperparámetros
  learning_rate = 0.01
  epochs = 500

  # Model , Optimizer, Loss
  model = Net(input_shape=x.shape[1])
  optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

  loss_fn = nn.BCELoss()

  #----

  # El tamaño del batch (cuantas muestras se van a leer en paralelo para actulizar la red)
  batch_size = 64

  # Creamos los conjuntos de datos
  trainset = dataset(x_train,y_train)
  testset = dataset(x_test,y_test)

  trainset_size = len(trainset)
  number_of_batches = int(trainset_size/batch_size) + 1

  # Creamos los lectores de datos de cada conjunto de datos
  trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
  testloader = DataLoader(testset,batch_size=batch_size,shuffle=True)

  # Variables auxiliares
  losses = []
  accur = []

  print('Training with {} of size {}.'.format(number_of_batches, batch_size))
  print('Start training.....')

  # Bucle principal de entrenamiento 
  for epoch in range(epochs):

    # Bucle sobre el conjunto de datos leyendo batches para actualizar los pesos de la red
    for batch_number,(_x,_y) in enumerate(trainloader):
      # Calculamos la salida de la red dada la entrada
      output = model(_x)
      # Calculamos la pérdida de la red (loss)
      loss = loss_fn(output,_y.reshape(-1,1))

      # Calculamos estadísticas de precision
      predicted = model(torch.tensor(x_train,dtype=torch.float32).to(device))
      acc = (predicted.reshape(-1).detach().numpy().round() == y_train).mean()

      # Propagamos el error y actualizamos los pesos de la ANN
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


    if epoch%2 == 0:
      losses.append(loss)
      accur.append(acc)
      print("Epoch {}\tLoss : {}\t Accuracy : {}".format(epoch,loss,acc))


  # Mostramos la evolución de la pérdida
  plt.plot(losses)
  plt.title('Loss vs Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('loss')

if __name__ == '__main__':   
  f1,f2 = readfiles()
  on,off = processfiles(f1,f2)
  clasificador(on,off)
