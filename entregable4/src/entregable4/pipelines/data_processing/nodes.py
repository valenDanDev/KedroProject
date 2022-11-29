"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def LoadByHour_File(file):
  datos = pd.read_csv(file)
  datos.index = pd.DatetimeIndex(datos["TIME"])
  datos = datos.resample('H').mean()
  return datos["SYSTEM Energ&#237;a total Fotovoltaica [W]"].plot()


def LoadByHour_Directory(directory):
  files = os.listdir(directory)
  datos = pd.DataFrame()
  for file in files:
    try:
      newData = LoadByHour_File(directory + "/" + file)
      datos = pd.concat([datos,newData])
      assert newData.shape[0]==24, "File with missing data: " + file
    except AssertionError as ErrorMissing:
      print(ErrorMissing)
    except FileNotFoundError as ErrorNotFile:
      print(ErrorNotFile)
    except errors.ParserError as ErrorParser:
      print(str(ErrorParser) + ", file: " +file)
  datos.columns = ["Potencia"]
  return datos

def fillMatrix(data):
  h = 6
  raw = 366
  column = 13
  indexD=["days"]
  row_indices = ["6:00", "7:00", "8:00", "9:00", "10:00", "11:00", "12:00","13:00", "14:00", "15:00","16:00", "17:00","18:00"]
  m = []
  for i in range(0,raw): 
    m.append([])
    for j in range(column):
      m[i].append(data["Potencia"][h])
      h = h+1
    h = h+11
  data_df = pd.DataFrame(m, columns=row_indices)
  return data_df




def standarize(data):
   dt = fillMatrix(data)
   scaler = StandardScaler().fit(dt)
   std=scaler.transform(dt)
   return std

def traspuesta(standar_data):
   dtt = dt.transpose()
   std_df=pd.DataFrame(std)
   return std


def clustering(standar_data):
   dtt = dt.transpose()
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler().fit(dtt)
   std=scaler.transform(dtt)
   std_df=pd.DataFrame(std)
   return std

def KMeans(data):
   kmeans = KMeans(n_clusters=2).fit(std_df_T)
   fig=plt.figure(figsize= (9,7))
   plt.scatter(x=std_df_T.index,y=std_df_T.iloc[:,7],c= kmeans.labels_, s=50 )
   plt.ylim(-1.5,3.0)
   plt.show()
   return plt

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    print(linkage_matrix)
    return dendrogram(linkage_matrix, **kwargs)


def dbscan(data):
   clustering = DBSCAN(eps=2, min_samples=2).fit(std_df_T)
   print(clustering.labels_)
   plt.figure(figsize=(9, 7))
   plt.scatter(x=std_df_T.index,y=std_df_T.iloc[:,5],c= clustering.labels_, s=50 )
   plt.xlabel("Días evaluados")
   plt.ylabel("Generación de energía")
   return plt

def PSA(data):
   pca=PCA(n_components=2)
   pca_info_data=pca.fit_transform(std_df_T)
   print(pca_info_data)
   pca_info_data_df=pd.DataFrame(data=pca_info_data,columns=["componente_1","componente_2"])
   pca_info_name_data=pd.concat([pca_info_data_df,std_df_T[ ["Kmeans_Cluster"]]],axis=1)
   pca_info_name_data
   return pca_info_name_data

def create_confusion_matrix(companies: pd.DataFrame):
    actuals = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    predicted = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]
    data = {"y_Actual": actuals, "y_Predicted": predicted}
    df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
    confusion_matrix = pd.crosstab(
        df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"]
    )
    sn.heatmap(confusion_matrix, annot=True)
    return plt

import plotly.express as px
import pandas as pd

# the below function uses plotly.express
def compare_passenger_capacity(preprocessed_shuttles: pd.DataFrame):
    fig = px.bar(
        data_frame=preprocessed_shuttles.groupby(["shuttle_type"]).mean().reset_index(),
        x="shuttle_type",
        y="passenger_capacity",
    )
    return fig


# the below function uses plotly.graph_objects
def compare_passenger_capacity(preprocessed_shuttles: pd.DataFrame):
    data_frame = preprocessed_shuttles.groupby(["shuttle_type"]).mean().reset_index()
    fig = go.Figure(
        [
            go.Bar(
                x=data_frame["shuttle_type"],
                y=data_frame["passenger_capacity"],
            )
        ]
    )
    fig.show()
    return fig
