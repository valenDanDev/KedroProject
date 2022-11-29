"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fillMatrix,LoadByHour_File,LoadByHour_Directory,standarize,traspuesta,clustering,KMeans,plot_dendrogram,dbscan,PSA,create_confusion_matrix,compare_passenger_capacity

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
           node(
                func=fillMatrix,
                inputs=["PotenciaHoras2020"],
                outputs=["Potencia12Horas"],
                name="pre_process_data",
            ),
             node(
                func=LoadByHour_File,
                inputs=["PotenciaHoras2020"],
                outputs=["Potencia_HoraDiaEspecifico"],
                name="LoadByHour_File",
            ),
            node(
                func=LoadByHour_Directory,
                inputs=["PotenciaHoras2020"],
                outputs=["Potencia_DiariaTodoAñoEspecifico"],
                name="LoadByHour_Directory",
            ),
            node(
                func=standarize,
                inputs=["Potencia12Horas"],
                outputs=["estandarizar_Potencia12_Horas"],
                name="standarize",
            ),  
             node(
                func=traspuesta,
                inputs=["estandarizar_Potencia12_Horas"],
                outputs=["traspuesta_Potencia_12Horas"],
                name="traspuesta",
            ),
              node(
                func=clustering,
                inputs=["traspuesta_Potencia_12Horas"],
                outputs=["clustering_clasificacion"],
                name="clustering",
            ),
              node(
                func=KMeans,
                inputs=["clustering_clasificacion"],
                outputs=["kmeans_cluster"],
                name="KMeans",
            ),
            node(
                func=plot_dendrogram,
                inputs=["clustering_clasificacion"],
                outputs=["cluster_jerarquico"],
                name="plot_dendrogram",
            ),
            node(
                func=dbscan,
                inputs=["clustering_clasificacion"],
                outputs=["DBSCAN_cluster"],
                name="dbscan",
            ),
            node(
                func=PSA,
                inputs=["clustering_clasificacion"],
                outputs=["PSA_graph"],
                name="PSA",
            ),
              node(
                func=compare_passenger_capacity,
                inputs=["preprocessed_shuttles"],
                outputs=["shuttle_passenger_capacity_plot"],
            ),
    ])
