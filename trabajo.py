# --------------------LIBRERÍAS----------------------------#
import json
import os
import io 
import sys
from webbrowser import get
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import cufflinks
import folium
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly_express as px
import requests
from branca.colormap import LinearColormap
from folium.plugins import (
    FastMarkerCluster, FloatImage, HeatMap, MarkerCluster)
from plotly.offline import init_notebook_mode, iplot
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
from PIL import Image
import streamlit as st
from plotly.subplots import make_subplots
import pydeck as pdk
from PIL import Image
import geopandas as gpd
import seaborn as sns
from folium.plugins import FastMarkerCluster
from branca.colormap import LinearColormap
import plotly_express as px
from scipy.stats import ttest_ind
from scipy.stats import shapiro
from scipy.stats import f_oneway
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import warnings


sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

def main():

    # --------------------------------------------------------CONFIGURACIÓN DE LA PÁGINA---------------------------------------------------#
    st.set_page_config(page_title="El cáncer en España",layout="wide", page_icon="🧪", initial_sidebar_state="expanded")
    st.set_option("deprecation.showPyplotGlobalUse", False)   

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # -----------------------------------------------------------------Cabecera----------------------------------------------------------------
    col1, col2= st.columns(2)

    with col1:
        st.title("Marco A. García Palomo")

    with col2:
        st.write(' ')
    col1, col2 = st.columns(2)

    with col1:
        st.title("Evolución del cáncer en España")

    with col2:
        st.write(' ')
    st.markdown('')
    st.markdown('')
    st.markdown('')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        
        image = Image.open('img/12-biologa-y-marcadores-tumorales-13-728.webp')
        st.image(image, caption="https://es.slideshare.net/basmedblog/12-biologa-y-marcadores-tumorales",use_column_width='auto')
    with col3:
        st.write(' ')

    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>El cáncer es una enfermedad en la que las células del cuerpo comienzan a crecer y dividirse de manera anormal y fuera de control. A medida que estas células se multiplican, pueden formar tumores y propagarse a otras partes del cuerpo. Hay muchos tipos diferentes de cáncer, y cada uno se origina en un tipo específico de célula.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Los factores de riesgo para el cáncer son variados y pueden incluir factores hereditarios, ambientales y de estilo de vida. Algunos de los factores de riesgo más comunes incluyen:</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Edad.</div> """, unsafe_allow_html=True)
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Historial familiar.</div> """, unsafe_allow_html=True)
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Estilo de vida.</div> """, unsafe_allow_html=True)
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Infecciones.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Es importante recordar que tener un factor de riesgo no siempre significa que se desarrollará cáncer. La mayoría de las personas que tienen factores de riesgo nunca desarrollan la enfermedad, y personas que desarrollan cáncer no tienen muchos factores de riesgo conocidos. Sin embargo, conocer los factores de riesgo puede ayudar a las personas a tomar medidas para reducir su riesgo de desarrollar cáncer y detectar la enfermedad en una etapa temprana, cuando es más tratable.</div> """, unsafe_allow_html=True)
    
    # -----------------------------------------------LECTURA DE DATOS Y PREPROCESAMIENTO------------------------------------#

  # Carga de datos
    in19 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Incidencia_2019_Provincia.csv', sep=';')
    in20 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Incidencia_2020_Provincia.csv', sep=';')
    in21 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Incidencia_2021_Provincia.csv', sep=';')
    in22 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Incidencia_2022_Provincia.csv', sep=';')
    mo19 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Mortalidad_2019_Provincia.csv', sep=';')
    mo20 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Mortalidad_2020_Provincia.csv', sep=';')
    mo21 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Mortalidad_2021_Provincia.csv', sep=';')
    mo22 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Mortalidad_2022_Provincia.csv', sep=';')
    pr19 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Prevalencia_2019_Provincia.csv', sep=';')
    pr20 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Prevalencia_2020_Provincia.csv', sep=';')
    pr21 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Prevalencia_2021_Provincia.csv', sep=';')
    pr22 = pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/Prevalencia_2022_Provincia.csv', sep=';')
    df= pd.read_csv('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/dimension_cancer.csv')
    mapa = gpd.read_file('C:/Users/marco/OneDrive/Escritorio/CURSO/Trabajo final/input/provincias-espanolas.geojson')

    # Preprocesado
        # Cambiamos la ',' por un '.'
    in19.iloc[:,1:] = in19.iloc[:,1:].applymap(lambda x: float(x.replace(",", ".")))

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in in19.columns:
        if col == 'Unidad Territorial':
            continue
        in19=in19.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    in19=in19.round(2)

    # Cambiamos la ',' por un '.'
    in20.iloc[:,1:] = in20.iloc[:,1:].applymap(lambda x: float(x.replace(",", ".")))

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in in20.columns:
        if col == 'Unidad Territorial':
            continue
        in20=in20.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    in20=in20.round(2)

    # Cambiamos la ',' por un '.'
    in21.iloc[:,1:] = in21.iloc[:,1:].applymap(lambda x: float(x.replace(",", ".")))

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in in21.columns:
        if col == 'Unidad Territorial':
            continue
    in21=in21.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    in21=in21.round(2)

    # Cambiamos la ',' por un '.'
    in22.iloc[:,1:] = in22.iloc[:,1:].applymap(lambda x: float(x.replace(",", ".")))

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in in22.columns:
        if col == 'Unidad Territorial':
            continue
        in22=in22.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    in22=in22.round(2)

    # Cambiamos la ',' por un '.'
    def replace_comma(x):
        if isinstance(x, str):
            return x.replace(",", ".")
        else:
            return x

    # Aplicar la función a las columnas de tipo object
    mo19 = mo19.applymap(replace_comma)

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in mo19.columns:
        if col == 'Unidad Territorial':
            continue
        mo19=mo19.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    mo19=mo19.round(2)

    # Cambiamos la ',' por un '.', aplicando la función anterior
    mo20 = mo20.applymap(replace_comma)

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in mo20.columns:
        if col == 'Unidad Territorial':
            continue
        mo20=mo20.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    mo20=mo20.round(2)

    # Cambiamos la ',' por un '.', usando la función.
    mo21 = mo21.applymap(replace_comma)

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in mo21.columns:
        if col == 'Unidad Territorial':
            continue
        mo21=mo21.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    mo21=mo21.round(2)

    # Cambiamos la ',' por un '.'
    mo22= mo22.applymap(replace_comma)

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in mo22.columns:
        if col == 'Unidad Territorial':
            continue
        mo22=mo22.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    mo22=mo22.round(2)

    # Cambiamos la ',' por un '.'
    pr19.iloc[:,1:] = pr19.iloc[:,1:].applymap(lambda x: float(x.replace(",", ".")))

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in pr19.columns:
        if col == 'Unidad Territorial':
            continue
        pr19=pr19.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    pr19=pr19.round(2)

    # Cambiamos la ',' por un '.'
    pr20.iloc[:,1:] = pr20.iloc[:,1:].applymap(lambda x: float(x.replace(",", ".")))

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in pr20.columns:
        if col == 'Unidad Territorial':
            continue
        pr20=pr20.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    pr20=pr20.round(2)

    # Cambiamos la ',' por un '.'
    pr21.iloc[:,1:] = pr21.iloc[:,1:].applymap(lambda x: float(x.replace(",", ".")))

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in pr21.columns:
        if col == 'Unidad Territorial':
            continue
        pr21=pr21.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    pr21=pr21.round(2)

    # Cambiamos la ',' por un '.'
    pr22.iloc[:,1:] = pr22.iloc[:,1:].applymap(lambda x: float(x.replace(",", ".")))

    # Hacemos una función para que nos pase todos los datos de str a float, menos la columna 'Unidad Territorial'
    for col in pr22.columns:
        if col == 'Unidad Territorial':
            continue
        pr22=pr22.astype({str(col):'float'})

    # Redondeamos todo el dataset a 2 decimales para trabajar más cómodo
    pr22=pr22.round(2)

    # Cambiar los nombres de datos de in22 para que coincida con los nombres de mapa

    in22.replace({'Unidad Territorial': 'Valencia/València'}, {'Unidad Territorial': 'València' }, inplace=True)
    in22.replace({'Unidad Territorial': 'Palmas, Las'}, {'Unidad Territorial': 'Las Palmas' }, inplace=True)
    in22.replace({'Unidad Territorial': 'Balears, Illes'}, {'Unidad Territorial': 'Illes Balears' }, inplace=True)
    in22.replace({'Unidad Territorial': 'Alicante/Alacant'}, {'Unidad Territorial': 'Alacant' }, inplace=True)
    in22.replace({'Unidad Territorial': 'Coruña, A'}, {'Unidad Territorial': 'A Coruña' }, inplace=True)
    in22.replace({'Unidad Territorial': 'Rioja, La'}, {'Unidad Territorial': 'La Rioja' }, inplace=True)
    in22.replace({'Unidad Territorial': 'Gipuzkoa'}, {'Unidad Territorial': 'Gipuzcoa' }, inplace=True)
    in22.replace({'Unidad Territorial': 'Araba/Álava'}, {'Unidad Territorial': 'Araba' }, inplace=True)
    in22.replace({'Unidad Territorial': 'Castellón/Castelló'}, {'Unidad Territorial': 'Castelló' }, inplace=True)

    # Cambiar los nombres de datos de mo22 para que coincida con los nombres de mapa

    mo22.replace({'Unidad Territorial': 'Valencia/València'}, {'Unidad Territorial': 'València' }, inplace=True)
    mo22.replace({'Unidad Territorial': 'Palmas. Las'}, {'Unidad Territorial': 'Las Palmas' }, inplace=True)
    mo22.replace({'Unidad Territorial': 'Balears. Illes'}, {'Unidad Territorial': 'Illes Balears' }, inplace=True)
    mo22.replace({'Unidad Territorial': 'Alicante/Alacant'}, {'Unidad Territorial': 'Alacant' }, inplace=True)
    mo22.replace({'Unidad Territorial': 'Coruña. A'}, {'Unidad Territorial': 'A Coruña' }, inplace=True)
    mo22.replace({'Unidad Territorial': 'Rioja. La'}, {'Unidad Territorial': 'La Rioja' }, inplace=True)
    mo22.replace({'Unidad Territorial': 'Gipuzkoa'}, {'Unidad Territorial': 'Gipuzcoa' }, inplace=True)
    mo22.replace({'Unidad Territorial': 'Araba/Álava'}, {'Unidad Territorial': 'Araba' }, inplace=True)
    mo22.replace({'Unidad Territorial': 'Castellón/Castelló'}, {'Unidad Territorial': 'Castelló' }, inplace=True)

    # Cambiar los nombres de datos de pr22 para que coincida con los nombres de mapa

    pr22.replace({'Unidad Territorial': 'Valencia/València'}, {'Unidad Territorial': 'València' }, inplace=True)
    pr22.replace({'Unidad Territorial': 'Palmas, Las'}, {'Unidad Territorial': 'Las Palmas' }, inplace=True)
    pr22.replace({'Unidad Territorial': 'Balears, Illes'}, {'Unidad Territorial': 'Illes Balears' }, inplace=True)
    pr22.replace({'Unidad Territorial': 'Alicante/Alacant'}, {'Unidad Territorial': 'Alacant' }, inplace=True)
    pr22.replace({'Unidad Territorial': 'Coruña, A'}, {'Unidad Territorial': 'A Coruña' }, inplace=True)
    pr22.replace({'Unidad Territorial': 'Rioja, La'}, {'Unidad Territorial': 'La Rioja' }, inplace=True)
    pr22.replace({'Unidad Territorial': 'Gipuzkoa'}, {'Unidad Territorial': 'Gipuzcoa' }, inplace=True)
    pr22.replace({'Unidad Territorial': 'Araba/Álava'}, {'Unidad Territorial': 'Araba' }, inplace=True)
    pr22.replace({'Unidad Territorial': 'Castellón/Castelló'}, {'Unidad Territorial': 'Castelló' }, inplace=True)

    # Cambiar nombres de CCAA para ponerlos bien 

    df.replace({'CCAA': 'Navarra, Comunidad Foral de'}, {'CCAA': 'Comunidad Floral de Navarra' }, inplace=True)
    df.replace({'CCAA': 'Palmas, Las'}, {'CCAA': 'Las Palmas' }, inplace=True)
    df.replace({'CCAA': 'Balears, Illes'}, {'CCAA': 'Illes Balears' }, inplace=True)
    df.replace({'CCAA': 'Alicante/Alacant'}, {'CCAA': 'Alacant' }, inplace=True)
    df.replace({'CCAA': 'Coruña, A'}, {'CCAA': 'A Coruña' }, inplace=True)
    df.replace({'CCAA': 'Rioja, La'}, {'CCAA': 'La Rioja' }, inplace=True)
    df.replace({'CCAA': 'Gipuzkoa'}, {'CCAA': 'Gipuzcoa' }, inplace=True)
    df.replace({'CCAA': 'Araba/Álava'}, {'CCAA': 'Araba' }, inplace=True)
    df.replace({'CCAA': 'Castellón/Castelló'}, {'CCAA': 'Castelló' }, inplace=True)
    df.replace({'CCAA': 'Asturias, Principado de'}, {'CCAA': 'Principado de Asturias' }, inplace=True)
    df.replace({'CCAA': 'Madrid, Comunidad de'}, {'CCAA': 'Comunidad de Madrid' }, inplace=True)
    df.replace({'CCAA': 'Murcia, Región de'}, {'CCAA': 'Región de Murcia' }, inplace=True)
    
    #-----------------------------------------Mapa de españa------------------------------------------------------------#

    # Realiza un merge entre los datos y el mapa

    merged = mapa.merge(in22, left_on='provincia', right_on='Unidad Territorial')
    map_dict = merged.set_index('provincia')['Tasa total'].to_dict()
    color_scale = LinearColormap(['yellow','red'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

    # Hacer el mapa

    def get_color(feature):
        value = map_dict.get(feature['properties']['provincia'])
        return color_scale(value)
    map1 = folium.Map(location=[40.0000000, -4.0000000], zoom_start=5)
    folium.GeoJson(data=merged,
                name='España',
                tooltip=folium.features.GeoJsonTooltip(fields=['provincia', 'Tasa total'],
                                                        labels=True,
                                                        sticky=False),
                style_function= lambda feature: {
                    'fillColor': get_color(feature),
                    'color': 'black',
                    'weight': 1,
                    'dashArray': '5, 5',
                    'fillOpacity':0.5
                    },
                highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map1)
    

    # Realiza un merge entre los datos y el mapa

    merged = mapa.merge(mo22, left_on='provincia', right_on='Unidad Territorial')
    map_dict = merged.set_index('provincia')['Tasa total'].to_dict()
    color_scale = LinearColormap(['yellow','red'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

    # Hacer el mapa

    def get_color(feature):
        value = map_dict.get(feature['properties']['provincia'])
        return color_scale(value)
    map2 = folium.Map(location=[40.0000000, -4.0000000], zoom_start=5)
    folium.GeoJson(data=merged,
                name='España',
                tooltip=folium.features.GeoJsonTooltip(fields=['provincia', 'Tasa total'],
                                                        labels=True,
                                                        sticky=False),
                style_function= lambda feature: {
                    'fillColor': get_color(feature),
                    'color': 'black',
                    'weight': 1,
                    'dashArray': '5, 5',
                    'fillOpacity':0.5
                    },
                highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map2)
    

    # Realiza un merge entre los datos y el mapa

    merged = mapa.merge(pr22, left_on='provincia', right_on='Unidad Territorial')
    map_dict = merged.set_index('provincia')['Tasa total'].to_dict()
    color_scale = LinearColormap(['yellow','red'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

    # Hacer el mapa

    def get_color(feature):
        value = map_dict.get(feature['properties']['provincia'])
        return color_scale(value)
    map3 = folium.Map(location=[40.0000000, -4.0000000], zoom_start=5)
    folium.GeoJson(data=merged,
                name='España',
                tooltip=folium.features.GeoJsonTooltip(fields=['provincia', 'Tasa total'],
                                                        labels=True,
                                                        sticky=False),
                style_function= lambda feature: {
                    'fillColor': get_color(feature),
                    'color': 'black',
                    'weight': 1,
                    'dashArray': '5, 5',
                    'fillOpacity':0.5
                    },
                highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map3)
    
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.title("Mapas de España: incidencia, mortalidad y prevalencia del cáncer")
    st.markdown('')
    st.markdown('')
    st.markdown('')
   
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>La incidencia hace referencia a los nuevos casos de cáncer detectados en un periodo de tiempo determinado (habitualmente un año), mientras que por tasa de incidencia entendemos la razón de nuevos casos diagnosticados por cada 100.000 habitantes (es decir, la división de los nuevos casos detectados en un año entre la población total de referencia susceptible de enfermar, multiplicado por 100.000). Este indicador nos va a dar una medida del impacto directo que tiene un tipo de cáncer específico, sin considerar otros factores como la supervivencia relativa del mismo.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>La mortalidad es el único indicador del que tenemos registros observados para la totalidad de población de referencia en España.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>La prevalencia (de periodo), en cambio, se refiere al número de personas que han sido diagnosticadas de la enfermedad en el periodo de tiempo de referencia (en este caso cuatro años), y continúan vivas; mientras que la tasa de prevalencia es el número de prevalentes por cada 100.000 habitantes. Se calcula dividiendo el número de personas diagnosticadas en un periodo determinado que continúan viviendo, entre la población total de referencia. La prevalencia está relacionada, por tanto, con la supervivencia de cada tipo de cáncer: cuanto mayor sea la supervivencia mayor será el número de prevalentes.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown('')
    col1,col2,col3= st.columns(3)
    with col1:
        st.write("  Mapa de España con la insidencias total en 2022")
        st_folium(map1, returned_objects=[])
    with col2:
        st.write("  Mapa de España con la mortalidad total en 2022")
        st_folium(map2, returned_objects=[])
    with col3:
        st.write("  Mapa de España con la prevalencia total en 2022")
        st_folium(map3, returned_objects=[])

    #------------------------------------------------------TABS---------------------------------------------------#
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.title("Búsqueda de datos")
    st.markdown('')
    st.markdown('')
    st.markdown('')
    tabs = st.tabs(['Dimensiones y tasa media total del cáncer por CCAA',
                    'Dimensión de los tipos de cáncer (2019 y 2022)','Incidencia del cáncer de mama', 
                    'Incidencia del cáncer de próstata', 'Incidencia del cáncer colorrectal', 
                    'Incidencia del melanoma de piel', 'Incidencia del cáncer por grupo etarios',
                    'Incidencia del cáncer por sexo', 'Mortalidad de los tipos de cáncer',
                    'Evolución del cáncer en España'])

    # -------------------------------------------------------TAB 1-----------------------------------------------------#
    tab_plots = tabs[0]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        col1,col2= st.columns(2)
        with col1:
            
            # Hacemos un DataFrame con las CCAA y diemensión total
            dfin= df[df['Dimensión'] == 'Incidencia']
            dfina1= dfin[[ 'CCAA', 'Dimension total']]

            # Agrupar por CCAA y calcular la suma de 'Tasa total'
            di= dfina1.groupby('CCAA')['Dimension total'].sum()
            di= pd.DataFrame(di)
            # Hacer gráfico de dimensión total
            fig1 = px.bar(di, x=di.index, y='Dimension total', color= 'Dimension total',title = 'Dimensión total por CCAA',
            labels = {'CCAA': 'Unidad territorial', 'Media': 'Dimensión total del cáncer'})
            st.plotly_chart(fig1)
            
        with col2:
            
            # Hacer un DataFrame con CCAA y tasa total
            dfinpre = df[(df['Dimensión'] == 'Incidencia') | (df['Dimensión'] == 'Prevalencia')]
            datos_ccaa_tasa = dfinpre.loc[:, ["CCAA", "Tasa total"]]
            media_ccaa_tasa = datos_ccaa_tasa.groupby("CCAA")["Tasa total"].mean()
            media_ccaa_tasa = pd.DataFrame(media_ccaa_tasa)

            # Hacemos el gráfico de la tasa total media por CCAA
            fig2 = px.bar(media_ccaa_tasa, x=media_ccaa_tasa.index, y='Tasa total', color= 'Tasa total',title = 'Tasa total media por CCAA',
            labels = {'CCAA': 'Unidad territorial', 'Tasa total': 'Tasa total del cáncer'})
            st.plotly_chart(fig2)

    # -------------------------------------------------------TAB 2-----------------------------------------------------#
    tab_plots = tabs[1]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')
        
      
        col1,col2= st.columns(2)
        with col1:
            
            # Vamos a hacer un dataframe con incidencia y prevalencia del 2019
            dfinpre = df[(df['Dimensión'] == 'Incidencia') | (df['Dimensión'] == 'Prevalencia')]
            dfinpre19=dfinpre[dfinpre['Año']==2019]
            dftipo= dfinpre19[['Cerebro', 'Cervix',
                'Colorectal', 'Esofago', 'Estómago', 'Glándulas salivares', 'Hígado',
                'Hipofaringe', 'Labio, cavidad oral', 'Laringe', 'Leucemia',
                'Linfoma Hodgkin', 'Linfoma No-Hodgkin', 'Mama', 'Melanoma de piel',
                'Mesotelioma', 'Mieloma Multiple', 'Nasofaringe', 'Orofaringe',
                'Otros de piel', 'Ovario', 'Páncreas', 'Pene', 'Próstata', 'Pulmón',
                'Riñón', 'Sarcoma Kaposi', 'Testículo', 'Tiroides', 'Utero', 'Vagina',
                'Vejiga', 'Vesicula Biliar', 'Vulva']]
            dftipo.sum()
            tipo= pd.DataFrame(dftipo.sum())
            tipo.columns = ['Suma']

            # Creamos la gráfica del número total de todos los tipos de cáncer 

            fig3 = px.bar(tipo, x=tipo.index, y='Suma', color= 'Suma',title = 'Número de personas con tipos de cáncer (2019)',
                        labels = {'index': 'Tipos de cáncer', 'Suma': 'Dimensión del cáncer'})
            st.plotly_chart(fig3)
            
        with col2:

            # Vamos a hacer un dataframe sw incidencia y prevalencia para el año 2022
            dfinpre = df[(df['Dimensión'] == 'Incidencia') | (df['Dimensión'] == 'Prevalencia')]
            dfinpre22= dfinpre[dfinpre['Año']==2022]
            dftipo22= dfinpre22[['Cerebro', 'Cervix',
                'Colorectal', 'Esofago', 'Estómago', 'Glándulas salivares', 'Hígado',
                'Hipofaringe', 'Labio, cavidad oral', 'Laringe', 'Leucemia',
                'Linfoma Hodgkin', 'Linfoma No-Hodgkin', 'Mama', 'Melanoma de piel',
                'Mesotelioma', 'Mieloma Multiple', 'Nasofaringe', 'Orofaringe',
                'Otros de piel', 'Ovario', 'Páncreas', 'Pene', 'Próstata', 'Pulmón',
                'Riñón', 'Sarcoma Kaposi', 'Testículo', 'Tiroides', 'Utero', 'Vagina',
                'Vejiga', 'Vesicula Biliar', 'Vulva']]
            dftipo22.sum()
            tipo22= pd.DataFrame(dftipo22.sum())
            tipo22.columns = ['Suma']

            # Creamos la gráfica con el número total de cada tipo de cáncer
            fig4 = px.bar(tipo22, x=tipo22.index, y='Suma', color= 'Suma',title = 'Número de personas con tipos de cáncer (2022)',
                        labels = {'index': 'Tipos de cáncer', 'Suma': 'Dimensión del cáncer'})
            st.plotly_chart(fig4)

    # -------------------------------------------------------TAB 3-----------------------------------------------------#
    tab_plots = tabs[2]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        col1,col2= st.columns(2)
     
            
        with col1:
            # Vamos a hacer un dataframe Con CCAA y mama

            dfina2= dfin[[ 'CCAA', 'Mama']]

            # Sumamos todos los casos de incidencia por CCAA
            ma= dfina2.groupby('CCAA')['Mama'].sum()
            ma= pd.DataFrame(ma)

            # Creamos la gráfica

            fig5 = px.bar(ma, x=ma.index, y='Mama', color= 'Mama',title = 'Dimensión del cáncer de mama por CCAA',
                    labels = {'CCAA': 'Unidad territorial', 'Mama': 'Dimensión del cáncer'})
            st.plotly_chart(fig5)
        
        with col2:
            # Creamos el dataframe
            dfm1= dfin[(dfin['CCAA'] == 'Andalucía') | (dfin['CCAA'] == 'Cataluña')| (dfin['CCAA']== 'Comunidad de Madrid')]
            dfmm1= dfm1[[ 'CCAA', 'Mama']]

            # Agrupar por CCAA y cáncer de piel
            mam= dfmm1.groupby('CCAA')['Mama'].sum()
            mam= pd.DataFrame(mam)

            mam= mam.assign(Tasa_total=[299.72,283.72,284.43])

            fig16= px.bar(mam, x=mam.index, y='Tasa_total', color= 'Tasa_total',title = 'Tasa total del cáncer de mama por CCAA',
                labels = {'CCAA': 'Unidad territorial', 'Mama': 'Tasa del cáncer'})
            st.plotly_chart(fig16)  

    # -------------------------------------------------------TAB 4-----------------------------------------------------#
    tab_plots = tabs[3]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        col1,col2= st.columns(2)
       
            
        with col1:
            
            # Vamos a hacer un dataframe Con CCAA y próstata

            dfina3= dfin[[ 'CCAA', 'Próstata']]

            # Sumamos todos los casos de incidencia por CCAA
            pro= dfina3.groupby('CCAA')['Próstata'].sum()
            pro= pd.DataFrame(pro)
            
            # Creamos la gráfica

            fig6 = px.bar(pro, x=pro.index, y='Próstata', color= 'Próstata',title = 'Dimensión del cácer de próstata por CCAA',
                labels = {'CCAA': 'Unidad territorial', 'Próstata': 'Dimensión del cáncer'})
            st.plotly_chart(fig6)
        
        with col2:
            # Creamos el dataframe
            dfmm2= dfm1[[ 'CCAA', 'Próstata']]

            # Agrupar por CCAA y cáncer de piel
            pros= dfmm2.groupby('CCAA')['Próstata'].sum()
            pros= pd.DataFrame(pros)

            # Creamos gráfica
            pros= pros.assign(Tasa_total=[301.27,283.07,258.91])
            fig17 = px.bar(pros, x=pros.index, y='Tasa_total', color= 'Tasa_total',title = 'Tasa total del cáncer de próstata por CCAA',
                labels = {'CCAA': 'Unidad territorial', 'Próstata': 'Tasa del cáncer'})

            st.plotly_chart(fig17)

        

    # -------------------------------------------------------TAB 5-----------------------------------------------------#
    tab_plots = tabs[4] 
    with tab_plots: 
        st.markdown('')
        st.markdown('')
        st.markdown('')
    
        col1,col2= st.columns(2)
       
        with col1:
            
            # Vamos a hacer un dataframe Con CCAA y colorectal

            dfina5= dfin[[ 'CCAA', 'Colorectal']]
           
            # Sumamos todos los casos de incidencia por CCAA
            co= dfina5.groupby('CCAA')['Colorectal'].sum()
            co= pd.DataFrame(co)
            
            # Creamos la gráfica

            fig7 = px.bar(co, x=co.index, y='Colorectal', color= 'Colorectal',title = 'Dimensión del cácer coloorrectal por CCAA',
                labels = {'CCAA': 'Unidad territorial', 'Próstata': 'Dimensión del cáncer'})
            st.plotly_chart(fig7)
        
        with col2:
            # Creamos el dataframe

            dfmco= dfm1[[ 'CCAA', 'Colorectal']]

            # Agrupar por CCAA y cáncer de piel
            colo= dfmco.groupby('CCAA')['Colorectal'].sum()
            colo= pd.DataFrame(colo)

            colo= colo.assign(Tasa_total=[313.25,332.84,312.22])
            fig18= px.bar(colo, x=colo.index, y='Tasa_total', color= 'Tasa_total',title = 'Tasa total del cáncer colorrectal por CCAA',
                labels = {'CCAA': 'Unidad territorial', 'Tasa_total': 'Tasa del cáncer'})
            
            st.plotly_chart(fig18)
    # -------------------------------------------------------TAB 6-----------------------------------------------------#
    tab_plots = tabs[5]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')
    
        col1,col2= st.columns(2)
   
        with col1:
            
            # Vamos a hacer un dataframe Con CCAA y cáncer de piel

            dfina4= dfin[[ 'CCAA','Otros de piel']]

            # Sumamos todos los casos de incidencia por CCAA
            me= dfina4.groupby('CCAA')['Otros de piel'].sum()
            me= pd.DataFrame(me)
            
            # Creamos la gráfica

            fig8= px.bar(me, x=me.index, y='Otros de piel', color= 'Otros de piel',title = 'Dimensión del cácer de piel por CCAA',
                labels = {'CCAA': 'Unidad territorial', 'Otros de piel': 'Dimensión del cáncer'})
            st.plotly_chart(fig8)
        
        with col2:
            
            # Creamos el dataframe

            dfsn= dfin[(dfin['CCAA'] == 'Andalucía') | (dfin['CCAA'] == 'Galicia')| (dfin['CCAA'] == 'País Vasco')]
            dfnsp= dfsn[[ 'CCAA', 'Otros de piel']]

            # Agrupar por CCAA y cáncer de piel
            pi= dfnsp.groupby('CCAA')['Otros de piel'].sum()
            pi= pd.DataFrame(pi)

            pi= pi.assign(Tasa_total=[160.25,254.68,218.54])

            # Creamos la gráfica
            fig15 = px.bar(pi, x=pi.index, y='Tasa_total', color= 'Tasa_total',title = 'Tasa total en Andalucía y Galicia',
                        labels = {'CCAA': 'Unidad territorial', 'Tasa_total': 'Tasa del cáncer'})
            st.plotly_chart(fig15)


        # -------------------------------------------------------TAB 7-----------------------------------------------------#
    tab_plots = tabs[6]
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')
    
        col1,col2,col3= st.columns(3)
        with col1:
            st.write(" ")
            
        with col2:
            
            # Vamos a hacer un dataframe solo con la edad y su media de incidencia
            dfedad= in22[['0-39', '40-49', '50-59', '60-69', '+70']]
            dfedad.mean()
            edad= pd.DataFrame(dfedad.mean())
            edad.columns = ['Media']
            
            # Creamos la gráfica

            fig9= px.bar(edad, x=edad.index, y='Media', color= 'Media',title = 'Incidencia del cáncer por grupo etario',
                labels = {'index': 'Grupo etario', 'Media': 'Incidencia del cáncer'})
            st.plotly_chart(fig9)

            # Como ya sigue una normal podemos hacr un ANOVA

            # Dividir los datos
            jovenes = np.log(in22['0-39'])
            maduros= np.log(in22['40-49'])
            mayores= np.log(in22['50-59'])
            muymayores= np.log(in22['60-69'])
            ancianos = np.log(in22['+70'])

            #Realizar un ANOVA
            stat, p = f_oneway(jovenes,maduros,mayores,muymayores,ancianos)
            print('stat=%.3f, p=%.41f' % (stat, p))

            # Imprimir el resultado
            if p > 0.05:
                st.write(f'No hay una diferencia significativa en la incidencia de cáncer por edad (p={p:.41f}, t={stat:.4f}).')
                
            else:
                st.write(f'Hay una diferencia significativa en la incidencia de cáncer por edad (p={p:.41f}, t={stat:.4f}).')
        with col3:
            st.write(" ")


        
        # -------------------------------------------------------TAB 8-----------------------------------------------------#
    tab_plots = tabs[7]
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')
    
        col1,col2,col3= st.columns(3)
        with col1:
            st.write(" ")
            
        with col2:
            
           

            # Vamos a hacer un dataframe solo con los sexos y su media de incidencia

            dfsexo= in22[['Hombre','Mujer']]
            sexo= pd.DataFrame(dfsexo.mean())
            # Creamos la gráfica

            fig10 = px.pie(sexo, values=0, names=sexo.index, title='Porcentaje de incidencia de cáncer por sexo')
            st.plotly_chart(fig10)

            # Dividir los datos en dos muestras: hombres y mujeres
            hombres = np.log(in22['Hombre'])
            mujeres = np.log(in22['Mujer'])

            # Realizar una prueba t de dos muestras
            t, p = ttest_ind(hombres, mujeres)

            # Imprimir el resultado
            if p > 0.05:
                st.write(f'No hay una diferencia significativa en la incidencia de cáncer entre hombres y mujeres (p={p:.4f}, t={t:.4f}).')
            else:
                st.write(f'Hay una diferencia significativa en la incidencia de cáncer entre hombres y mujeres (p={p:.4f}, t={t:.4f}).')
        with col3:
            st.write(" ")
# -------------------------------------------------------TAB 9-----------------------------------------------------#
    tab_plots = tabs[8]
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')
    
        col1,col2,col3= st.columns(3)
        with col1:
            st.write(" ")
            
        with col2:
            
           

            # Vamos a hacer un dataframe 
            dfmo = df[(df['Dimensión'] == 'Mortalidad')]
            dfmotipo= dfmo[['Cerebro', 'Cervix',
                'Colorectal', 'Esofago', 'Estómago', 'Glándulas salivares', 'Hígado',
                'Hipofaringe', 'Labio, cavidad oral', 'Laringe', 'Leucemia',
                'Linfoma Hodgkin', 'Linfoma No-Hodgkin', 'Mama', 'Melanoma de piel',
                'Mesotelioma', 'Mieloma Multiple', 'Nasofaringe', 'Orofaringe',
                'Otros de piel', 'Ovario', 'Páncreas', 'Pene', 'Próstata', 'Pulmón',
                'Riñón', 'Sarcoma Kaposi', 'Testículo', 'Tiroides', 'Utero', 'Vagina',
                'Vejiga', 'Vesicula Biliar', 'Vulva']]
            dfmotipo.sum()
            tipomo= pd.DataFrame(dfmotipo.sum())
            tipomo.columns = ['Suma']

            # Creamos la gráfica

            fig11= px.bar(tipomo, x=tipomo.index, y='Suma', color= 'Suma',title = 'Número de muertes por tipos de cáncer',
                        labels = {'index': 'Tipos de cáncer', 'Media': 'Cantidad de muertes'})
            st.plotly_chart(fig11)
        with col3:
            st.write(" ")

        st.markdown('')
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>La letalidad de un tipo de cáncer se refiere a la proporción de personas diagnosticadas con ese cáncer que mueren como resultado de la enfermedad. El cálculo de la letalidad de un tipo de cáncer puede ser útil para evaluar la gravedad de la enfermedad y la eficacia de los tratamientos disponibles. Para calcular la letalidad de un tipo de cancer, se necesitan dos datos: </div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>1. El número de personas diagnosticadas con ese cáncer durante un período determinado.</div> """, unsafe_allow_html=True)
        
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>2. El número de personas que fallecieron debido a ese cáncer durante el mismo periodo.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Una vez que se tienen estos datos, se puede calcular la letalidad utilizando la siguiente fórmula:</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Letalidad = (Número de muertes por cáncer / Número de personas diagnosticadas con cáncer) x 100</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown('')

        col1,col2,col3= st.columns(3)
        with col1:
            st.write(" ")
            
        with col2:

            # Vamos a hacer un dataframe 
            dfinpre = df[(df['Dimensión'] == 'Incidencia') | (df['Dimensión'] == 'Prevalencia')]
            dftipo= dfinpre[['Cerebro', 'Cervix',
                'Colorectal', 'Esofago', 'Estómago', 'Glándulas salivares', 'Hígado',
                'Hipofaringe', 'Labio, cavidad oral', 'Laringe', 'Leucemia',
                'Linfoma Hodgkin', 'Linfoma No-Hodgkin', 'Mama', 'Melanoma de piel',
                'Mesotelioma', 'Mieloma Multiple', 'Nasofaringe', 'Orofaringe',
                'Otros de piel', 'Ovario', 'Páncreas', 'Pene', 'Próstata', 'Pulmón',
                'Riñón', 'Sarcoma Kaposi', 'Testículo', 'Tiroides', 'Utero', 'Vagina',
                'Vejiga', 'Vesicula Biliar', 'Vulva']]
            dftipo.sum()
            tipo= pd.DataFrame(dftipo.sum())
            tipo.columns = ['Suma']
            # Calculamos el porcentaje de muertes por tipo de cáncer. Letalidad = (Número de muertes por cáncer / Número de personas diagnosticadas con cáncer) x 100
            pormu= ((tipomo/tipo)*100).round(2)

            # Creamos la gráfica

            fig12 = px.bar(pormu, x=pormu.index, y='Suma', color= 'Suma',title = 'Letalidad de los tipos de cáncer',
                        labels = {'index': 'Tipos de cáncer', 'Suma': 'Letalidad (%)'})
            st.plotly_chart(fig12)
        with col3:
            st.write(" ")
# -------------------------------------------------------TAB 10-----------------------------------------------------#
    tab_plots = tabs[9]
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')
        col1,col2= st.columns(2)
        with col1:
            # Hacemos Dataframe
            dfina= dfin[[ 'Año', 'Tasa total']]

            # Agrupar por año y calcular la media de 'Tasa total'
            media_anual = dfina.groupby('Año')['Tasa total'].mean()
            media_anual= pd.DataFrame(media_anual)

            # Graficar la evolución del dataset
            fig13 = px.line(media_anual, x=media_anual.index, y='Tasa total', title='Tasa total de cáncer por año')
            fig13.update_layout(xaxis_tickangle=-90)
            st.plotly_chart(fig13)    
                    
        with col2:
       
            # Tomamos los mejores valores
            p = 1 # Coeficientes de autoregresión
            d = 2 # Orden de diferenciación
            q = 0 # Ajuste media móvil

            # Ajustamos modelo a datos
            model = ARIMA(media_anual, order=(p,d,q))
            model_fit = model.fit()

            # Tomamos predicciones
            preds = model_fit.predict()

            # Creamos figura
            fig14 = px.line(media_anual, x=media_anual.index, y='Tasa total', title='Tasa total de cáncer por año')
            fig14.add_scatter(x=media_anual.index[d:], y=preds[d:], mode='lines', name='Predicciones', line=dict(color='red', dash='dash'))
            fig14.update_layout(xaxis_tickangle=-90)
            st.plotly_chart(fig14)
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>ARIMA es un modelo que utiliza las observaciones anteriores, el error de predicción y la diferenciación para hacer predicciones en series de tiempo estacionarias.  Es útil para predecir valores en series de tiempo estacionarias, es decir, aquellos que no muestran cambios significativos en su medio o variación a lo largo del tiempo. </div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Prediciones futuras de los próximos tres años utilizando el modelo temporal ARIMA.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown('')
        col1,col2= st.columns(2)
        with col1:
            # Leer los datos en un DataFrame
            datos = pd.DataFrame({
                'Año': ['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01'],
                'Tasa Total': [623.331923, 632.226731, 638.918846, 647.947885]
            })

            # Convertir la columna 'Año' a tipo datetime y establecerla como índice
            datos['Año'] = pd.to_datetime(datos['Año'])
            datos.set_index('Año', inplace=True)

            # Crear el modelo ARIMA y ajustarlo a los datos
            modelo = ARIMA(datos, order=(1, 1, 1))
            modelo_fit = modelo.fit()

            # Hacer predicciones futuras
            preds = modelo_fit.predict(start='2020-01-01', end='2025-01-01', typ='levels')

            # Unir las predicciones con los datos originales
            prediccion = pd.concat([datos, preds], axis=0)

            # Graficar los datos y las predicciones
            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(prediccion.index, prediccion)
            ax.legend()
            ax.set_title('Tasa total por año')
            ax.set_xlabel('Año')
            ax.set_ylabel('Tasa Total')
            st.pyplot()
        
        with col2:
            st.write(" ")
    
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.title("Conclusiones")
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>1. Mirando las tasas medias veíamos que no había mucha diferencia entre las CCAA.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>2. Los cáncer más comunes son: de mama, de próstata, colorrectal y de piel.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>3. El cáncer de mama y próstata, mirando las CCAA con más población, se encuentran más en Andalucia y Cataluña.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>4. El cáncer de piel está más representado en el norte que en el sur.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>5. Cuanto mayor es la persona más posibilides tendrá de sufrir algún tipo de cáncer de piel.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>6. Los hombres son significativamente más propenso de sufrir cáncer.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>7. Los cáncer que producen más muertes al año son colorrectal y pulmón, pero los más letales son los de vesicula biliar y de páncreas.</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>8. La predicción futura en España es que el cáncer va aumentando poco a poco cada año.</div> """, unsafe_allow_html=True)
       
if __name__ == '__main__':
    main()