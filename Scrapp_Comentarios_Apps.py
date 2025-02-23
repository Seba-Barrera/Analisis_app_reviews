#########################################################################
# App de Analisis de comentarios reviews App
#########################################################################


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [A] Importacion de librerias
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Obtener versiones de paquetes instalados
# !pip list > requirements.txt

import streamlit as st
from streamlit.components.v1 import html
import tempfile

# librerias para data
import pandas as pd
import numpy as np
from collections import Counter
import ast 

# librerias para graficos
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# librerias para otras visualizaciones 
from wordcloud import WordCloud # https://www.edureka.co/community/64068/installing-wordcloud-using-unable-getting-following-error
from pyvis.network import Network

# libreria para analitica textual
from mlxtend.frequent_patterns import apriori, association_rules


import warnings
warnings.filterwarnings('ignore')


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [B] Creacion de funciones internas utiles
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


#=======================================================================
# [B.0] LECTURA DE TABLA
#=======================================================================

# leer data
df_scrapp5 = pd.read_csv(
  'df_scrapp5.csv',
  sep=';',
  decimal=',',
  encoding='utf-8-sig'
)


#=======================================================================
# [B.1] Grafico de barra de nota promedio 
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graf_barra_nota(
  df_input,
  SO,
  App
  ):
  
  df = df_input[df_input['App'].isin(App)]
      
  df_agg = pd.concat([
    df[['calificacion','App','SO']],
    df[['calificacion','App']].assign(SO='Total')
    ]).groupby(['App','SO']).agg(    
      Conteo = pd.NamedAgg(column='calificacion',aggfunc=len),
      Prom = pd.NamedAgg(column='calificacion',aggfunc=np.mean),
      Desv = pd.NamedAgg(column='calificacion',aggfunc=np.std)
      ).reset_index()

  fig = px.bar(
    df_agg[df_agg['SO'].isin(SO)],
    x = 'App',
    y = 'Prom',
    color = 'SO',
    error_y= 'Desv',
    barmode='group'
  )
  
  return fig




#=======================================================================
# [B.2] Distribucion de clusters por nombre de app
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graf_barra_clusters(
  df_input,
  SO,
  App,
  N_grupos
  ):
  
  df_aux1 = df_input[df_input['SO'].isin(SO)]
  df_aux1 = df_aux1[df_aux1['App'].isin(App)]
  
  # Contar registros por App y categoría
  df_aux1 = df_aux1.groupby(
    ['App',f'NomC{N_grupos}']
    ).size().reset_index(name='conteo')
  
  df_aux1.columns = ['App','Agrupacion','conteo']

  # Calcular porcentaje dentro de cada App
  df_aux1['porcentaje'] = df_aux1.groupby('App')['conteo'].transform(
    lambda x: x / x.sum() 
    )

  # Crear gráfico de barras apilado (suma 100%)
  fig = px.bar(
    df_aux1, 
    x='App',
    y='porcentaje', 
    color='Agrupacion'
    )

  fig.update_layout(barmode='stack', yaxis=dict(tickformat='.0%'))  # Apila y formatea %
  
  return fig






#=======================================================================
# [B.3] Insertar saltos de linea
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def insertar_saltos_linea(
  texto, 
  x,
  caracter_salto = "<br>"  
  ):
  palabras = texto.split()  # Divide el texto en palabras
  lineas = [' '.join(palabras[i:i + x]) for i in range(0, len(palabras), x)]
  return caracter_salto.join(lineas)  # Une las líneas con saltos de línea


#=======================================================================
# [B.4] Grafico en el espacio 2D/3D de distribucion de comentarios
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graf_scatter(
  df_input,
  dimensiones,
  tercer_eje,
  color,
  forma,
  tamaño,
  filtro_App,
  filtro_SO
  ):
  

  dic_color = dict(zip(
    ['Segmentacion'+str(i) for i in range(4,11)]+['App','calificacion','SO'],
    ['NomC'+str(i) for i in range(4,11)]+['App','calificacion','SO']
  ))
  
  dic_tercer_eje = dict(zip(
    ['Segmentacion'+str(i) for i in range(4,11)]+['Dim3','App','calificacion','SO'],
    ['NomC'+str(i) for i in range(4,11)]+['Dim3','App','calificacion','SO']
  ))
  

  # crear df filtrado segun opciones de filtros (SO, App)
  df_scatter = df_input.loc[
    (df_input['App'].isin(filtro_App)) & 
    (df_input['SO'].isin(filtro_SO))
    ]

  df_scatter['texto'] = df_scatter['comentario'].apply(
    lambda x: insertar_saltos_linea(x,6)
    )
  
  df_scatter = df_scatter.rename(columns={
    'TSNE1': 'Dim1',
    'TSNE2': 'Dim2',
    'TSNE3': 'Dim3',
    dic_color[color]: color,
    dic_tercer_eje[tercer_eje]: tercer_eje
  })


  if dimensiones == '2D':
    
    codigo_fig = f'px.scatter(df_scatter,x = "Dim1",y = "Dim2",color ="{color}",hover_name= "texto"'
       
  else:
    
    codigo_fig = f'px.scatter_3d(df_scatter,x = "Dim1",y = "Dim2",z = "{tercer_eje}",color ="{color}",hover_name= "texto"'
  
  
  
  if forma != 'N.A.':
    codigo_fig+=',symbol = "SO"'
    
  if tamaño != 'N.A.':
    codigo_fig+=',size = "calificacion"'
  
  codigo_fig+=')'

  #print(codigo_fig)
  fig = eval(codigo_fig)
  #exec(codigo_fig)
  
  return fig




#=======================================================================
# [B.5] Nubes de palabras muy aperturados 
#=======================================================================


@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graf_nube(
  df_input,
  filtro_App,
  filtro_SO,
  filtro_calificacion,
  filtro_tipos_palabra,
  segmentacion,
  filtro_palabras_puntuales = '' # deben ir separadas por coma
  ):
  
  dic_segmentacion = dict(zip(
    ['Segmentacion'+str(i) for i in range(4,11)],
    ['NomC'+str(i) for i in range(4,11)]
  ))
  
  df_nube = df_input.loc[
    (df_input['App'].isin(filtro_App)) & 
    (df_input['SO'].isin(filtro_SO)) & 
    (df_input['calificacion'].isin(filtro_calificacion)) 
  ]
  
  df_nube['comentario2'] = df_nube['comentario2'].apply(ast.literal_eval)
  df_nube['tipos_palabra'] = df_nube['tipos_palabra'].apply(ast.literal_eval)

  df_nube2 = df_nube[[
    'App',
    dic_segmentacion[segmentacion],
    'comentario2',
    'tipos_palabra'  
    ]].explode(
      ['comentario2','tipos_palabra']
      )
  
  df_nube3 = df_nube2[df_nube2['tipos_palabra'].isin(filtro_tipos_palabra)]
  df_nube3.columns = ['App','Segmento','Palabra','Tipo_palabra']
    
  # quitar palabras especificas 
  df_nube3 = df_nube3[
    ~df_nube3['Palabra'].isin(filtro_palabras_puntuales.split(','))
    ]
 
  # Obtener las Apps y Segmentos únicos
  apps = df_nube3['App'].unique()
  segmentos = df_nube3['Segmento'].unique()

  # Crear una figura con subplots
  fig, axes = plt.subplots(len(segmentos)+1, len(apps)+1, figsize=(18, 10))

  # Generar nubes de palabras para cada combinación (App, Segmento)
  for i, segmento in enumerate(segmentos):
    for j, app in enumerate(apps):
      ax = axes[i, j] if len(segmentos) > 1 else axes[j]  # Manejo de 1 fila/columna
      
      # Filtrar palabras para esta combinación de App y Segmento
      palabras = df_nube3[
        (df_nube3['App'] == app) & 
        (df_nube3['Segmento'] == segmento)
        ]['Palabra']

      # ax.set_title(f'{app} - {segmento}', fontsize=10)
      ax.axis('off')  # Quitar ejes

      if not palabras.empty:
        texto = ' '.join(palabras)  # Convertir palabras a un solo string
        wordcloud = WordCloud(background_color='white', colormap='viridis').generate(texto)
        ax.imshow(wordcloud, interpolation='bilinear')

  # Nube total por cada fila (Segmento)
  for i, segmento in enumerate(segmentos):
    ax = axes[i, -1]  # Última columna
    palabras = df_nube3[
      (df_nube3['Segmento'] == segmento)
      ]['Palabra']
    ax.axis('off')

    if not palabras.empty:
      texto = ' '.join(palabras)  # Convertir palabras a un solo string
      wordcloud = WordCloud(background_color='white', colormap='cool').generate(texto)
      ax.imshow(wordcloud, interpolation='bilinear')


  # Nube total por cada columna (App)
  for j, app in enumerate(apps):
    ax = axes[-1, j]  # Última fila
    palabras = df_nube3[
      (df_nube3['App'] == app)
      ]['Palabra']
    ax.axis('off')

    if not palabras.empty:
      texto = ' '.join(palabras)  # Convertir palabras a un solo string
      wordcloud = WordCloud(background_color='white', colormap='cool').generate(texto)
      ax.imshow(wordcloud, interpolation='bilinear')


  # Nube total de todo el DataFrame (Esquina inferior derecha)
  ax = axes[-1, -1]
  ax.axis('off')
  texto = ' '.join(df_nube3['Palabra'])
  wordcloud = WordCloud(background_color='white', colormap='inferno').generate(texto)
  ax.imshow(wordcloud, interpolation='bilinear')


  # Nombre de columnas 
  for ax, c in zip(axes[0], range(len(apps)+1)):
    ax.set_title(
      (list(apps)+['Total'])[c],
      fontsize = 12
      )

  # nombre de filas 
  for ax, f in zip(axes[:,0], range(len(segmentos)+1)):
    ax.text(
      -90,
      100,
      insertar_saltos_linea((list(segmentos)+['Total'])[f],3,'\n'),
      rotation=0,
      ha = 'right',
      va = 'bottom',
      fontsize = 12
    )


  # retornar entregable 
  return fig




#=======================================================================
# [B.6] Tabla de reglas de asociacion usando apriori entre palabras
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def generar_df_apriori(
  df_input,
  filtro_App,
  filtro_SO,
  filtro_calificacion,
  min_soporte = 0.005,
  min_confianza = 0.05
  ):
  
  # crear copia agregando id
  df_apriori = df_input.copy()
  df_apriori['id'] = range(1,1+len(df_apriori))

  df_apriori['comentario2'] = df_apriori['comentario2'].apply(ast.literal_eval)
  df_apriori['tipos_palabra'] = df_apriori['tipos_palabra'].apply(ast.literal_eval)

  # generar explode de palabras 
  df_apriori = df_apriori.loc[
    (df_apriori['App'].isin(filtro_App)) & 
    (df_apriori['SO'].isin(filtro_SO)) & 
    (df_apriori['calificacion'].isin(filtro_calificacion)) 
    ,[
      'id',
      'comentario2',
      'tipos_palabra'  
      ]].explode(
      ['comentario2','tipos_palabra']
      )
      
  df_apriori['valor']=1


  # limpiar df antes de continuar
  df_apriori['comentario2'] = df_apriori['comentario2'].astype(str)
  df_apriori['exluir'] = df_apriori['comentario2'].apply(
    lambda x: 1 if len(x)<3 or any(c in x for c in ['.','/','$','-',':','%','1','2','3','4','5','6','7','8','9','0']) 
              else 0
    )

  # pivotear para formato de apriori
  df_apriori2 = df_apriori[
    (df_apriori['tipos_palabra'].isin([
      'adjetivo',
      'sustantivo',
      'verbo',
      'adverbio',
      ])) & 
    (df_apriori['exluir']==0) 
    ].pivot_table(
      index='id', 
      columns='comentario2', 
      values='valor', 
      fill_value=0
      )

  # generar frecuencia
  frecuencia_items = apriori(
    df_apriori2, 
    min_support=min_soporte, 
    use_colnames=True
    )

  # generar reglas
  reglas = association_rules(
    frecuencia_items,
    metric = 'confidence',
    min_threshold=min_confianza
  )

  # generar df de reglas 
  df_reglas = pd.DataFrame({
    'antecedente': [list(x) for x in reglas['antecedents']],
    'consecuente': [list(x) for x in reglas['consequents']],
    'items_antecedente': [len(list(x)) for x in reglas['antecedents']],
    'items_consecuente': [len(list(x)) for x in reglas['consequents']],
    'soporte_antecedente': reglas['antecedent support'],
    'soporte_consecuente': reglas['consequent support'],
    'soporte': reglas['support'],
    'confianza': reglas['confidence'],
    'lift': reglas['lift'],
    'leverage': reglas['leverage'],
    'conviccion': reglas['conviction']
    }).sort_values(by='soporte', ascending = False)
  
  return df_reglas





#=======================================================================
# [B.7] Asignar lista de colores segun lista de valores 
#=======================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def num2color(
  lista_numeros,
  lista_colores
  ):
  
  ind_colores = np.interp(
    lista_numeros,
    [min(lista_numeros),max(lista_numeros)],
    [0,len(lista_colores)]
    )
  
  entregable = [
    lista_colores[int(np.floor(x))-1] for x in ind_colores
    ]
  
  return entregable



#=======================================================================
# [B.8] Grafico + Grafo de reglas de asociacion 
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graficar_apriori(
  df_input,
  filtro_App,
  filtro_SO,
  filtro_calificacion,
  top_n_reglas_scatter = 80,
  top_n_reglas_grafo = 40
  ):  
  
  # generar df de apriori 
  df_reglas_apriori = generar_df_apriori(
    df_input = df_input,
    filtro_App = filtro_App,
    filtro_SO = filtro_SO,
    filtro_calificacion = filtro_calificacion,
    min_soporte = 0.005,
    min_confianza = 0.05
    )
      
  df_reglas_apriori2 = df_reglas_apriori[
    ['antecedente','consecuente','soporte','confianza']
    ]

  # quitar reglas reciprocas quedandose con la de mayor confianza
  df_reglas_apriori2['regla_comb'] = df_reglas_apriori2.apply(
    lambda x: frozenset(set(x['antecedente']) | set(x['consecuente'])), 
    axis=1
    )
  
  df_reglas_apriori3 = df_reglas_apriori2.loc[
    df_reglas_apriori2.groupby('regla_comb')['confianza'].idxmax()
    ].drop(columns=['regla_comb']).sort_values(
      by='soporte',ascending = False
      ).reset_index(drop=True)
  

  df_reglas_apriori3['antecedente'] = df_reglas_apriori3['antecedente'].apply(
    lambda x: '+'.join(x)
  )
  df_reglas_apriori3['consecuente'] = df_reglas_apriori3['consecuente'].apply(
    lambda x: '+'.join(x)
  )
  df_reglas_apriori3['regla'] = df_reglas_apriori3.apply(
    lambda x: x['antecedente']+' -> '+x['consecuente'],
    axis=1
  )
        
  df_reglas_apriori3['soporte'] = df_reglas_apriori3['soporte'].apply(lambda x: round(x,4))
  df_reglas_apriori3['confianza'] = df_reglas_apriori3['confianza'].apply(lambda x: round(x,4))
  
  
  fig = px.scatter(
    df_reglas_apriori3.head(top_n_reglas_scatter),
    x='soporte',
    y='confianza',
    hover_name = 'regla',
    color = 'soporte',
    size = 'confianza'
  )


  # crear df de nodos
  df_reglas_apriori4 = df_reglas_apriori3.head(top_n_reglas_grafo)
    
  df_nodo = pd.DataFrame(
    Counter(pd.concat([
      df_reglas_apriori4['antecedente'],df_reglas_apriori4['consecuente']
    ])).items(),
    columns = ['Item','Frecuencia']
  )
  
  # crear id para nodo
  df_nodo['id_nodo'] = range(1,1+len(df_nodo))
  
  # agregar color segun frecuencia
  df_nodo['color'] = num2color(
    lista_colores = px.colors.sequential.Viridis,
    lista_numeros=df_nodo['Frecuencia']
  )
  
  # incorporar id_nodo a df de reglas
  df_conexiones = df_reglas_apriori4.merge(
    df_nodo[['Item','id_nodo']],
    left_on = 'antecedente',
    right_on='Item'
    ).rename(
      columns = {'id_nodo':'id_nodo_a'}
    ).drop('Item',axis=1).merge(
      df_nodo[['Item','id_nodo']],
      left_on = 'consecuente',
      right_on='Item'
    ).rename(
      columns = {'id_nodo':'id_nodo_c'}
    ).drop('Item',axis=1)
  
  # agregar color segun confianza
  df_conexiones['color'] = num2color(
    lista_colores = px.colors.sequential.Plotly3[::-1],
    lista_numeros = df_conexiones['confianza']
    )
  
  # crear un grafo 
  grafo = Network(
    notebook = True,
    directed = True,
    cdn_resources = 'remote'
    )

  # agregar nodos
  for i in range(len(df_nodo)):
    grafo.add_node(
      df_nodo.loc[i,'id_nodo'].item(),
      label = df_nodo.loc[i,'Item'],
      value = df_nodo.loc[i,'Frecuencia'].item(),
      color = df_nodo.loc[i,'color']
      )


  # agregar conexiones 
  for i in range(len(df_conexiones)):
    grafo.add_edge(
      df_conexiones.loc[i,'id_nodo_a'].item(),
      df_conexiones.loc[i,'id_nodo_c'].item(),
      value = df_conexiones.loc[i,'confianza'].item(),
      title = 'soporte: '+str(round(df_conexiones.loc[i,'soporte'].item(),2)),
      color = df_conexiones.loc[i,'color']
      )

  # configurar mejor diseño de grafo
  grafo.repulsion(
    node_distance = 200,
    central_gravity = 0.3,
    spring_length = 200,
    spring_strength = 0.02,
    damping = 0.05
    )

  # guardar grafo
  # grafo.save_graph(nombre_grafo)
  
  # retornar entregables 
  return fig,grafo,df_reglas_apriori





#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [C] Generacion de la App
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://emojiterra.com/es/search/

st.set_page_config(layout='wide')

# titulo inicial 
st.markdown('# 📲 Analisis de reviews Apps')

# autoria 
st.markdown('**Autor :point_right: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')





# Crear tres tabs
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
  '📋 Detalle Reseñas', 
  '📈 Calificaciones',
  '📊 Distribucion Cluster', 
  '🧮 Reseñas en el Espacio', 
  '☁️ Nube de Palabras',
  '📐 Reglas de asociacion'
  ])




#_____________________________________________________________________________
# 1. Detalle Reseñas
    

with tab1:  
  
  lista_cols1 = [
    'usuario', 
    'fecha', 
    'comentario',
    'calificacion', 
    'SO', 
    'link', 
    'App', 
    'NomC4', 
    'NomC5',
    'NomC6',
    'NomC7',
    'NomC8',
    'NomC9',
    'NomC10' 
    ]

  st.data_editor(
    df_scrapp5[lista_cols1].rename(
      columns=dict(zip(lista_cols1,[
        'Usuario',
        'Fecha',
        'Reseña',
        'Calificacion',
        'Sistema Operativo',
        'Link',
        'Aplicacion',
        'Cluster(N=4)',
        'Cluster(N=5)',
        'Cluster(N=6)',
        'Cluster(N=7)',
        'Cluster(N=8)',
        'Cluster(N=9)',
        'Cluster(N=10)'
        ]))), 
    use_container_width=True, 
    disabled=True,
    hide_index=True
    )



#_____________________________________________________________________________
# 2. Calificaciones


with tab2:  
  
  tab2_col1,tab2_col2 = st.columns([1,4])  
  
  tab2_SO = tab2_col1.multiselect(
    'Sistema Operativo:', 
    sorted(list(set(df_scrapp5['SO']))),
    default = sorted(list(set(df_scrapp5['SO']))),
    key='tab2_SO'
    )
  
  tab2_App = tab2_col2.multiselect(
    'Aplicacion:', 
    sorted(list(set(df_scrapp5['App']))),
    default = sorted(list(set(df_scrapp5['App']))),
    key='tab2_App'
    )

  fig2 = graf_barra_nota(
    df_input = df_scrapp5,
    SO = tab2_SO,
    App = tab2_App
    )

  st.plotly_chart(fig2)



#_____________________________________________________________________________
# 3. Distribucion Cluster (grafico de barras apiladas)
    

with tab3:  
  
  tab3_col1,tab3_col2,tab3_col3 = st.columns([1,4,1])  
  
  tab3_SO = tab3_col1.multiselect(
    'Sistema Operativo:', 
    sorted(list(set(df_scrapp5['SO']))),
    default = sorted(list(set(df_scrapp5['SO']))),
    key='tab3_SO'
    )
  
  tab3_App = tab3_col2.multiselect(
    'Aplicacion:', 
    sorted(list(set(df_scrapp5['App']))),
    default = sorted(list(set(df_scrapp5['App']))),
    key='tab3_App'
    )
  
  tab3_Nclusters = tab3_col3.slider(
    'Selecciona Cantidad de Agrupaciones:', 
    min_value=4, 
    max_value=10, 
    value=5, 
    step=1,
    key='tab3_Nclusters'
    )

  fig3 = graf_barra_clusters(
    df_input = df_scrapp5,
    SO = tab3_SO,
    App =  tab3_App,
    N_grupos = tab3_Nclusters
    )

  st.plotly_chart(fig3)




#_____________________________________________________________________________
# 4. Grafico de Reseñas (dispersion 2D/3D)
 

with tab4:  
  
  tab4_col1a,tab4_col2a = st.columns([1,4])  
  
  tab4_SO = tab4_col1a.multiselect(
    'Sistema Operativo:', 
    sorted(list(set(df_scrapp5['SO']))),
    default = sorted(list(set(df_scrapp5['SO']))),
    key='tab4_SO'
    )
  
  tab4_App = tab4_col2a.multiselect(
    'Aplicacion:', 
    sorted(list(set(df_scrapp5['App']))),
    default = sorted(list(set(df_scrapp5['App']))),
    key='tab4_App'
    )
  
  
  tab4_col1b,tab4_col2b,tab4_col3b,tab4_col4b,tab4_col5b = st.columns([1,1,1,1,1])  
  
  tab4_dims = tab4_col1b.radio(
    label='Cantidad de dimensiones:', 
    options=['2D','3D'], 
    horizontal=True,
    key='tab4_dims'
    )
  
  tab4_tercer_eje = tab4_col2b.selectbox(
    label='Variable en tercer eje:', 
    options = ['Segmentacion'+str(i) for i in range(4,11)]+['Dim3','App','calificacion','SO'],
    index=7,
    key = 'tab4_tercer_eje'    
    )
  
  
  tab4_color = tab4_col3b.selectbox(
    label='Color:', 
    options = ['Segmentacion'+str(i) for i in range(4,11)]+['App','calificacion','SO'],
    index = 9,
    key = 'tab4_color'    
    )
  
  tab4_forma = tab4_col4b.radio(
    label='Forma:', 
    options=['N.A.','SO'], 
    horizontal=True,
    key='tab4_forma'
    )
  
  tab4_tamano = tab4_col5b.radio(
    label='Tamaño:', 
    options=['N.A.','calificacion'], 
    horizontal=True,
    key='tab4_tamano'
    )
  

  fig4 = graf_scatter(
    df_input = df_scrapp5,
    dimensiones = tab4_dims,
    tercer_eje = tab4_tercer_eje, # ['Segmentacion'+str(i) for i in range(4,11)]+['Dim3','App','calificacion','SO']
    color = tab4_color, # ['Segmentacion'+str(i) for i in range(4,11)]+['App','calificacion','SO']
    forma = tab4_forma, # ['SO','N.A.']
    tamaño = tab4_tamano, # ['calificacion','N.A.']
    filtro_App = tab4_App,
    filtro_SO = tab4_SO
    )

  st.plotly_chart(fig4)


#_____________________________________________________________________________
# 5. Nube de Palabras


with tab5:  
  
  tab5_col1a,tab5_col2a,tab5_col3a = st.columns([1,4,1])  
    
  tab5_SO = tab5_col1a.multiselect(
    'Sistema Operativo:', 
    sorted(list(set(df_scrapp5['SO']))),
    default = sorted(list(set(df_scrapp5['SO']))),
    key='tab5_SO'
    )
  
  tab5_App = tab5_col2a.multiselect(
    'Aplicacion:', 
    sorted(list(set(df_scrapp5['App']))),
    default = sorted(list(set(df_scrapp5['App']))),
    key='tab5_App'
    )
  
  tab5_calificacion = tab5_col3a.multiselect(
    'Calificacion:', 
    sorted(list(set(df_scrapp5['calificacion']))),
    default = sorted(list(set(df_scrapp5['calificacion']))),
    key='tab5_calificacion'
    )
  
  
  tab5_col1b,tab5_col2b,tab5_col3b = st.columns([1,5,1]) 

  tab5_segmentacion = tab5_col1b.selectbox(
    label='Segmentacion:', 
    options = ['Segmentacion'+str(i) for i in range(4,11)],
    index = 0,
    key = 'tab5_segmentacion'    
    )
  
  opciones_tipos_palabra = df_scrapp5['tipos_palabra'].apply(ast.literal_eval).explode().dropna().unique().tolist()
  
  tab5_tipos_palabra = tab5_col2b.multiselect(
    'Tipos de Palabra:', 
    opciones_tipos_palabra,
    default = opciones_tipos_palabra,
    key='tab5_tipos_palabra'
    )
  
  tab5_palabras_puntuales = tab5_col3b.text_input(
    label='Palabras a quitar:',
    value='',
    placeholder = 'Ej: app,banco,aplicación',
    key='tab5_palabras_puntuales'
    )


  fig5 = graf_nube(
    df_input = df_scrapp5,
    filtro_App = tab5_App,
    filtro_SO = tab5_SO,
    filtro_calificacion = tab5_calificacion,
    filtro_tipos_palabra = tab5_tipos_palabra,
    segmentacion = tab5_segmentacion, # ['Segmentacion'+str(i) for i in range(4,11)]
    filtro_palabras_puntuales=tab5_palabras_puntuales # deben ir separadas por coma
    )

  st.pyplot(fig5)




#_____________________________________________________________________________
# 6. Reglas de asociacion (detalle de df + scatter + grafo)
 

with tab6:  

  tab6_col1a,tab6_col2a,tab6_col3a = st.columns([1,4,1])  
    
  tab6_SO = tab6_col1a.multiselect(
    'Sistema Operativo:', 
    sorted(list(set(df_scrapp5['SO']))),
    default = sorted(list(set(df_scrapp5['SO']))),
    key='tab6_SO'
    )
  
  tab6_App = tab6_col2a.multiselect(
    'Aplicacion:', 
    sorted(list(set(df_scrapp5['App']))),
    default = sorted(list(set(df_scrapp5['App']))),
    key='tab6_App'
    )
  
  tab6_calificacion = tab6_col3a.multiselect(
    'Calificacion:', 
    sorted(list(set(df_scrapp5['calificacion']))),
    default = sorted(list(set(df_scrapp5['calificacion']))),
    key='tab6_calificacion'
    )


  fig6, grafo6,df_reglas6 = graficar_apriori(
    df_input = df_scrapp5,
    filtro_App = tab6_App,
    filtro_SO = tab6_SO,
    filtro_calificacion = tab6_calificacion,
    top_n_reglas_scatter = 90,
    top_n_reglas_grafo = 50
    )

  with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as archivo_temporal:
    grafo6.save_graph(archivo_temporal.name)
    ruta_archivo_temporal = archivo_temporal.name
  
  with open(ruta_archivo_temporal, 'r', encoding='utf-8') as archivo:
    grafo_html = archivo.read()
  
  
  
  
  with st.expander('Tabla con detalle de reglas', expanded=False):
    st.data_editor(
      df_reglas6, 
      use_container_width=True, 
      disabled=True,
      hide_index=True
      )
    
  with st.expander('Grafico de dispersion de reglas', expanded=False):
    st.plotly_chart(fig6)
    
  # with st.expander('Grafo de reglas de asociacion', expanded=True):
    # st.components.v1.html(grafo_html, height=500, scrolling=True)



# !streamlit run Scrapp_Comentarios_Apps_V3.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/45_App_BenchMark WebScrap + IA (12-01-25)/App/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit




