#########################################################################
# App de Analisis de comentarios reviews App
#########################################################################


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [A] Importacion de librerias
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Obtener versiones de paquetes instalados
# !pip list > requirements.txt

import streamlit as st

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
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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
# [B.7] Grafico + Grafo de reglas de asociacion 
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graficas_apriori(
  df_reglas_apriori,
  top_n_reglas_grafo = 50,
  top_n_reglas_scatter = 50,
  tipo_distribucion_grafo = 'kamada_kawai_layout', # kamada_kawai_layout, spring_layout, circular_layout
  factor_separacion_grafo = 3
  ):
  
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

  
  fig1 = px.scatter(
    df_reglas_apriori3.head(top_n_reglas_scatter),
    x='soporte',
    y='confianza',
    hover_name = 'regla',
    color = 'soporte',
    size = 'confianza'
  )
  
  
  df_reglas_apriori4 = df_reglas_apriori3.head(top_n_reglas_grafo)
    
  # Crear grafo dirigido
  Grafo = nx.DiGraph()
  
  # Contar la frecuencia de cada nodo (antecedente o consecuente)
  frecuencia_nodos = pd.Series(
    df_reglas_apriori4['antecedente'].explode()
    ).append(
      df_reglas_apriori4['consecuente'].explode()
      ).value_counts()
  
  # Añadir nodos al grafo
  for nodo, frecuencia in frecuencia_nodos.items():
    Grafo.add_node(nodo, frequency=frecuencia)

  # Añadir aristas con atributos de soporte y confianza
  for _, fila in df_reglas_apriori4.iterrows():
    Grafo.add_edge(
      fila['antecedente'], 
      fila['consecuente'], 
      support=fila['soporte'], 
      confidence=fila['confianza']
      )

  # Layout del grafo
  if tipo_distribucion_grafo=='kamada_kawai_layout': 
    pos = nx.kamada_kawai_layout(Grafo)
  elif tipo_distribucion_grafo=='spring_layout': 
    pos = nx.spring_layout(Grafo, seed=42,k=1,iterations=100)
  elif tipo_distribucion_grafo=='circular_layout': 
    pos = nx.circular_layout(Grafo)
  else:
    pos = nx.kamada_kawai_layout(Grafo)
    
  # Aumenta para más separación
  pos = {node: (x*factor_separacion_grafo,y*factor_separacion_grafo) for node, (x, y) in pos.items()}


  # Normalización para el color de las aristas
  valores_confianza = [Grafo.edges[borde]['confidence'] for borde in Grafo.edges]
  norm = mcolors.Normalize(vmin=min(valores_confianza), vmax=max(valores_confianza))
  cmap = cm.get_cmap('coolwarm')
  
  # Crear figura de Plotly
  fig2 = go.Figure()

  # Añadir nodos
  for nodo in Grafo.nodes:
    
    x, y = pos[nodo]
    tamano = Grafo.nodes[nodo]['frequency'] * 2
    fig2.add_trace(go.Scatter(
      x=[x], 
      y=[y],
      mode='markers+text',
      marker=dict(size=tamano, color='#636EFA', line=dict(width=2, color='DarkSlateGrey')),
      text=nodo,
      textposition='top center',
      hoverinfo='text',
      hovertext=f"Elemento: {nodo}<br>Frecuencia: {Grafo.nodes[nodo]['frequency']}"
    ))


  # Añadir aristas con flechas
  for borde in Grafo.edges:
    
    x0, y0 = pos[borde[0]]
    x1, y1 = pos[borde[1]]
    soporte = Grafo.edges[borde]['support']
    confianza = Grafo.edges[borde]['confidence']
    color = mcolors.to_hex(cmap(norm(confianza)))

    # Línea principal (sin flecha)
    fig2.add_trace(go.Scatter(
      x=[x0, x1], 
      y=[y0, y1],
      mode='lines',
      line=dict(width=soporte, color=color),
      hoverinfo='text',
      text=f"{borde[0]} → {borde[1]}<br>Soporte: {soporte:.2f}<br>Confianza: {confianza:.2f}"
      ))

    # Flecha al final de la arista
    fig2.add_annotation(
      x=x1, y=y1,
      ax=x0, ay=y0,
      xref='x', yref='y',
      axref='x', ayref='y',
      showarrow=True,
      arrowhead=3,
      arrowsize=1,
      arrowwidth=max(0.1,soporte * 100),
      arrowcolor=color
      )

  # Diseño final
  fig2.update_layout(
    title='Grafo de Reglas de Asociación',
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    margin=dict(l=10, r=10, t=40, b=10),
    plot_bgcolor='white'
  )
  
  return fig1,fig2




#=======================================================================
# [B.8] Entregable Global de reglas de asociacion
#=======================================================================



@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def entregables_apriori(
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
  
  fig1,fig2 = graficas_apriori(
    df_reglas_apriori = df_reglas_apriori,
    top_n_reglas_grafo = top_n_reglas_grafo,
    top_n_reglas_scatter= top_n_reglas_scatter,
    tipo_distribucion_grafo='kamada_kawai_layout',
    factor_separacion_grafo = 3
    )

  
  # retornar entregables 
  return fig1,fig2,df_reglas_apriori




#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [C] Generacion de la App
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://emojiterra.com/es/search/

st.set_page_config(layout='wide')

# titulo inicial 
st.markdown('# 📲 Analisis de reviews Apps')

# autoria 
st.markdown('**Autor 👉 [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')



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
  
  
  fig6, grafo6,df_reglas6 = entregables_apriori(
    df_input = df_scrapp5,
    filtro_App = tab6_App,
    filtro_SO = tab6_SO,
    filtro_calificacion = tab6_calificacion,
    top_n_reglas_scatter = 85,
    top_n_reglas_grafo = 25
    )

  
  with st.expander('Tabla con detalle de reglas', expanded=False):
    st.data_editor(
      df_reglas6, 
      use_container_width=True, 
      disabled=True,
      hide_index=True
      )
    
  with st.expander('Grafico de dispersion de reglas', expanded=False):
    st.plotly_chart(fig6)
    
  with st.expander('Grafo de reglas de asociacion', expanded=True):
    st.plotly_chart(grafo6)



# !streamlit run APP_Scrapp_Comentarios_Apps_V2.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/45_App_BenchMark WebScrap + IA (12-01-25)/App/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit


