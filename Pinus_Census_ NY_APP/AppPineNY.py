import pandas as pd
from json import load
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide") #Faz a visualização dos textos e figuras se estenderem para o tamanho da tela


################################ Carregando os dados ################################
#Base de dados das árvores
temp_df = pd.read_csv('pine_NY_2015.csv')

#Mapa de bairros de NY
f = open('NTA_map.geojson')
nta_json = load(f)
for i in list(range(0,len(nta_json['features']))):
    json_id=nta_json['features'][i]['properties']['ntacode']
    nta_json['features'][i]['id']=json_id

    
################################ Funções de gráfico ################################
#PlotlyTAble
def PotlyTable (df):
    fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns),
                                               fill_color = '#636efa',
                                               align='center'),
                                   cells=dict(values=[df[col] for col in list(df.columns.values) if True],
                                              fill_color='#9E9B9B',
                                              align='center',
                                              height=30))])
    return fig
#PieChartPlotly

def PieChartPlotly (df,col):
    #df = Dataset
    #col = Coluna para agrupar os dados
    countCol = df.groupby(by=col).size()
    labels = list(countCol.index)
    values = list(countCol.values)
    figPie = go.Figure(data=[go.Pie(labels=labels,
                                    values=values,
                                    textinfo='label+percent+value',
                                    insidetextorientation='radial')
                            ])
    return figPie

#BoxPlotPlotly
def BoxPlotPlotly(df,col,colValue):
    #df = Dataset
    #col = Coluna para agrupar os dados
    #colValue = Coluna dos valores que serão distribuidos
    data=[]
    for t in df[col].value_counts().index:
        data.append(go.Box(y=df[colValue].loc[df[col]==t],
                       name = t))
    BoxPlotfig = go.Figure(data=data)
    return BoxPlotfig

#ChoropletEspPlotly
def ChoropletEspPlotly(df,col_count,option,georef,colref,name):
    #df = Dataset
    #col_count = ,option,georef,colref,name
    ChoropletGRaf = go.Choroplethmapbox(geojson=georef, 
                                locations=df[colref].loc[df[col_count]==option].value_counts().index, 
                                z=df[colref].loc[df[col_count]==option].value_counts().values,
                                colorbar=dict(title=name),
                                colorscale="YlGn",marker_opacity=0.5
                                )
    return ChoropletGRaf

#ScatterMapPotly 
def ScatterMapPotly(df,option,colOption,label):
    Scatter = go.Scattermapbox(lat=df.loc[df[colOption]==option]['latitude'],
                           lon=df.loc[df[colOption]==option]['longitude'],
                           mode='markers',
                           marker=go.scattermapbox.Marker(size=5,color='black'),
                           hovertext=df.loc[df[colOption]==option][label])
    return Scatter
################################Iniciando as vizializações################################

#Começar com o contexto dos dados
#Adicionando um título

st.markdown("# Relatório do senso das árvores de pinus de na cidade de Nova York em 2015")
st.markdown("Esse conjunto de dados inclui um registro para cada árvore na cidade de Nova York e inclui a localização da árvore por bairro e latitude/longitude, espécies por nome latino e nomes comuns, tamanho e  saúde. O censo de árvores de 2015 foi conduzido pela equipe da 'NYC Parks and Recreation, TreesCount!', equipe do programa e centenas de voluntários.")
st.markdown("**Datasets**")
st.markdown("[[New York NTA]](**Cr%C3%A9ditos**%20-%20%5Bhttps://data.cityofnewyork.us/City-Government/NTA-map/d3qk-pfyz%5D) - [[Tree Census in New York City]](https://www.kaggle.com/datasets/nycparks/tree-census)")

st.subheader('Processando dados originais')
code = '''
df_2015=pd.read_csv('new_york_tree_census_2015.csv')
df_2015[['spc_common','tree_dbh', 'stump_diam','status','health','problems','problems', 'root_stone','root_grate', 'root_other','nta','latitude','longitude']].head()
temp_df=df_2015.dropna().loc[df_2015.dropna()['spc_common'].str.contains('pine')][['spc_common','tree_dbh', 'stump_diam','status','health','problems','problems', 'root_stone','root_grate', 'root_other','nta','latitude','longitude']]
'''
st.code(code, language='python')

st.markdown("# Resumo dos dados")

        
rowstable = st.slider('Quantas linhas quer explorar?', 10, 30, 10)

#Tabela com amostra dos dados    
simpletable = temp_df.loc[:,['spc_common','tree_dbh','stump_diam','health','nta','latitude','longitude']].head(rowstable)

PotlyTableSt = PotlyTable(simpletable).update_layout(margin={"r":10,"t":50,"l":0,"b":0},title='Amostra do censo de Espécies de Pinus em NY')
st.plotly_chart(PotlyTableSt,use_container_width=True)



####Visualização de PieChart e BoxPlot
#Gráfico de pizza geral
figPie = PieChartPlotly(temp_df,"spc_common").update_layout(margin={"r":0,"t":50,"l":0,"b":0}, title='Percentual de Espécies de Pinus em N.Y.')
#Gráfico BoxPlot de DAP das espécies
BoxPlotfig = BoxPlotPlotly(df = temp_df, col = 'spc_common',colValue = 'tree_dbh').update_layout(title="Box-plot DAP por Espécie",yaxis=dict( title='DAP (cm)'),margin={"r":0,"t":50,"l":0,"b":0})


####Montando as visualizações dos gráficos de  Box-Plot e Choroplet de espécie
col1_1, col1_2 = st.columns(2)
with col1_1: 
    st.plotly_chart(figPie, use_container_width=True)
with col1_2: 
    st.plotly_chart(BoxPlotfig, use_container_width=True)



####Montando uma lista das espécies de pinus
esp = [t for t in temp_df['spc_common'].value_counts().index if True]


st.markdown("# Dados por espécie")

####Montando os inputs de filtro dos gráficos de espécie gráfico
col2_1, col2_2 = st.columns([2.3, 1]) #Estruturando a visualização
with col2_1: #Caixa de seleção de espécie
    option = st.selectbox('Qual esptécie você quer analisar?',(esp))

with col2_2: #Slider de DAP da espécie
    tree_dbh_slice = st.slider('Verificar DAP\'s maiores que;', 0, int(temp_df.loc[temp_df['spc_common']==option].tree_dbh.max()), 1)

#Gráfico Choroplet de espécie
df_filter = temp_df.loc[temp_df['tree_dbh']>=tree_dbh_slice]
Choroplet = ChoropletEspPlotly(df = df_filter,
                                   col_count = 'spc_common',
                                   option = option,
                                   georef =nta_json,
                                   colref='nta',
                                   name='Num. Árvores')

Scatter = ScatterMapPotly(df=df_filter,
                          option=option,
                          colOption='spc_common',
                          label='tree_dbh')


layout=go.Layout(margin={"r":0,"t":50,"l":0,"b":0},
                 mapbox_style="carto-positron",
                 mapbox_zoom=8.5,mapbox_center = {"lat": 40.7, "lon": -73.86},
                 title=f"Número de árvores de {option} por Bairro de NY com DAP >= que {tree_dbh_slice}",)

figChoroplet_Scatter = go.Figure(data=[Choroplet,Scatter],layout=layout)



#Gráfico Box-Plot de espécie
figBoxEsp = BoxPlotPlotly(df=temp_df.loc[temp_df['spc_common']==option],
                           col = 'spc_common',
                           colValue = 'tree_dbh').update_layout(margin={"r":0,"t":50,"l":0,"b":0},yaxis=dict( title='DAP (cm)'),title=f"Box-plot DAP de {option}")

####Montando as visualizações dos gráficos de  Box-Plot e Choroplet de espécie
col3_1, col3_2 = st.columns([4, 1])

with col3_1:
    st.plotly_chart(figChoroplet_Scatter, use_container_width=True)
with col3_2:
    st.plotly_chart(figBoxEsp, use_container_width=True)



