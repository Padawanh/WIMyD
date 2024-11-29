# Databricks notebook source
# MAGIC %md
# MAGIC # 📃 Escopo
# MAGIC Responsáveis: chsilva\
# MAGIC 28/06/2024
# MAGIC ----
# MAGIC ## 🎯 Alvo
# MAGIC ###**Definir o alvo passa em responder as perguntas abaixo;**
# MAGIC
# MAGIC * **Perguntas de contexto**
# MAGIC   * _Que problema estamos resolvendo?_
# MAGIC   * _Para quem? Quando eles experimentam esse problema?_
# MAGIC   * _Quão prioritário é isso? Por que isso é importante? Por que resolver esse problema é urgente?_
# MAGIC   * Que dados, pesquisas e feedback temos que explicam esse problema?
# MAGIC   * Com quais clientes estamos trabalhando ou ouvindo para entender melhor esse problema?
# MAGIC  
# MAGIC * **Perguntas para iniciar a análise**
# MAGIC   * Qual é o objetivo da análise?
# MAGIC   * Temos os dados? Quais são as fontes de dados?
# MAGIC   * Quais são os dados relevantes para a análise?
# MAGIC   * Existe dependências?
# MAGIC ----
# MAGIC
# MAGIC ## 💭Proposta
# MAGIC > **Exemplos de perguntas para responder:**
# MAGIC * Como estamos resolvendo esse problema? Que alternativas consideramos? Por que aterrissamos com isso?
# MAGIC * Como os dados serão coletados, armazenados, processados e limpos?
# MAGIC * Qual é a forma geral desta solução? Você tem mocks, protótipos, comparações relevantes?
# MAGIC * Quais são as técnicas de análise que serão usadas?
# MAGIC * Como saberemos que resolvemos esse problema? O que vamos medir?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 📚Imports

# COMMAND ----------


import pandas as pd
import numpy as np

#Visualização de dados
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# #Para estatisticas
# import pingouin as pg
import scipy
from scipy.stats import chi2_contingency
from scipy.stats import norm
# import itertools

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# COMMAND ----------


# plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
plt.style.use('seaborn')

#Configurações de visualização do pandas
pd.options.display.precision = 10
pd.options.display.float_format = "{:.3f}".format
pd.options.display.max_info_columns = 200
pd.set_option('display.width', 2500)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.options.display.max_colwidth = 100

# COMMAND ----------

# MAGIC %md #⚙️ Funções

# COMMAND ----------

# MAGIC %md ##Visualização

# COMMAND ----------

from typing import Union, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame

# media_category - Calcula a média das colunas numéricas agrupadas por cluster e retorna um dataframe.
def media_category(df: DataFrame,cat_col: str,count_col: str) -> DataFrame:
  """
    Calcula a média das colunas numéricas agrupadas por cluster e retorna um dataframe.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada.
    cat_col : str
        Nome da coluna que contém a informação do cluster.
    count_col : str
        Nome da coluna que contém a informação das contagens.

    Retornos
    -------
    df_medias : pandas.DataFrame
        DataFrame da média das colunas numéricas agrupadas por cluster.
  """
  df = df.copy()
  select_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()+[cat_col]
  df_medias = df[select_cols].groupby(cat_col,as_index=False).mean()
  df_medias
  KMeansDf_contagem = df[[cat_col,count_col]].groupby(cat_col,as_index=False).count()

  df_medias = KMeansDf_contagem.merge(df_medias, left_on=cat_col, right_on=cat_col)
  sumDF = df_medias.sum()
  sumDF.name = 'sumDF'
  return df_medias

# distribuion_serie - Plota gráficos de barras com as contagens de diferentes colunas, agrupadas por categoria.
def distribuion_serie(df: DataFrame,name_serie: str,hue=None,figsize=(10, 16))-> None:
  """
    Plota gráficos de barras com as contagens de diferentes colunas, agrupadas por categoria.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada.
    listFeatures : list[str]
        Lista com os nomes das colunas que serão utilizadas nos gráficos.
    CategoryCol : str
        Nome da coluna que contém a informação da categoria.
    CountCol : str
        Nome da coluna que contém a informação das contagens.
    n : int, opcional
        Número de categorias a serem consideradas.
    figsize : tuple, opcional
        Tamanho da figura do gráfico.

    Retornos
    -------
    None
  """
  fig, ax = plt.subplots(3,figsize=figsize)
  fig.subplots_adjust(wspace=0.6)
  fig.subplots_adjust(hspace=0.6)
  ax[0] = sns.histplot(df,x=name_serie,ax=ax[0],hue=hue,kde=True)
  ax[0].set(xlabel=f"{name_serie}", ylabel="Frequência")
  ax[0].set_title(f"{name_serie} , DistPlot")

  ax[1] = sns.ecdfplot(df,x=name_serie,
                       hue=hue,
                       ax=ax[1])
  ax[1].set(xlabel=f"{name_serie} comulativo", ylabel="% comulative")
  ax[1].set_title(f"{name_serie}")
  if hue==None:
    ax[2] = sns.boxplot(x=df[name_serie])
  else:
    ax[2] = sns.boxplot(x=df[name_serie],  y=df[hue])
  ax[2].set(xlabel=f"{name_serie}")
  ax[2].set_title(f"Distribuição da {name_serie}")
  if hue==None:
    print(df[name_serie].describe())
  else:
    print(df[[name_serie,hue]].groupby(hue).describe().sort_values((name_serie,'mean')))

# plot_cluster_metrics
def plot_cluster_metrics(df: DataFrame,list_features: list,category_col: str,count_col: str,n: int=5,figsize=(24, 12))-> None:
  top_contends = df[['userId',category_col]].groupby([category_col]).count().sort_values(count_col,ascending=False).head(n)
  df_cluster = df.copy()
  df_cluster_contend = df_cluster[df_cluster[category_col].isin(list(top_contends.index))]

  fig, ax = plt.subplots(2,2,figsize=figsize)
  fig.subplots_adjust(wspace=0.3)
  fig.subplots_adjust(hspace=0.2)
  sns.barplot(data=df_cluster_contend[[category_col,list_features[0],count_col]].groupby([category_col,list_features[0]], as_index=False).count(),
      x=list_features[0], y=count_col, hue=category_col,ax=ax[0,0])
  ax[0,0].legend(title=category_col, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


  sns.barplot(data=df_cluster_contend[[category_col,list_features[1],count_col]].groupby([category_col,list_features[1]], as_index=False).count(),
  x=list_features[1], y=count_col, hue=category_col,ax=ax[0,1])
  ax[0,1].legend(title=category_col, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


  sns.barplot(data=df_cluster_contend[[category_col,list_features[2],count_col]].groupby([category_col,list_features[2]], as_index=False).count(),
  x=list_features[2], y=count_col, hue=category_col,ax=ax[1,0])
  ax[1,0].legend(title=category_col, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# plot_faixas - Plota gráficos de barras com diferentes colunas
def plot_faixas(df: DataFrame,ClusterCol: str,CountCol: str,listCols: list)-> None:
  """
    Plota gráficos de barras com diferentes colunas

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada.
    ClusterCol : str
        Nome da coluna que contém a informação do cluster.
    CountCol : str
        Nome da coluna que contém a informação das contagens.
    listCols : list[str]
        Lista com os nomes das colunas que serão utilizadas nos gráficos.

    Retornos
    -------
    None
  """
  fig, ax = plt.subplots(2,2,figsize=(20, 12))
  fig.subplots_adjust(wspace=0.6)
  fig.subplots_adjust(hspace=0.2)
  sns.barplot(data=df[[ClusterCol,listCols[0],CountCol]].groupby([ClusterCol,listCols[0]], as_index=False).count(),
      x=listCols[0], y=CountCol, hue=ClusterCol,ax=ax[0,0])
  ax[0,0].legend(title=ClusterCol, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

  sns.barplot(data=df[[ClusterCol,listCols[1],CountCol]].groupby([ClusterCol,listCols[1]], as_index=False).count(),
  x=listCols[1], y=CountCol, hue=ClusterCol,ax=ax[1,0])
  ax[1,0].legend(title=ClusterCol, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

  sns.barplot(data=df[[ClusterCol,listCols[2],CountCol]].groupby([ClusterCol,listCols[2]], as_index=False).count(),
  x=listCols[2], y=CountCol, hue=ClusterCol,ax=ax[0,1])
  ax[0,1].legend(title=ClusterCol, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

  sns.barplot(data=df[[ClusterCol,listCols[3],CountCol]].groupby([ClusterCol,listCols[3]], as_index=False).count(),
  x=listCols[3], y=CountCol, hue=ClusterCol,ax=ax[1,1])
  ax[1,1].legend(title=ClusterCol, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

# plot_proportions_cluster - Plota a distribuição de usuários por cluster em um gráfico de barras.
def plot_proportions_cluster(df: DataFrame,ClusterColumn: str,CountColumn: str,axe=None, SelectCluster: int=None ,title: str='Distribuição de usuários por cluster',vert: bool=False):
  """
    Plota a distribuição de usuários por cluster em um gráfico de barras.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada contendo as colunas `ClusterColumn` e `CountColumn`.
    ClusterColumn : str
        Nome da coluna que contém a informação do cluster.
    CountColumn : str
        Nome da coluna que contém a informação das contagens.
    axe : matplotlib.Axis, opcional
        Eixo a ser utilizado em um gráfico já existente. Caso não seja passado, um novo será criado.
    SelectCluster : int ou None, opcional
        Número do cluster a ser realçado na cor vermelha. Caso não seja passado, todas as barras terão a mesma cor.
    title : str, opcional
        Título do gráfico.
    vert : bool, opcional
        Orientação das barras, True para vertical e False para horizontal.

    Retornos
    -------
    bar : seaborn.axisgrid.Barplot
        Objeto do gráfico de barras criado.
  """
  PlotFig = None
  if axe==None:
    fig, ax = plt.subplots(1 ,figsize=(8,5))
    bar = ax
    PlotFig = True
  else:
    bar = axe
  Cluster_contagem = df[[CountColumn,ClusterColumn]].groupby(ClusterColumn,as_index=False).count()
  Cluster_contagem = Cluster_contagem.sort_values(CountColumn,ascending=False).reset_index(drop=True)#.query('userId > 1000')
  Cluster_contagem.columns =[ 'Cluster','Count']
  
  if SelectCluster is None:
    palette=None
  else:
    palette = len(Cluster_contagem)*["blue"]
    palette[Cluster_contagem[Cluster_contagem['Cluster']==SelectCluster].index[0]]="red"
  if vert:
    bar  = sns.barplot(y=Cluster_contagem['Cluster'], x=Cluster_contagem['Count'], palette=palette)
    bar.set(xlabel='Contagem', title=title,ylabel='Clusters')
    bar.bar_label(bar.containers[0])
    return bar
  else:
    bar  = sns.barplot(x=Cluster_contagem['Cluster'], y=Cluster_contagem['Count'], palette=palette)
    bar.set(ylabel='Contagem', title=title,xlabel='Clusters')
    bar.bar_label(bar.containers[0])
    return bar
  # if PlotFig:
  #   return bar.show()

# explorar_categoria - Calcula a representatividade percentual dos valores de uma coluna categórica.
def explorar_categoria(df: DataFrame,count_col: str,category_col: str,explore_range: int) -> None:
  """
    Calcula a representatividade percentual dos valores de uma coluna categórica.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada contendo as colunas `category_col` e `count_col`.
    count_col : str
        Nome da coluna que contém os valores a serem contados.
    category_col : str
        Nome da coluna que contém as categorias.
    explore_range : int
        Número de categorias a serem exploradas.

    Retornos
    -------
    None
  """
  if explore_range>len(df[category_col].unique()):
    explore_range=len(df[category_col].unique())
    print('explore_range ajustado para quantidade de categorias')
  percente = list()
  for i in range(1,explore_range):
    count = df[[category_col,count_col]].groupby(category_col).count().sort_values(count_col,ascending=False).head(i)
    percente.append(count[count_col].sum()/df.shape[0])
  percente = percente*100

  plt.plot(percente[:explore_range-1])
  plt.xlabel(f'Número de categorias em  {category_col}')
  plt.ylabel('Representatividade %')
  plt.title(f'Representatividade de categorias em  {category_col}')
  plt.show()
  print(f'A quantidade de valores vazios é de {df[category_col].isnull().sum()/len(df):.2%}')
  print(f'O número toral de categorias é de {len(df[category_col].unique())}')

# COMMAND ----------

# MAGIC %md ##ETL e Stats

# COMMAND ----------



def count_zeros(df: DataFrame):
    return df.apply(lambda x: x.value_counts(normalize=True).get(0, 0))

def chi_square_test(df_original: DataFrame, cat_col: str, groupby_col: str, print_along: bool=True, show_figs:bool=True):
  print(f'chi square com {cat_col}')
  
  df = df_original.copy()

  agg_dict = {}
  
  df[cat_col] = df[cat_col].apply(lambda x: str(x) if not pd.isna(x) else x)

  for f in df[cat_col].unique():
    df[f] = df[cat_col].apply(lambda x: 1 if x == f else 0)
    agg_dict[f] = 'sum'

  df_count_formats = (
    df
    .groupby([groupby_col])
    .agg(agg_dict)
    .reset_index()
  )

  for col in df_count_formats.columns:
    if col == groupby_col:
      continue
    if df_count_formats[col].min() == 0:
      df_count_formats.drop(columns=[col], inplace=True)

  if show_figs:
    print('tabela de contingência')
    display(df_count_formats)
    
  real_values = df_count_formats.drop(columns=[groupby_col]).values

  chi2, p, dof, expected = chi2_contingency(df_count_formats.drop(columns=[groupby_col]).values)

  if print_along:
    print(f"chi2 statistic:     {chi2:.5g}")
    print(f"p-value:            {p:.5g}")
    print(f"degrees of freedom: {dof}")
    
  if show_figs:
    print('valores esperados')
    display(pd.DataFrame(data=expected, columns=df_count_formats.drop(columns=[groupby_col]).columns))
    
    chi_test_values = np.zeros(shape=expected.shape)
    for i in range(0, len(expected)):
      for j in range(0, len(expected[i])):
        # Mantém os valores negativos para identificar abaixo do esperado
        chi_test_values[i][j] = (abs(expected[i][j] - real_values[i][j])*(expected[i][j] - real_values[i][j]))/expected[i][j]
  
    print('dados do teste por célula da tabela de contingência')
    fig = px.imshow(pd.DataFrame(data=chi_test_values, columns=df_count_formats.drop(columns=[groupby_col]).columns))
    fig.show()
    
  return p

#Compara array de valores não parametricos de dataframes
def TestMannWhitneyScipyTable (df1: DataFrame,df2: DataFrame,VarList: list,alternative: str='two-sided',alfa: float=0.05,label_x: str="X",label_y: str="Y"):
  pvalue = list()
  MeanX = list()
  MedianX = list()
  MedianY = list()
  MeanY  = list()
  MadX = list()
  MadY = list()
  stdX =list()
  stdY = list()
  UX = list()
  UY = list()
  f = list()
  for i in VarList:
    U1, p = scipy.stats.mannwhitneyu(df1[i],df2[i],alternative=alternative)
    pvalue.append(p)
    UX.append(U1)
    UY.append(len(df1[i])*len(df2[i]) - U1)
    f.append(U1/(len(df1[i])*len(df2[i])))
    MeanX.append(df1[i].mean())
    MedianX.append(df1[i].median())
    MeanY.append(df2[i].mean())
    MedianY.append(df2[i].median())
    MadX.append(df1[i].mad())
    MadY.append(df2[i].mad())
    stdX.append(df1[i].std())
    stdY.append(df2[i].std())

  result = pd.DataFrame({"Feature":VarList,
                         "pvalue":pvalue,
                         'Aceitar H0, são iguais':np.array(pvalue)>alfa,
                         'Mean '+label_x:MeanX,
                         'Mean '+label_y:MeanY,
                         'Median '+label_x:MedianX,
                         'Median '+label_y:MedianY,
                         'U1': UX,
                         'U2': UY,
                         'f': f,
                         'r': np.array(f) - (1-np.array(f)),
                         'n_values_'+label_x: len(df1),
                         'n_values_'+label_y: len(df2),
                         'Mad'+label_x: MadX,
                         'Mad'+label_y: MadY,
                         'std'+label_x: stdX,
                         'std'+label_y: stdY
                         })

  result['U1'] = result['U1'].apply(lambda x: '{:.2e}'.format(x))
  result['U2'] = result['U2'].apply(lambda x: '{:.2e}'.format(x))

  return result



def z_test_two_prop(
    p1: float,
    p2: float,
    n1: int,
    n2: int,
    alfa: float = 0.05,
    alternative: str = "two-sided",
    show_prints: bool = True,
) -> Tuple[float, float, bool, float, float]:
    """
    Conducts a two-proportion z-test to compare the population proportions of two independent groups.
    Proporção de duas amostras, Exemplo em video [aqui](https://www.youtube.com/watch?v=g7hLCFMkEKs) e também [esse.](https://statplace.com.br/blog/testes-para-comparacao-de-proporcoes/)

    Args:
        p1 (float): Proportion of successes in the first group.
        p2 (float): Proportion of successes in the second group.
        n1 (int): Sample size of the first group.
        n2 (int): Sample size of the second group.
        alfa (float, optional): Significance level (default: 0.05).
        alternative (str, optional): Alternative hypothesis (default: 'two-sided').
            - 'two-sided': Tests if p1 is different from p2 (default).
            - 'less': Tests if p1 is less than p2.
            - 'greater': Tests if p1 is greater than p2.
        show_prints (bool, optional): Whether to print results (default: True).

    Returns:
        Tuple[float, float, bool, float, float]:
            - z_score (float): The calculated z-score.
            - p_value (float): The p-value associated with the test.
            - reject_null (bool): True if the null hypothesis is rejected, False otherwise.
            - ci_lower (float): Lower bound of the confidence interval.
            - ci_upper (float): Upper bound of the confidence interval.

    Raises:
        ValueError: If `alternative` is not one of 'two-sided', 'less', or 'greater'.
    """

    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se

    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(
            "Invalid alternative hypothesis. Must be 'two-sided', 'less', or 'greater'."
        )

    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z)))
    elif alternative == "less":
        p_value = norm.cdf(z)
    elif alternative == "greater":
        p_value = 1 - norm.cdf(z)

    z_alpha = norm.ppf(1 - alfa / 2)
    ci_lower = (p1 - p2) - z_alpha * se
    ci_upper = (p1 - p2) + z_alpha * se

    # Effect size calculation can be added here (e.g., Cohen's d)

    if show_prints:
        if p_value < alfa:
            print(f"Z-score: {z:.4f}, P-value: {p_value:.4f}")
            print("Reject the null hypothesis.")
        else:
            print(f"Z-score: {z:.4f}, P-value: {p_value:.4f}")
            print("Fail to reject the null hypothesis.")

    return z, p_value, p_value < alfa, ci_lower, ci_upper


# #Exemplo de uso:
# n2 = 400  # tamanho da primeira amostra
# p2 = 132/n2  # proporção na primeira amostra
# n1 = 600   # tamanho da segunda amostra
# p1 = 228/n1  # proporção na segunda amostra
# alfa = 0.05  # nível de significância
# alternative = 'two-sided'  # 'two-sided', 'less', 'greater'

# z_test_two_prop(p1, p2, n1, n2, alfa, alternative)


def teste_z_table(
    df: DataFrame,
    p1: str = "p1",
    p2: str = "p2",
    n1: str = "n1",
    n2: str = "n2",
    alfa: float = 0.05,
    alternative: str = "two-sided",
) -> DataFrame:
    """
    Conducts Z-tests for two proportions across all rows of a DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing columns for proportions (p1, p2) and sample sizes (n1, n2).
        p1 (str, optional): Column name for proportion in the first group (default: "p1").
        p2 (str, optional): Column name for proportion in the second group (default: "p2").
        n1 (str, optional): Column name for sample size in the first group (default: "n1").
        n2 (str, optional): Column name for sample size in the second group (default: "n2").
        alfa (float, optional): Significance level (default: 0.05).
        alternative (str, optional): Alternative hypothesis (default: "two-sided").
            - "two-sided": Tests if p1 is different from p2 (default).
            - "less": Tests if p1 is less than p2.
            - "greater": Tests if p1 is greater than p2.

    Returns:
        DataFrame: A new DataFrame with added columns:
            - "p_value": P-value for each Z-test.
            - f"p_value < alfa {alfa}": Boolean indicating if p-value is less than alpha.
            - "z": Z-score for each Z-test.
            - "ci_lower": Lower bound of the confidence interval for each test.
            - "ci_upper": Upper bound of the confidence interval for each test.

    Raises:
        ValueError: If any column names (p1, p2, n1, n2) are not found in the DataFrame.
    """

    if not all(col in df.columns for col in [p1, p2, n1, n2]):
        raise ValueError(
            "Invalid column names. Ensure all columns (p1, p2, n1, n2) exist in the DataFrame."
        )

    z = []
    p_value = []
    p_alfa = []
    ci_lower = []
    ci_upper = []

    for index, row in df.iterrows():
        z_add, p_value_add, p_alfa_add, ci_lower_add, ci_upper_add = z_test_two_prop(
            row[p1],
            row[p2],
            row[n1],
            row[n2],
            alfa=alfa,
            alternative=alternative,
            show_prints=False,
        )
        z.append(z_add)
        p_value.append(p_value_add)
        p_alfa.append(p_alfa_add)
        ci_lower.append(ci_lower_add)
        ci_upper.append(ci_upper_add)

    df_return = df.copy()
    df_return["p_value"] = p_value
    df_return[f"p_value < alfa {alfa}"] = p_alfa  # Consistent column name formatting
    df_return["z"] = z
    df_return["ci_lower"] = ci_lower
    df_return["ci_upper"] = ci_upper

    return df_return


# #Exemplo de uso:
# tabela = pd.DataFrame(
#     {
#         "p1": [0.5, 0.704, 0.3, 0.5, 0.704],
#         "p2": [0.51, 0.55, 0.5, 0.704, 0.3],
#         "n1": [1000000, 100000, 1000, 1000, 1000],
#         "n2": [1000, 10000, 100, 100, 100],
#     }
# )
# teste_z_table(df=tabela, p1="p1", p2="p2", n1="n1", n2="n2")