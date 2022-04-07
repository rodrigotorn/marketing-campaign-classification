import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_corr_matrix(df: pd.DataFrame):
  corr = df.corr()
  mask = np.triu(np.ones_like(corr, dtype=bool))
  f, ax = plt.subplots(figsize=(11, 9))
  cmap = sns.diverging_palette(20, 230, as_cmap=True)
  return sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    square=True, linewidths=.5, cbar_kws={"shrink": .5})


def plot_boxplot(df: pd.DataFrame, feature: str):
  f, ax = plt.subplots(figsize=(7, 5))
  return sns.boxplot(x='Response', y=feature, data=df)


def plot_pca(df: pd.DataFrame):
  fig = plt.figure(figsize = (7,5))
  ax = fig.add_subplot(1,1,1)

  ax.set_xlabel('Principal Component 1', fontsize = 15)
  ax.set_ylabel('Principal Component 2', fontsize = 15)
  ax.set_title('2 component PCA', fontsize = 20)
  targets = ['0', '1']
  colors = ['r', 'g']

  for target, color in zip(targets, colors):
    indicesToKeep = df['Response'].astype(str) == target
    ax.scatter(
      df.loc[indicesToKeep, 'pca1'],
      df.loc[indicesToKeep, 'pca2'],
      c = color,
      s = 50,
    )
  ax.legend(targets)
  ax.grid()
  return fig
