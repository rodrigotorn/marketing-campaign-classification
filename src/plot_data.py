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


def plot_barplot_clusters(df: pd.DataFrame, features: list):
  f, ax = plt.subplots(1, 3, figsize=(15, 5))
  sns.barplot(ax=ax[0], x='Cluster', y=features[0], data=df)
  sns.barplot(ax=ax[1], x='Cluster', y=features[1], data=df)
  sns.barplot(ax=ax[2], x='Cluster', y=features[2], data=df)
  return f


def plot_pca(df: pd.DataFrame):
  fig = plt.figure(figsize = (7,5))
  ax = fig.add_subplot(1,1,1)

  ax.set_xlabel('Principal Component 1', fontsize = 15)
  ax.set_ylabel('Principal Component 2', fontsize = 15)
  ax.set_title('2 component PCA', fontsize = 20)
  targets = ['0', '1', '2']
  colors = ['b', 'y', 'g']

  for target, color in zip(targets, colors):
    indicesToKeep = df['Cluster'].astype(str) == target
    ax.scatter(
      df.loc[indicesToKeep, 'pca1'],
      df.loc[indicesToKeep, 'pca2'],
      c = color,
      s = 50,
    )
  ax.legend(targets)
  ax.grid()
  return fig


def plot_overfit_analysis(
  x: pd.Series,
  train_scores: pd.Series,
  test_scores: pd.Series,
  x_label: str,
  y_label: str,
  title: str,
):
  f, ax = plt.subplots(figsize=(7, 5))
  plt.plot(x, train_scores, '-o', label='Train')
  plt.plot(x, test_scores, '-o', label='Test')
  plt.legend()
  plt.grid()
  ax.set_xlabel(x_label, fontsize = 15)
  ax.set_ylabel(y_label, fontsize = 15)
  ax.set_title(title, fontsize = 20)
  return f.show()


if __name__ == '__main__':
  pass
