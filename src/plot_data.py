import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style='whitegrid')


def plot_corr_matrix(df: pd.DataFrame, title: str):
  f, ax = plt.subplots(figsize=(11, 9))
  ax.set_title(title)

  corr = df.corr()
  mask = np.triu(np.ones_like(corr, dtype=bool))
  cmap = sns.diverging_palette(20, 230, as_cmap=True)

  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    square=True, linewidths=.5, cbar_kws={'shrink': .5})
  return f


def plot_boxplot(df: pd.DataFrame, feature: str, title: str):
  f, ax = plt.subplots(figsize=(7, 5))
  ax.set_title(title)
  sns.boxplot(x='Response', y=feature, data=df)
  return f


def plot_barplot_clusters(df: pd.DataFrame, features: list, title: str):
  f, ax = plt.subplots(1, 3, figsize=(18, 5))
  f.suptitle(title)

  sns.barplot(ax=ax[0], x='Cluster', y=features[0], data=df)
  sns.barplot(ax=ax[1], x='Cluster', y=features[1], data=df)
  sns.barplot(ax=ax[2], x='Cluster', y=features[2], data=df)
  return f


def plot_pca(df: pd.DataFrame, title: str):
  f, ax = plt.subplots(figsize=(7, 5))
  ax.set_title(title)
  ax.set_xlabel('PCA1')
  ax.set_ylabel('PCA2')

  sns.scatterplot(
      x='pca1',
      y='pca2',
      hue='Cluster',
      palette='deep',
      data=df,
  )
  return f


def plot_overfit_analysis(
  df: pd.DataFrame,
  x_label: str,
  y_label: str,
  title: str,
):
  f, ax = plt.subplots(figsize=(7, 5))
  ax.set_title(title)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)

  sns.lineplot(
    x='param_max_depth',
    y='mean_test_score',
    data=df,
    label='Test'
  )
  sns.lineplot(
    x='param_max_depth',
    y='mean_train_score',
    data=df,
    label='Train'
  )
  plt.legend()
  return f


if __name__ == '__main__':
  pass
