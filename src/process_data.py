import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


def find_outliers(df: pd.DataFrame):
  q1 = df.quantile(0.25)
  q3 = df.quantile(0.75)
  iqr = q3 - q1

  numerical_features = df.select_dtypes(include=['int', 'float'])
  outliers = (numerical_features < (q1 - 1.5 * iqr)) \
    | (numerical_features > (q3 + 1.5 * iqr))

  return outliers


def knn_imput(df: pd.DataFrame, n_neighbors:int=5):
  outliers = find_outliers(df)
  df[outliers] = np.nan

  numerical_features = df.select_dtypes(include=['int', 'float'])
  non_numerical_features = df.select_dtypes(exclude=['int', 'float'])

  imputer = KNNImputer(n_neighbors=n_neighbors)
  imputed_df = pd.DataFrame(
    imputer.fit_transform(numerical_features),
    index = numerical_features.index,
    columns = numerical_features.columns,
  )
  return imputed_df.join(non_numerical_features)


def calc_bigger_rel_diff_by_response(df: pd.DataFrame):
  grouped_df = df.groupby('Response').median()
  return (grouped_df.loc[1]/grouped_df.loc[0]) \
    .sort_values(ascending=False).index[0]


def create_feature_num_accepted_cmp(df: pd.DataFrame):
  campaigns = df.loc[:,'AcceptedCmp3':'AcceptedCmp2'].astype(int)
  df['NumAcceptedCmp'] = campaigns.apply(sum, axis=1)
  return df


def calc_pca(df: pd.DataFrame):
  pca = PCA(n_components=2).fit_transform(df)
  return pd.DataFrame(
    data=pca,
    columns=['pca1', 'pca2'],
    index=df.index,
  )


def kmeans_clustering(df: pd.DataFrame):
  kmeans = KMeans(n_clusters=3, random_state=0)
  clusters = kmeans.fit(df)
  labels = pd.Series(clusters.labels_, index=df.index, name='Cluster')

  return pd.concat([
      df[['Income', 'NumAcceptedCmp', 'NumCatalogPurchases']],
      labels,
  ], axis=1)


def preprocess(df: pd.DataFrame):
  x = pd.get_dummies(df,
    columns=df.select_dtypes(include='category').columns.to_list()
  )
  kept_columns = \
    VarianceThreshold(0.01).fit(x).get_feature_names_out(x.columns)
  x = x[kept_columns]
  scaler = MinMaxScaler()
  return pd.DataFrame(
    scaler.fit_transform(x),
    index=x.index,
    columns=x.columns,
  )


if __name__ == '__main__':
  pass
