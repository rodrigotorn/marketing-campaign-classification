import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


def create_feature_accepted_any_cmp(df: pd.DataFrame):
  campaigns = df.loc[:,'AcceptedCmp3':'AcceptedCmp2'].astype(int)
  df['NumAcceptedCmp'] = campaigns.apply(sum, axis=1)
  return df


def calc_pca(df: pd.DataFrame):
  numerical_features = df.select_dtypes(include=['int', 'float'])
  standarzed_features = StandardScaler().fit_transform(numerical_features)

  pca = PCA(n_components=2).fit_transform(standarzed_features)
  pca_df = pd.DataFrame(
    data=pca,
    columns=['pca1', 'pca2'],
    index=df.index,
  )
  return pd.concat([pca_df, df[['Response']]], axis = 1)
