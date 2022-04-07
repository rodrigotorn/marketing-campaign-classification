import pandas as pd

CATEGORY_COLUMNS = [
  'AcceptedCmp1',
  'AcceptedCmp2',
  'AcceptedCmp3',
  'AcceptedCmp4',
  'AcceptedCmp5',
  'Response',
  'Complain',
  'Education',
  'Marital_Status',
]

def convert_types(df: pd.DataFrame, columns: list, desired_type: str):
  for column in columns:
    df[column] = df[column].astype(desired_type)
  return df

def read_data(**kargs):
  raw_df = pd.read_csv(**kargs)
  df = convert_types(raw_df, CATEGORY_COLUMNS, 'category')
  df['Dt_Customer'] = pd.to_datetime(
    df['Dt_Customer'],
    format='%Y-%m-%d',
  )
  df.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True)
  return df

if __name__ == '__main__':
  pass
