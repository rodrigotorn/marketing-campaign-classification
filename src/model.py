import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV


def eval_model(x: pd.DataFrame, y: pd.Series, model):
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  cv_results = cross_validate(
    model,
    x,
    y,
    cv=skf,
    scoring=['accuracy', 'balanced_accuracy'],
  )
  return (cv_results['test_accuracy'].mean(),
            cv_results['test_balanced_accuracy'].mean())


def grid_search_model(x: pd.DataFrame, y: pd.Series, model, params: dict):
  grid_search = GridSearchCV(
    model,
    params,
    scoring='balanced_accuracy',
    n_jobs=-1,
    cv=5,
    return_train_score=True,
  )
  grid_search.fit(x, y)
  return pd.DataFrame(grid_search.cv_results_)


if __name__ == '__main__':
  pass
