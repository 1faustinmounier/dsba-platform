import pandas as pd
from typing import List
from .model_registry import load_model
from sklearn.metrics import accuracy_score, f1_score

def compare_models_simple(model_ids: List[str],
                          X_eval: pd.DataFrame,
                          y_eval: pd.Series,
                          metrics: List[str]) -> pd.DataFrame:
    
    results = []
    
    supported_metrics = ['accuracy', 'f1']
    valid_metrics = [m.lower() for m in metrics if m.lower() in supported_metrics]

    for model_id in model_ids:
        model = load_model(model_id)
        y_pred = model.predict(X_eval)

        scores = {'model_id': model_id}
        for metric_name in valid_metrics:
            if metric_name == 'accuracy':
                scores[metric_name] = accuracy_score(y_eval, y_pred)
            elif metric_name == 'f1':
                scores[metric_name] = f1_score(y_eval, y_pred, average='binary', zero_division=0)

        results.append(scores)

    comparison_df = pd.DataFrame(results)

    final_cols = ['model_id'] + valid_metrics
    for metric in valid_metrics:
        if metric not in comparison_df.columns:
            comparison_df[metric] = float('nan')

    return comparison_df[final_cols]
