import os
import numpy as np
import joblib
import xgboost as xgb
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, classification_report, confusion_matrix

def train_model(df, season, week, features, numeric_features):
    X = df[["player_id"] + features]
    y = df['touchdown']
    player_ids = X['player_id'].values
    X = X.drop('player_id', axis=1)
    ordinal_features = ["report_status"]
    report_status_order = ["Healthy", "Minor", 'Questionable', 'Doubtful', 'Out']

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X[numeric_features] = imputer.fit_transform(X[numeric_features])
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    encoder = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=[report_status_order]), ordinal_features)
        ],
        remainder='passthrough'
    )
    X_encoded = encoder.fit_transform(X)
    X_train, X_test, y_train, y_test, player_ids_train, player_ids_test = train_test_split(
        X_encoded, y, player_ids, test_size=0.2, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
    dtest = xgb.DMatrix(X_test, label=y_test)
    threshold = 0.5

    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('eta', 0.01, 0.9),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'scale_pos_weight': len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1]),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        model = xgb.train(param, dtrain, num_boost_round=100)
        y_pred_prob = model.predict(dtest)
        loss = log_loss(y_test, y_pred_prob)
        return loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, timeout=600)
    best_params = study.best_trial.params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1]),
    })
    best_model = xgb.train(best_params, dtrain, num_boost_round=100)
    best_model.save_model(f"Model/xgboost_model-{season}-{week}.json")
    joblib.dump(imputer, f"Model/imputer-{season}-{week}.pkl")
    joblib.dump(scaler, f"Model/scaler-{season}-{week}.pkl")
    joblib.dump(encoder, f"Model/encoder-{season}-{week}.pkl")

    y_pred_prob = best_model.predict(dtest)
    y_pred_binary = [1 if prob > threshold else 0 for prob in y_pred_prob]
    print(classification_report(y_test, y_pred_binary))
    print(confusion_matrix(y_test, y_pred_binary))
    return classification_report(y_test, y_pred_binary)

def predict_week(season, week, current_week, features, numeric_features):
    new_data = current_week
    X = new_data[["player_id"] + features]
    player_ids = X['player_id'].values
    X = X.drop('player_id', axis=1)
    imputer = joblib.load(f"Model/imputer-{season}-{week}.pkl")
    scaler = joblib.load(f"Model/scaler-{season}-{week}.pkl")
    encoder = joblib.load(f"Model/encoder-{season}-{week}.pkl")
    X[numeric_features] = imputer.transform(X[numeric_features])
    X[numeric_features] = scaler.transform(X[numeric_features])
    X_encoded = encoder.transform(X)
    dnew = xgb.DMatrix(X_encoded)
    best_model = xgb.Booster()
    best_model.load_model(f"Model/xgboost_model-{season}-{week}.json")
    y_pred_new = best_model.predict(dnew)
    import pandas as pd
    predictions_df = pd.DataFrame({
        'player_id': player_ids,
        'probability': y_pred_new
    })
    predictions_df_sorted = predictions_df.sort_values(by='probability', ascending=False)
    output = pd.merge(predictions_df_sorted, new_data[["season", "week", "player_id", "player_name", "team", "against", "report_status", "played"]], how='inner', on='player_id')
    output = pd.merge(output, current_week[["player_id", "touchdown"]], how='left', on='player_id')
    output["touchdown"] = np.where(output["played"] == 1, output["touchdown"], np.nan)
    output["played"] = np.where(output["played"] == 1, "Y", "N")
    filtered_list_with_na = [x for x in output["touchdown"].values if pd.notna(x)]
    filtered_list_all_values = [output["probability"].values[i] for i in range(len(output["touchdown"].values)) if pd.notna(output["touchdown"].values[i])]
    final_df = pd.DataFrame({
        'touchdown': filtered_list_with_na,
        'probability': filtered_list_all_values
    })
    y_pred_binary = [1 if prob > 0.5 else 0 for prob in final_df["probability"]]
    if len(final_df["touchdown"].value_counts()) > 0:
        print(classification_report(final_df["touchdown"], y_pred_binary))
        print(confusion_matrix(final_df["touchdown"], y_pred_binary))
    return output

def retrain_model(season, week, df, features, numeric_features):
    model_filename = f"Model/xgboost_model-{season}-{week}.json"
    if os.path.exists(model_filename):
        return False, f"Model for season {season}, week {week} has already been trained."
    else:
        train_model(df, season, week, features, numeric_features)
        return True, f"Model (using data prior to season {season}, week {week}) has been retrained and saved."