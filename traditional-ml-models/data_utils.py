# data_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, fbeta_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import os

def load_and_preprocess(csv_path, balance_method="normal"):

    df = pd.read_csv(csv_path)
    
    # Fill missing age with median
    if df['age'].isnull().sum() > 0:
        df['age'] = df['age'].fillna(df['age'].median())
    
    # Separate features and labels
    X = df.drop(columns=['label'])
    y = df['label']

     # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    if balance_method == "over":
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    elif balance_method == "under":
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    
   

    # Train-test split
    
    
    class_weights_dict = None
    if balance_method == "weighted":
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    
    return X_train, X_test, y_train, y_test, scaler, class_weights_dict

def find_best_threshold_for_f2(y_true, y_probs, beta=np.sqrt(3)):
    best_threshold = 0.5
    best_f2 = 0

    thresholds = np.arange(0.1, 1.01, 0.01)
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=beta)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t

    return best_threshold, best_f2

def evaluate_model(model, X_test, y_test, threshold=0.3, beta=2):
    y_probs = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f2 = find_best_threshold_for_f2(y_test, y_probs, beta=beta)
    y_pred = (y_probs >= best_thresh).astype(int)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Print metrics
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"F2 Score  : {best_f2:.4f}")
    print(f"Best Threshold  : {best_thresh}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "f2_score" : best_f2,
        "best_threshold": best_thresh,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
        
    }

def train_and_save_model(model, X_train, X_test, y_train, y_test, save_path):
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
    return metrics

def initialise_model(ml_type= "lr", class_weights = None):
    model = None
    if ml_type == "lr":
        model = LogisticRegression(max_iter=1000, C=1.0, penalty='l2', class_weight=class_weights)
    if ml_type == "svm":
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight=class_weights)
    if ml_type == "xgboost":
        model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric='logloss', use_label_encoder=False,
        class_weight=class_weights
    )
    if ml_type == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weights)
    cal_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5)
    return cal_model

def save_results_to_csv(csv_path, results):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
        else:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame([results])
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

def train(dataset_paths, ml_type, balance_method):
    for dataset_path in dataset_paths:
        print(f"\nProcessing {dataset_path}...")
        
        X_train, X_test, y_train, y_test, scaler, class_weights = load_and_preprocess(dataset_path, balance_method)
        
        base_dataset_name = os.path.basename(dataset_path)
        model_name = f"{ml_type}_{balance_method}_{base_dataset_name}"
        save_dir = f"models/{ml_type}"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, model_name.replace(".csv", ".joblib"))
        
        model = initialise_model(ml_type, class_weights)
        
        metrics = train_and_save_model(model, X_train, X_test, y_train, y_test,save_path)

        metrics["ml_type"] = ml_type
        metrics["balance_method"] = balance_method
        metrics["dataset"] = os.path.basename(dataset_path)
    
        save_results_to_csv("results_summary.csv", metrics)

