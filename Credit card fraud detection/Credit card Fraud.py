import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, callbacks

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify

DATA_PATH = 'creditcard.csv'  
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_SAMPLING_STRATEGY = 0.1 
UNDERSAMPLE_STRATEGY = 0.5  
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

ALERT_PROB_THRESHOLD = 0.5 
AUTOENCODER_THRESHOLD = None  

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def eda(df, show_plots=True):
    print('Dataset shape:', df.shape)
    print(df['Class'].value_counts())
    if show_plots:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        sns.histplot(df['Amount'], bins=50, log_scale=True)
        plt.title('Transaction Amount (log scale)')
        plt.subplot(1,3,2)
        sns.histplot(df['Time'], bins=50)
        plt.title('Transaction Time')
        plt.subplot(1,3,3)
        sns.countplot(x='Class', data=df)
        plt.title('Class distribution (0=Normal,1=Fraud)')
        plt.tight_layout()
        plt.show()
    if show_plots:
        plt.figure(figsize=(8,4))
        sns.boxplot(x='Class', y='Amount', data=df[df['Amount']<500])
        plt.yscale('log')
        plt.title('Amount by Class (zoomed)')
        plt.show()

def preprocess(df, use_smote=True, undersample=False, scale=True):
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = None
    if scale:
        scaler = StandardScaler()
        X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    if use_smote:
        sm = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=RANDOM_STATE)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print('After SMOTE, counts:', np.bincount(y_train_res))
    else:
        X_train_res, y_train_res = X_train, y_train
    if undersample:
        rus = RandomUnderSampler(sampling_strategy=UNDERSAMPLE_STRATEGY, random_state=RANDOM_STATE)
        X_train_res, y_train_res = rus.fit_resample(X_train_res, y_train_res)
        print('After undersampling, counts:', np.bincount(y_train_res))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    return X_train_res, X_test, y_train_res, y_test

def evaluate_model(model, X_test, y_test, thresh=ALERT_PROB_THRESHOLD, show_cm=True):
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:,1]
        preds = (probs >= thresh).astype(int)
    else:
        try:
            preds = model.predict(X_test)
            preds = np.where(preds==-1, 1, 0)
            probs = None
        except Exception:
            preds = model.predict(X_test)
            probs = None
    print(classification_report(y_test, preds, digits=4))
    if show_cm:
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    if probs is not None:
        try:
            auc = roc_auc_score(y_test, probs)
            print('ROC AUC:', auc)
        except Exception:
            pass
    return preds

def train_logistic(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, 'logistic.joblib'))
    return lr

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest.joblib'))
    return rf

def train_isolation_forest(X_train):
    iso = IsolationForest(n_estimators=100, contamination=0.001, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train)
    joblib.dump(iso, os.path.join(MODEL_DIR, 'isolation_forest.joblib'))
    return iso

def build_autoencoder(input_dim, encoding_dim=14):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    encoded = layers.Dense(int(encoding_dim/2), activation='relu')(encoded)
    decoded = layers.Dense(int(encoding_dim/2), activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(X_train, X_val, epochs=50, batch_size=128):
    input_dim = X_train.shape[1]
    ae = build_autoencoder(input_dim)
    early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = ae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, X_val), callbacks=[early], verbose=2)
    preds_val = ae.predict(X_val)
    mse = np.mean(np.power(X_val - preds_val, 2), axis=1)
    threshold = np.percentile(mse, 99.5)
    print('Autoencoder validation MSE threshold:', threshold)
    ae.save(os.path.join(MODEL_DIR, 'autoencoder.h5'))
    return ae, threshold, history

def create_flask_app(model_path=None, model_type='rf'):
    app = Flask(__name__)
    scaler = None
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    model = None
    if model_type == 'rf':
        model = joblib.load(os.path.join(MODEL_DIR, 'random_forest.joblib'))
    elif model_type == 'lr':
        model = joblib.load(os.path.join(MODEL_DIR, 'logistic.joblib'))
    elif model_type == 'iso':
        model = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.joblib'))
    elif model_type == 'ae':
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'autoencoder.h5'))
    else:
        raise ValueError('Unknown model_type')
    ae_threshold = None
    if model_type == 'ae':
        tpath = os.path.join(MODEL_DIR, 'ae_threshold.joblib')
        if os.path.exists(tpath):
            ae_threshold = joblib.load(tpath)
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        if scaler is not None and {'Time','Amount'}.issubset(df.columns):
            df[['Time','Amount']] = scaler.transform(df[['Time','Amount']])
        if model_type in ['rf','lr']:
            probs = model.predict_proba(df)[:,1]
            is_fraud = (probs >= ALERT_PROB_THRESHOLD).astype(int)
            return jsonify({'fraud_prob': float(probs[0]), 'flagged': int(is_fraud[0])})
        elif model_type == 'iso':
            pred = model.predict(df)
            flagged = int(pred[0] == -1)
            return jsonify({'flagged': flagged})
        elif model_type == 'ae':
            recon = model.predict(df.values)
            mse = np.mean(np.power(df.values - recon, 2), axis=1)
            flagged = int(mse[0] > ae_threshold)
            return jsonify({'reconstruction_error': float(mse[0]), 'flagged': flagged})
    return app

def run_pipeline(do_eda=True, use_smote=True, undersample=False):
    df = load_data()
    if do_eda:
        eda(df)
    X_train_res, X_test, y_train_res, y_test = preprocess(df, use_smote=use_smote, undersample=undersample, scale=True)
    print('\nTraining Logistic Regression...')
    lr = train_logistic(X_train_res, y_train_res)
    print('Evaluating Logistic Regression:')
    evaluate_model(lr, X_test, y_test)
    print('\nTraining Random Forest...')
    rf = train_random_forest(X_train_res, y_train_res)
    print('Evaluating Random Forest:')
    evaluate_model(rf, X_test, y_test)
    print('\nTraining Isolation Forest on non-fraud transactions...')
    normal_X = df[df['Class']==0].drop('Class', axis=1)
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib')) if os.path.exists(os.path.join(MODEL_DIR, 'scaler.joblib')) else None
    if scaler is not None:
        normal_X[['Time','Amount']] = scaler.transform(normal_X[['Time','Amount']])
    iso = train_isolation_forest(normal_X)
    print('Evaluating Isolation Forest on test set:')
    evaluate_model(iso, X_test, y_test)
    print('\nTraining Autoencoder on normal transactions...')
    normal = df[df['Class']==0].drop('Class', axis=1)
    if scaler is not None:
        normal[['Time','Amount']] = scaler.transform(normal[['Time','Amount']])
    Xn_train, Xn_val = train_test_split(normal, test_size=0.2, random_state=RANDOM_STATE)
    ae, threshold, history = train_autoencoder(Xn_train.values, Xn_val.values, epochs=50)
    joblib.dump(threshold, os.path.join(MODEL_DIR, 'ae_threshold.joblib'))
    print('Evaluating Autoencoder on X_test...')
    recon = ae.predict(X_test.values)
    mse = np.mean(np.power(X_test.values - recon, 2), axis=1)
    preds = (mse > threshold).astype(int)
    print(classification_report(y_test, preds, digits=4))
    print('\nPipeline completed. Models saved in', MODEL_DIR)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print(f'Please place the Kaggle creditcard.csv at: {DATA_PATH} and re-run.')
    else:
        run_pipeline(do_eda=True, use_smote=True, undersample=False)
