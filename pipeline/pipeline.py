from scipy.signal import butter, filtfilt
import numpy as np
import wfdb
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import load_model
import joblib

class Pipeline():
    def __init__(self, cnn_model_filepath, traditional_ml_model_filepath, scalar):
        self.cnn_model_filepath =cnn_model_filepath
        self.traditional_ml_model_filepath = traditional_ml_model_filepath
        self.scalar = scalar

    def __load_ptbxl_signal(self, filepath):
        record = wfdb.rdrecord(filepath)
        return record.p_signal, record.sig_name

    def __highpass_filter(self, signal, fs=500, cutoff=0.5, order=3):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
        return filtfilt(b, a, signal, axis=0)

    def __normalize_signal(self, signal):
        return (signal - np.mean(signal, axis=0)) / np.std(signal, axis=0)

    def __preprocess_signal(self, signal):
        signal = self.__highpass_filter(signal)
        signal = self.__normalize_signal(signal)
        return signal


    def __load_feature_extractor_model(self, full_model_filepath):
        f1 = F1Score(num_classes=1, average='micro', threshold=0.5, name='f1')
        full_model = load_model(full_model_filepath, custom_objects={'f1': f1})
        if len(full_model.layers) == 0:
            raise ValueError(f"No layers found in model loaded from: {full_model_filepath}")
        feature_model = full_model.layers[0]
        
        return feature_model

    def __extract_features(self, feature_extractor_model, ecg_signal):
        ecg_signal = np.expand_dims(ecg_signal, axis=0)
        features = feature_extractor_model.predict(ecg_signal, verbose=0)
        # if features.ndim == 1:
        #     features = features.reshape(-1, 1)
        
        return features

    def __combine_metadata_with_ecg_features(self, features, age, sex):
        metadata = np.array([[age, sex]])
        final_features = np.concatenate([features, metadata], axis=1)
        return final_features

    def __load_traditional_ml_model(self, model_filepath):
        model = joblib.load(model_filepath)
        return model

    def __get_scaled_features(self, scalar, features):
        return scalar.transform(features)

    def __predict_hf_probability(self, model, features):
        y_prob = model.predict(features)
        return y_prob
    
    def predict(self, signal_filepath, age, sex):
        preprocessed_signal = self.__preprocess_signal(self.__load_ptbxl_signal(signal_filepath))
        feature_extractor_model = self.__load_feature_extractor_model(self.cnn_model_filepath)
        features = self.__combine_metadata_with_ecg_features(self.__extract_features(feature_extractor_model, preprocessed_signal), age, sex)
        traditional_ml_model = self.__load_traditional_ml_model(self.traditional_ml_model_filepath)
        return self.__predict_hf_probability(traditional_ml_model, self.__get_scaled_features(self.scalar, features))