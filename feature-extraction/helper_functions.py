import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import F1Score

f1 = F1Score(num_classes=1, average='micro', threshold=0.5, name='f1')

def load_feature_extractor_model(full_model_path):
    full_model = load_model(full_model_path, custom_objects={'f1': f1})
    if len(full_model.layers) == 0:
        raise ValueError(f"No layers found in model loaded from: {full_model_path}")
    feature_model = full_model.layers[0]
    
    return feature_model

def get_important_metadata(dataset_path):
    df = pd.read_csv(dataset_path)
    return df[['age', 'sex', 'label']]

def extract_features_as_dataframe(model, data, index=None):
    
    features = model.predict(data, verbose=0)
    
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    columns = [f'feature_{i}' for i in range(features.shape[1])]

    return pd.DataFrame(features, columns=columns, index=index)

def combine_features_and_metadata(features_df, metadata_df):
    return pd.concat([features_df, metadata_df], axis=1)

def get_features_dataframe_with_metadata(full_model_path, dataset_path, ecg_signals):
    feature_model = load_feature_extractor_model(full_model_path)
    features_df = extract_features_as_dataframe(feature_model, ecg_signals)
    metadata_df = get_important_metadata(dataset_path)
    return combine_features_and_metadata(features_df, metadata_df)