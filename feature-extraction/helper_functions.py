import pandas as pd

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

def get_features_dataframe_with_metadata(feature_model, dataset_path, ecg_signals, model_type = None):
    features_df = extract_features_as_dataframe(feature_model, ecg_signals)
    metadata_df = get_important_metadata(dataset_path)
    return combine_features_and_metadata(features_df, metadata_df)