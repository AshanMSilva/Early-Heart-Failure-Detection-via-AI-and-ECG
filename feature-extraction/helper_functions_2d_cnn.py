import pandas as pd
from tensorflow.keras.models import load_model, Model

def load_2d_cnn_feature_extractor_model(full_model_path):
    """
    Loads a trained 2D CNN model and returns a feature extraction model 
    that outputs from the second-to-last layer.
    """
    full_model = load_model(full_model_path)
    if len(full_model.layers) == 0:
        raise ValueError(f"No layers found in model loaded from: {full_model_path}")
    
    # Extract features from the layer before the final output
    feature_model = Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
    return feature_model

def get_important_metadata(dataset_path):
    """
    Loads age, sex, and label columns from a metadata CSV file.
    """
    df = pd.read_csv(dataset_path)
    return df[['age', 'sex', 'label']]

def extract_features_as_dataframe(model, data, index=None):
    """
    Uses a feature extraction model to get feature vectors from input data.
    Returns a DataFrame.
    """
    features = model.predict(data, verbose=0)

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    columns = [f'feature_{i}' for i in range(features.shape[1])]
    return pd.DataFrame(features, columns=columns, index=index)

def combine_features_and_metadata(features_df, metadata_df):
    """
    Combines feature DataFrame with metadata (age, sex, label).
    """
    return pd.concat([features_df, metadata_df], axis=1)

def get_features_dataframe_with_metadata(full_model_path, dataset_path, ecg_signals):
    """
    Main function to generate the final DataFrame containing extracted features 
    from the 2D CNN and metadata.
    """
    feature_model = load_2d_cnn_feature_extractor_model(full_model_path)
    features_df = extract_features_as_dataframe(feature_model, ecg_signals)
    metadata_df = get_important_metadata(dataset_path)
    return combine_features_and_metadata(features_df, metadata_df)
