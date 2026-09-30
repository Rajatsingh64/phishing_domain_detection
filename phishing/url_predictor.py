from phishing.url_extractor import extract_url_features


def predictor(model, url: str, model_feature_names_file_path: str):
    """Return the model prediction for the URL without manual whitelist/blacklist overrides."""
    df = extract_url_features(url=url, model_feature_names_path=model_feature_names_file_path)

    y_pred = model.predict(df)[0]
    y_proba = model.predict_proba(df)

    return y_pred, y_proba
