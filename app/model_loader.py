from catboost import CatBoostClassifier

def load_churn_model(model_path: str):
    """Загружаем обученную модель оттока.
    Args:
        model_path (str): Путь до модели.
    """
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


if __name__ == "__main__":
   
    
    # загружаем модель
    model = load_churn_model(model_path='models/catboost_churn_model.bin')
    
    # если загрузка успешна, выводим имена признаков
    if model is not None:
        print(f"Model parameter names: {model.feature_names_}")
