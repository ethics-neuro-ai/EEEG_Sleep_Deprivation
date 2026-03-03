from src.preprocess import preprocess_dataset
from src.train import train_model
from src.evaluate import evaluate_model

DATA_PATH = "/Volumes/backup/eeg_sleep_deprivation"
SAVE_PATH = "./data/preprocessed"
MODEL_PATH = "./saved_models/eegnet_sleep_model.keras"

if __name__ == "__main__":

    preprocess_dataset(DATA_PATH, SAVE_PATH)

    model, history, X_test, y_test = train_model(
        SAVE_PATH,
        MODEL_PATH
    )

    evaluate_model(model, X_test, y_test)