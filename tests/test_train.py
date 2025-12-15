import os

def test_train_script_exists():
    assert os.path.exists("src/train.py")

def test_models_directory_created():
    os.makedirs("models", exist_ok=True)
    assert os.path.isdir("models")
