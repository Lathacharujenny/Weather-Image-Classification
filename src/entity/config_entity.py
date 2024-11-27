from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    data_zip_path: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass
class DataSplittingConfig:
    root_dir: Path
    data_path: Path

@dataclass
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    labels_path: Path
    models: Path
    visualize_results: Path
    efficientnet_model: Path
    resnet50_model: Path
    resnet152v2_model: Path
    mobilenet_model:Path 
    vgg19_model: Path
    xception_model: Path
    inceptionresnetv2_model: Path
    vgg16_model: Path
    resnet101_model: Path
    densenet201_model: Path

@dataclass
class ModelEvaluateConfig:
    root_dir: Path
    scores_json: Path
    scores_csv: Path
    visualize_scores: Path