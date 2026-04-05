from dataclasses import dataclass
from typing import Optional
# 1. Data Ingestion Artifacts:
@dataclass
class DataIngestionArtifacts:
    training_file_path: str
    test_file_path: str


# 2. Data Validation Artifacts:
@dataclass
class DataValidationArtifacts:
    validation_status: bool
    valid_train_file_path: Optional[str] = None
    valid_test_file_path: Optional[str] = None
    invalid_train_file_path: Optional[str] = None
    invalid_test_file_path: Optional[str] = None
    drift_report_file_path: Optional[str] = None


# 3. Data Cleaning Artifacts:
@dataclass
class DataCleaningArtifacts:
    clipped_training_file_path: Optional[str] = None
    clipped_testing_file_path: Optional[str] = None
    unclipped_training_file_path: Optional[str] = None
    unclipped_testing_file_path: Optional[str] = None
    simple_imputer_file_path: Optional[str] = None
    caps_file_path_yaml: Optional[str] = None

# 4. Feature engineering artifacts:
@dataclass
class FeatureEngineeringArtifacts:
    clipped_training_file_path: Optional[str] = None
    clipped_testing_file_path: Optional[str] = None
    unclipped_training_file_path: Optional[str] = None
    unclipped_testing_file_path: Optional[str] = None


# 5. Data pre-processing artifacts:
@dataclass
class PreProcessingArtifacts:
    pre_processor_file_path: str
    X_path: str
    y_path: str