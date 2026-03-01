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
    