from dataclasses import dataclass

# 1. Data Ingestion Artifacts:
@dataclass
class DataIngestionArtifacts:
    training_file_path: str
    test_file_path: str