import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

# Allow imports from app/
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
APP_DIR = os.path.join(PROJECT_ROOT, "app")

if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from analysis import (  # noqa: E402
    load_data,
    get_kpis,
    fraud_by_specialty,
    fraud_by_insurance,
    fraud_by_claim_status,
    high_risk_providers,
    suspicious_claims,
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "healthcare_fraud_detection.csv")


class FraudTools:
    def __init__(self, data_path: str = DATA_PATH):
        self.df = load_data(data_path)

    def get_overall_summary(self) -> Dict[str, Any]:
        """
        Return KPI summary of the fraud dataset.
        """
        return get_kpis(self.df)

    def get_fraud_by_specialty(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Return fraud summary grouped by provider specialty.
        """
        result = fraud_by_specialty(self.df).head(top_n).copy()
        return self._clean_records(result)

    def get_fraud_by_insurance(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Return fraud summary grouped by insurance type.
        """
        result = fraud_by_insurance(self.df).head(top_n).copy()
        return self._clean_records(result)

    def get_fraud_by_claim_status(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Return fraud summary grouped by claim status.
        """
        result = fraud_by_claim_status(self.df).head(top_n).copy()
        return self._clean_records(result)

    def get_high_risk_providers(
        self, min_claims: int = 20, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Return providers with highest fraud rates.
        """
        result = high_risk_providers(self.df, min_claims=min_claims).head(top_n).copy()
        return self._clean_records(result)

    def get_suspicious_claims(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Return suspicious claims identified using threshold logic.
        """
        result = suspicious_claims(self.df).head(limit).copy()
        return self._clean_records(result)

    def get_claim_details(self, claim_id: str) -> Dict[str, Any]:
        """
        Return details for a single claim.
        """
        result = self.df[self.df["Claim_ID"].astype(str) == str(claim_id)].copy()

        if result.empty:
            return {"error": f"No claim found for Claim_ID={claim_id}"}

        row = result.iloc[0].to_dict()
        return self._convert_record(row)

    def get_provider_details(self, provider_id: str) -> List[Dict[str, Any]]:
        """
        Return claims associated with a single provider.
        """
        result = self.df[self.df["Provider_ID"].astype(str) == str(provider_id)].copy()

        if result.empty:
            return [{"error": f"No provider found for Provider_ID={provider_id}"}]

        cols = [
            "Provider_ID",
            "Claim_ID",
            "Patient_ID",
            "Claim_Amount",
            "Approved_Amount",
            "Claim_Status",
            "Provider_Specialty",
            "Insurance_Type",
            "Visit_Type",
            "Is_Fraud",
        ]
        existing_cols = [c for c in cols if c in result.columns]
        return self._clean_records(result[existing_cols].head(25))

    def ask_dataset_metadata(self) -> Dict[str, Any]:
        """
        Return dataset metadata like row count, column names, and dtypes.
        """
        return {
            "row_count": int(len(self.df)),
            "column_count": int(len(self.df.columns)),
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
        }

    def _clean_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        records = df.to_dict(orient="records")
        return [self._convert_record(record) for record in records]

    def _convert_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, pd.Timestamp):
                cleaned[key] = value.isoformat()
            elif hasattr(value, "item"):
                cleaned[key] = value.item()
            else:
                cleaned[key] = value
        return cleaned


if __name__ == "__main__":
    tools = FraudTools()

    print("=== OVERALL SUMMARY ===")
    print(json.dumps(tools.get_overall_summary(), indent=2))

    print("\n=== FRAUD BY SPECIALTY ===")
    print(json.dumps(tools.get_fraud_by_specialty(), indent=2))

    print("\n=== HIGH RISK PROVIDERS ===")
    print(json.dumps(tools.get_high_risk_providers(), indent=2))

    print("\n=== SUSPICIOUS CLAIMS ===")
    print(json.dumps(tools.get_suspicious_claims(), indent=2))