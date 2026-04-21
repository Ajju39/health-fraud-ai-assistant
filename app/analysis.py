import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load and clean the healthcare fraud dataset.
    """
    df = pd.read_csv(path)

    # Fill missing values
    if "Insurance_Type" in df.columns:
        df["Insurance_Type"] = df["Insurance_Type"].fillna("Unknown")

    if "Provider_Specialty" in df.columns:
        df["Provider_Specialty"] = df["Provider_Specialty"].fillna("Unknown")

    if "Prior_Visits_12m" in df.columns:
        df["Prior_Visits_12m"] = df["Prior_Visits_12m"].fillna(0)

    # Convert date columns if present
    if "Claim_Submission_Date" in df.columns:
        df["Claim_Submission_Date"] = pd.to_datetime(
            df["Claim_Submission_Date"], errors="coerce"
        )

    if "Service_Date" in df.columns:
        df["Service_Date"] = pd.to_datetime(df["Service_Date"], errors="coerce")

    return df


def fraud_rate(df: pd.DataFrame) -> float:
    """
    Return overall fraud percentage.
    """
    if "Is_Fraud" not in df.columns or len(df) == 0:
        return 0.0
    return round(df["Is_Fraud"].mean() * 100, 2)


def fraud_by_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fraud summary grouped by provider specialty.
    """
    result = (
        df.groupby("Provider_Specialty", dropna=False)
        .agg(
            total_claims=("Claim_ID", "count"),
            fraud_claims=("Is_Fraud", "sum"),
            avg_claim_amount=("Claim_Amount", "mean"),
            avg_approved_amount=("Approved_Amount", "mean"),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = round(
        (result["fraud_claims"] / result["total_claims"]) * 100, 2
    )

    return result.sort_values(
        by=["fraud_rate_pct", "fraud_claims"], ascending=[False, False]
    )


def fraud_by_insurance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fraud summary grouped by insurance type.
    """
    result = (
        df.groupby("Insurance_Type", dropna=False)
        .agg(
            total_claims=("Claim_ID", "count"),
            fraud_claims=("Is_Fraud", "sum"),
            avg_claim_amount=("Claim_Amount", "mean"),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = round(
        (result["fraud_claims"] / result["total_claims"]) * 100, 2
    )

    return result.sort_values(
        by=["fraud_rate_pct", "fraud_claims"], ascending=[False, False]
    )


def fraud_by_claim_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fraud summary grouped by claim status.
    """
    result = (
        df.groupby("Claim_Status", dropna=False)
        .agg(
            total_claims=("Claim_ID", "count"),
            fraud_claims=("Is_Fraud", "sum"),
            avg_claim_amount=("Claim_Amount", "mean"),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = round(
        (result["fraud_claims"] / result["total_claims"]) * 100, 2
    )

    return result.sort_values(
        by=["fraud_rate_pct", "fraud_claims"], ascending=[False, False]
    )


def high_risk_providers(df: pd.DataFrame, min_claims: int = 20) -> pd.DataFrame:
    """
    Identify providers with high fraud rates.
    """
    result = (
        df.groupby("Provider_ID", dropna=False)
        .agg(
            total_claims=("Claim_ID", "count"),
            fraud_claims=("Is_Fraud", "sum"),
            avg_claim_amount=("Claim_Amount", "mean"),
            avg_approved_amount=("Approved_Amount", "mean"),
            avg_days_to_claim=("Days_Between_Service_and_Claim", "mean"),
            avg_monthly_claims=("Number_of_Claims_Per_Provider_Monthly", "mean"),
        )
        .reset_index()
    )

    result = result[result["total_claims"] >= min_claims].copy()

    result["fraud_rate_pct"] = round(
        (result["fraud_claims"] / result["total_claims"]) * 100, 2
    )

    return result.sort_values(
        by=["fraud_rate_pct", "fraud_claims", "avg_claim_amount"],
        ascending=[False, False, False],
    )


def suspicious_claims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify suspicious claims using simple threshold-based logic.
    """
    claim_amount_threshold = df["Claim_Amount"].quantile(0.95)
    days_to_claim_threshold = df["Days_Between_Service_and_Claim"].quantile(0.95)
    provider_monthly_claims_threshold = df[
        "Number_of_Claims_Per_Provider_Monthly"
    ].quantile(0.95)

    result = df[
        (df["Claim_Amount"] > claim_amount_threshold)
        | (df["Days_Between_Service_and_Claim"] > days_to_claim_threshold)
        | (
            df["Number_of_Claims_Per_Provider_Monthly"]
            > provider_monthly_claims_threshold
        )
    ].copy()

    result["Suspicion_Reason"] = ""

    result.loc[
        result["Claim_Amount"] > claim_amount_threshold, "Suspicion_Reason"
    ] += "High Claim Amount; "

    result.loc[
        result["Days_Between_Service_and_Claim"] > days_to_claim_threshold,
        "Suspicion_Reason",
    ] += "Late Claim Submission; "

    result.loc[
        result["Number_of_Claims_Per_Provider_Monthly"]
        > provider_monthly_claims_threshold,
        "Suspicion_Reason",
    ] += "High Provider Claim Volume; "

    cols_to_show = [
        "Claim_ID",
        "Provider_ID",
        "Patient_ID",
        "Claim_Amount",
        "Approved_Amount",
        "Claim_Status",
        "Provider_Specialty",
        "Insurance_Type",
        "Visit_Type",
        "Days_Between_Service_and_Claim",
        "Number_of_Claims_Per_Provider_Monthly",
        "Is_Fraud",
        "Suspicion_Reason",
    ]

    existing_cols = [col for col in cols_to_show if col in result.columns]

    return result[existing_cols].sort_values(
        by=["Is_Fraud", "Claim_Amount"], ascending=[False, False]
    )


def get_kpis(df: pd.DataFrame) -> dict:
    """
    Return summary KPIs for dashboard display.
    """
    total_claims = len(df)
    total_fraud_claims = int(df["Is_Fraud"].sum())
    fraud_pct = fraud_rate(df)
    total_claim_amount = round(df["Claim_Amount"].sum(), 2)
    total_approved_amount = round(df["Approved_Amount"].sum(), 2)

    return {
        "total_claims": total_claims,
        "total_fraud_claims": total_fraud_claims,
        "fraud_pct": fraud_pct,
        "total_claim_amount": total_claim_amount,
        "total_approved_amount": total_approved_amount,
    }