import json

from tools import FraudTools


def main():
    tools = FraudTools()

    menu = """
Healthcare Fraud MCP Tool Server (Prototype)

Choose an option:
1. Overall Summary
2. Fraud by Specialty
3. Fraud by Insurance
4. Fraud by Claim Status
5. High Risk Providers
6. Suspicious Claims
7. Claim Details
8. Provider Details
9. Dataset Metadata
0. Exit
"""

    while True:
        print(menu)
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            print(json.dumps(tools.get_overall_summary(), indent=2))

        elif choice == "2":
            top_n = int(input("Top N specialties: ") or "10")
            print(json.dumps(tools.get_fraud_by_specialty(top_n=top_n), indent=2))

        elif choice == "3":
            top_n = int(input("Top N insurance types: ") or "10")
            print(json.dumps(tools.get_fraud_by_insurance(top_n=top_n), indent=2))

        elif choice == "4":
            top_n = int(input("Top N claim statuses: ") or "10")
            print(json.dumps(tools.get_fraud_by_claim_status(top_n=top_n), indent=2))

        elif choice == "5":
            min_claims = int(input("Minimum claims per provider: ") or "20")
            top_n = int(input("Top N providers: ") or "10")
            print(
                json.dumps(
                    tools.get_high_risk_providers(
                        min_claims=min_claims,
                        top_n=top_n,
                    ),
                    indent=2,
                )
            )

        elif choice == "6":
            limit = int(input("How many suspicious claims?: ") or "20")
            print(json.dumps(tools.get_suspicious_claims(limit=limit), indent=2))

        elif choice == "7":
            claim_id = input("Enter Claim_ID: ").strip()
            print(json.dumps(tools.get_claim_details(claim_id), indent=2))

        elif choice == "8":
            provider_id = input("Enter Provider_ID: ").strip()
            print(json.dumps(tools.get_provider_details(provider_id), indent=2))

        elif choice == "9":
            print(json.dumps(tools.ask_dataset_metadata(), indent=2))

        elif choice == "0":
            print("Exiting tool server.")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()