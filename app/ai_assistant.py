import json
import os
from typing import Any, Dict, List

from openai import OpenAI
from mcp_server.tools import FraudTools


TOOLS = [
    {
        "type": "function",
        "name": "get_overall_summary",
        "description": "Get overall fraud KPIs for the healthcare claims dataset.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_fraud_by_specialty",
        "description": "Get fraud analysis grouped by provider specialty.",
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top rows to return",
                    "default": 10
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_fraud_by_insurance",
        "description": "Get fraud analysis grouped by insurance type.",
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top rows to return",
                    "default": 10
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_fraud_by_claim_status",
        "description": "Get fraud analysis grouped by claim status.",
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top rows to return",
                    "default": 10
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_high_risk_providers",
        "description": "Get providers with the highest fraud rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_claims": {
                    "type": "integer",
                    "description": "Minimum claims required per provider",
                    "default": 20
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top providers to return",
                    "default": 10
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_suspicious_claims",
        "description": "Get suspicious claims based on anomaly-style thresholds.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of suspicious claims to return",
                    "default": 10
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_claim_details",
        "description": "Get details for a specific claim by Claim_ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim_id": {
                    "type": "string",
                    "description": "Claim identifier"
                }
            },
            "required": ["claim_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_provider_details",
        "description": "Get claim details for a specific provider by Provider_ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "provider_id": {
                    "type": "string",
                    "description": "Provider identifier"
                }
            },
            "required": ["provider_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "ask_dataset_metadata",
        "description": "Get dataset metadata such as columns and row counts.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


class AIFraudAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tools = FraudTools()

    def _run_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "get_overall_summary":
            return self.tools.get_overall_summary()
        if name == "get_fraud_by_specialty":
            return self.tools.get_fraud_by_specialty(top_n=arguments.get("top_n", 10))
        if name == "get_fraud_by_insurance":
            return self.tools.get_fraud_by_insurance(top_n=arguments.get("top_n", 10))
        if name == "get_fraud_by_claim_status":
            return self.tools.get_fraud_by_claim_status(top_n=arguments.get("top_n", 10))
        if name == "get_high_risk_providers":
            return self.tools.get_high_risk_providers(
                min_claims=arguments.get("min_claims", 20),
                top_n=arguments.get("top_n", 10),
            )
        if name == "get_suspicious_claims":
            return self.tools.get_suspicious_claims(limit=arguments.get("limit", 10))
        if name == "get_claim_details":
            return self.tools.get_claim_details(claim_id=arguments["claim_id"])
        if name == "get_provider_details":
            return self.tools.get_provider_details(provider_id=arguments["provider_id"])
        if name == "ask_dataset_metadata":
            return self.tools.ask_dataset_metadata()

        return {"error": f"Unknown tool: {name}"}

    def ask(self, user_question: str) -> Dict[str, Any]:
        instructions = (
            "You are a healthcare fraud analytics assistant. "
            "Use the available tools when needed. "
            "Give concise, business-friendly answers. "
            "When tool results include tables, summarize the top insight first, "
            "then reference the returned data. "
            "Do not invent data that is not in the tool output."
        )

        first_response = self.client.responses.create(
            model="gpt-5.4-mini",
            instructions=instructions,
            input=user_question,
            tools=TOOLS,
        )

        tool_outputs: List[Dict[str, Any]] = []
        display_table = None

        for item in first_response.output:
            if item.type == "function_call":
                tool_name = item.name
                arguments = json.loads(item.arguments or "{}")
                result = self._run_tool(tool_name, arguments)

                if isinstance(result, list):
                    display_table = result
                elif isinstance(result, dict):
                    display_table = [result]

                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result),
                    }
                )

        if tool_outputs:
            second_response = self.client.responses.create(
                model="gpt-5.4-mini",
                instructions=instructions,
                input=tool_outputs,
                previous_response_id=first_response.id,
            )
            final_text = second_response.output_text
        else:
            final_text = first_response.output_text

        return {
            "answer": final_text,
            "table": display_table,
        }