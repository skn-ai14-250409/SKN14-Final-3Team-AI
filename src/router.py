from typing import Literal

from pydantic import BaseModel, Field

from src.slm.slm import SLM

class RouterResponse(BaseModel):
    category: Literal[
        "general_banking_FAQs",
        "industry_policies_and_regulations",
        "company_rules",
        "company_products"
    ] = Field(
        ...,
        description="Classify the query into one of the following categories: \
        1. general_banking_faqs – for common, non-company-specific banking knowledge (e.g., what is a checking account); \
        2. industry_policies_and_regulations – for banking industry compliance or regulatory queries (e.g., KYC, AML, Basel III); \
        3. company_rules – for internal rules and HR policies of our bank (e.g., vacation policy, dress code); \
        4. company_products – for questions about specific products offered by our bank (e.g., loan products, account types, interest rates)."
    )

class Router:
    def __init__(self):
        self.slm = SLM()
        self.router_response_cls = RouterResponse

    def route_prompt(self, prompt) -> str:
        prompt_category: RouterResponse = self.slm.get_structured_output(
            prompt,
            self.router_response_cls
        )
        return prompt_category.category 