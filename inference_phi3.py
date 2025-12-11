"""
Inference script for fine-tuned Phi-3 model.

Use this to test and deploy the fine-tuned Phi-3 model for financial recommendations.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class Phi3FinancialAdvisor:
    """Wrapper for fine-tuned Phi-3 financial recommendation model."""

    def __init__(
        self,
        model_dir: str,
        base_model: str = "microsoft/phi-3-mini-4k-instruct",
        use_lora: bool = True,
    ):
        """Initialize the financial advisor model.

        Args:
            model_dir: Directory of fine-tuned model
            base_model: Base model name if using LoRA
            use_lora: Whether model uses LoRA
        """
        print(f"Loading model from {model_dir}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_lora = use_lora

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )

        # Load model
        if use_lora:
            print(f"Loading base model {base_model}...")
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                trust_remote_code=True,
            )
            print(f"Loading LoRA weights from {model_dir}...")
            self.model = PeftModel.from_pretrained(base_model_obj, model_dir)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                trust_remote_code=True,
            )

        self.model.eval()
        print("✓ Model loaded and ready for inference\n")

    def create_customer_prompt(self, customer_profile: dict) -> str:
        """
        Create a prompt from customer profile.

        Args:
            customer_profile (dict): Customer financial profile

        Returns:
            str: Formatted prompt
        """
        prompt = f"""<system>You are an expert financial advisor specializing in age-group-based recommendations. Provide personalized, data-driven financial guidance.</system>
<user>Provide comprehensive financial recommendations for:

CUSTOMER PROFILE:
- Age Group: {customer_profile.get('age_group', 'Unknown')}
- Age: {customer_profile.get('age', 0)}
- Risk Level: {customer_profile.get('risk_level', 'Unknown')} (Score: {customer_profile.get('risk_score', 0)}/100)

FINANCIAL METRICS:
- Account Balance: ${customer_profile.get('account_balance', 0):.2f}
- Loan Amount: ${customer_profile.get('loan_amount', 0):.2f}
- Interest Rate: {customer_profile.get('interest_rate', 0):.2f}%
- Credit Limit: ${customer_profile.get('credit_limit', 0):.2f}
- Credit Card Balance: ${customer_profile.get('credit_card_balance', 0):.2f}
- Loan Status: {customer_profile.get('loan_status', 'Unknown')}

Please provide detailed recommendations addressing: risk mitigation, product optimization, debt management, and age-appropriate financial planning.</user>
<assistant>"""
        return prompt

    def generate_recommendation(
        self,
        customer_profile: dict = None,
        prompt: str = None,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate financial recommendation for a customer.

        Args:
            customer_profile (dict): Customer financial profile
            prompt (str): Custom prompt (overrides profile)
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter

        Returns:
            str: Generated recommendation
        """
        # Create prompt
        if prompt is None:
            if customer_profile is None:
                raise ValueError("Either customer_profile or prompt must be provided")
            prompt = self.create_customer_prompt(customer_profile)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                do_sample=True,
                use_cache=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def batch_recommendations(
        self,
        customer_profiles: list,
        max_new_tokens: int = 500,
    ) -> list:
        """
        Generate recommendations for multiple customers.

        Args:
            customer_profiles (list): List of customer profiles
            max_new_tokens (int): Maximum tokens to generate per customer

        Returns:
            list: List of generated recommendations
        """
        recommendations = []

        for i, profile in enumerate(customer_profiles):
            print(f"Generating recommendation {i+1}/{len(customer_profiles)}...")
            rec = self.generate_recommendation(
                customer_profile=profile,
                max_new_tokens=max_new_tokens,
            )
            recommendations.append(rec)

        return recommendations


# Example usage
if __name__ == "__main__":

    print("=" * 80)
    print("PHI-3 FINANCIAL ADVISOR - INFERENCE")
    print("=" * 80)

    # Initialize model
    model_dir = "drive/MyDrive/phi3_finetuned_simple"
    advisor = Phi3FinancialAdvisor(model_dir=model_dir, use_lora=True)

    # Example customer profiles
    sample_profiles = [
        {
            "age_group": "35-49: Established Professionals",
            "age": 42,
            "risk_level": "Medium",
            "risk_score": 55,
            "account_balance": 12500.00,
            "loan_amount": 150000.00,
            "interest_rate": 4.5,
            "credit_limit": 25000.00,
            "credit_card_balance": 8500.00,
            "loan_status": "Approved",
        },
        {
            "age_group": "25-34: Young Professionals",
            "age": 28,
            "risk_level": "High",
            "risk_score": 72,
            "account_balance": 3500.00,
            "loan_amount": 50000.00,
            "interest_rate": 6.5,
            "credit_limit": 15000.00,
            "credit_card_balance": 12000.00,
            "loan_status": "Approved",
        },
        {
            "age_group": "60+: Retirement",
            "age": 68,
            "risk_level": "Low",
            "risk_score": 25,
            "account_balance": 450000.00,
            "loan_amount": 0.00,
            "interest_rate": 3.0,
            "credit_limit": 50000.00,
            "credit_card_balance": 2000.00,
            "loan_status": "Approved",
        },
    ]

    # Generate recommendations
    print("\nGenerating recommendations for sample customers...\n")
    print("=" * 80)

    for i, profile in enumerate(sample_profiles):
        print(f"\n{'='*80}")
        print(f"CUSTOMER {i+1}: {profile['age_group']} (Age: {profile['age']})")
        print(f"Risk Score: {profile['risk_score']}/100 ({profile['risk_level']})")
        print(f"Account Balance: ${profile['account_balance']:.2f}")
        print("=" * 80)

        recommendation = advisor.generate_recommendation(
            customer_profile=profile,
            max_new_tokens=400,
        )

        # Extract just the assistant response
        if "<assistant>" in recommendation:
            response_start = recommendation.find("<assistant>") + len("<assistant>")
            recommendation_text = recommendation[response_start:].strip()
        else:
            recommendation_text = recommendation

        print("\nRECOMMENDATION:")
        print("-" * 80)
        print(recommendation_text)
        print("-" * 80)

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
