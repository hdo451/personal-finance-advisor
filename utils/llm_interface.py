from openai import OpenAI
from typing import Dict, Optional

class LLMInterface:
    """
    Centralized OpenAI ChatGPT management
    Tracks every call for cost analysis
    """

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.call_count = 0    # Track total calls
        self.total_cost = 0.0  # Track estimated cost

    def make_call(self, prompt: str, system_prompt: str = None, expect_json: bool = False) -> Optional[str]:
        """
        Single point for ALL LLM calls in your system
        Every agent must use this method
        """
        self.call_count += 1
        print(f"🤖 LLM Call #{self.call_count} - OpenAI API")

        try:
            # Build OpenAI-compatible messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Make ChatGPT API call
            request_payload = {
                "model": "gpt-4o-mini",
                "max_tokens": 3000,
                "messages": messages,
                "temperature": 0,
            }
            if expect_json:
                request_payload["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**request_payload)

            # Track cost (using project estimate)
            self.total_cost += 0.002

            content = response.choices[0].message.content
            if not content:
                return None
            return content
        
        except Exception as e:
            error_msg = f"❌ OpenAI API call failed: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
    def get_metrics(self) -> Dict:
        """Report usage statistics"""
        return {
            'total_calls': self.call_count,
            'estimated_cost': self.total_cost,
            'average_cost_per_call': 0.002
        }
        
    def reset_counters(self):
        """Reset for new analysis session"""
        self.call_count = 0
        self.total_cost = 0.0