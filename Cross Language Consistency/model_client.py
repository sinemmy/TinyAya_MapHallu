# Thin wrapper around the Cohere API.
import os
import time
import cohere
from config.experiment_config import (
    COHERE_API_KEY_ENV,
    TEMPERATURE,
    MAX_TOKENS,
    REQUEST_DELAY_S,
)


class AyaClient:
    def __init__(self):
        api_key = os.environ.get(COHERE_API_KEY_ENV)
        if not api_key:
            raise EnvironmentError(
                f"Missing API key. Set the {COHERE_API_KEY_ENV} environment variable.\n"
                f"Run: export COHERE_API_KEY=your_key_here"
            )
        self.client = cohere.ClientV2(api_key=api_key)

    def query(
        self,
        model_name: str,
        prompt: str,
        n_samples: int = 1,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> list[str | None]:
        responses = []
        for i in range(n_samples):
            try:
                response = self.client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.message.content[0].text.strip()
                responses.append(text)
            except Exception as e:
                print(f"  [API error] model={model_name} sample={i}: {e}")
                responses.append(None)
            time.sleep(REQUEST_DELAY_S)
        return responses