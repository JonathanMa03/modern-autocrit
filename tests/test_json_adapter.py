from backend.services.llm.base import BaseLLMProvider, LLMResponse
from backend.services.llm.json_adapter import StructuredJsonAdapter


class RecordingProvider(BaseLLMProvider):
    provider_name = "test"

    def __init__(self) -> None:
        self.max_output_tokens = 0

    @property
    def default_model(self) -> str:
        return "test-model"

    def generate(
        self, prompt: str, *, model: str, temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ) -> LLMResponse:
        self.max_output_tokens = max_output_tokens
        return LLMResponse(text='{"rows": []}', provider="test", model=model)


def test_structured_adapter_reserves_enough_output_for_large_batches() -> None:
    provider = RecordingProvider()
    result = StructuredJsonAdapter(provider, model="test-model").generate_json(
        "extract", output_schema={"type": "object"}
    )

    assert result == {"rows": []}
    assert provider.max_output_tokens == 16000
