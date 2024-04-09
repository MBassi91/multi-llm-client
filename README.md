# Multi-LLM Client

A Python library providing a unified interface to interact with various Large Language Models (LLMs), including Google's Vertex AI and OpenAI's GPT models.

## Features

- Abstracted LLM client interface for easy integration of multiple providers
- Built-in support for Google Vertex AI LLMs (Gemini and Bison)
- Prompt optimization using few-shot learning
- Extensible design to add more LLM providers

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MBassi91/multi-llm-client.git
   cd multi-llm-client
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Vertex AI Client

```python
from llm_clients.vertexai_client import VertexAILLM

client = VertexAILLM(model_name="gemini", project_id="your-project-id")
response = client.generate_text("Hello, world!")
print(response)
```

### Prompt Optimization

```python
optimized_prompt = client.prompt_optimizer(
    "Tell me about the history of ancient Egypt.",
    few_shots=["Example 1: Ancient Egypt was...", "Example 2: The pyramids were..."]
)
response = client.generate_text(optimized_prompt)
print(response)
```

## Adding New LLM Providers

1. Create a new client file in the `llm_clients/` directory, e.g., `openai_client.py`.
2. Implement the `BaseLLMClient` interface from `llm_clients/base_client.py`.
3. Add usage examples in the `examples/` directory.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.