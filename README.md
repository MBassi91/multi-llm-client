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
from vertexai_client import VertexAILLM

# Initialize the Vertex AI LLM client
client = VertexAILLM(model_name="gemini")

# Generate text using the client
text_input = "Once upon a time, in a faraway land, there was a"
response = client.generate_text(text_input)
print("Generated text:")
print(response)
```

### Prompt Optimization

```python
# Optimize a system prompt using few-shot examples
system_prompt = "Write a short story about a magical adventure."
few_shots = [
    "Example 1: In a enchanted forest, a brave young girl named Lily discovered a hidden portal that led her to a world filled with talking animals and magical creatures. With the help of her new friends, she embarked on a quest to find a legendary treasure and save the magical realm from an evil sorcerer.",
    "Example 2: Max, a curious boy, stumbled upon an old book in his grandfather's attic. As he opened the book, he was transported to a mystical land where dragons roamed the skies and wizards cast powerful spells. Max soon realized that he was the chosen one, destined to bring peace to the land by uniting the feuding dragons and wizards."
]
optimized_prompt = client.prompt_optimizer(
    system_prompt,
    few_shots=few_shots,
    few_shots_limit=2
)
print("\nOptimized prompt:")
print(optimized_prompt)

# Generate text using the optimized prompt
optimized_response = client.generate_text(optimized_prompt)
print("\nGenerated text with optimized prompt:")
print(optimized_response)
```

## Adding New LLM Providers

1. Create a new client file in the `llm_clients/` directory, e.g., `openai_client.py`.
2. Implement the `BaseLLMClient` interface from `llm_clients/base_client.py`.
3. Add usage examples in the `llm_clients/` directory.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.