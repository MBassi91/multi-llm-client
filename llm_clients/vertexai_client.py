import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel

class VertexAILLM:
    def __init__(self, model_name="gemini", project_id=None, location="us-central1",
                 max_output_tokens=2048, temperature=0, top_p=1, top_k=40):
        """
        Initialize the Vertex AI LLM client with specified parameters.

        Parameters:
        - model_name (str, optional): The name of the model to use. Defaults to "gemini".
        - project_id (str, optional): Google Cloud project ID. If None, defaults to Vertex AI's default process.
        - location (str, optional): Google Cloud location for the API call. Defaults to "us-central1".
        - max_output_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2048.
        - temperature (float, optional): Controls randomness. Lower means more deterministic. Defaults to 0.
        - top_p (float, optional): Nucleus sampling parameter. Defaults to 1.
        - top_k (int, optional): Controls diversity. Defaults to 40.
        """
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        if project_id:
            vertexai.init(project=project_id, location=location)
        else:
            vertexai.init(location=location)

    def generate_text(self, text_input, system_prompt=None):
        """
        Generate text using the configured model.

        Parameters:
        - text_input (str): The specific input text for the model to generate content based on.
        - system_prompt (str, optional): The system prompt or context for the model.

        Returns:
        - str: The generated text response from the specified LLM.
        """
        text = f"{system_prompt}\n{text_input}" if system_prompt else f"{text_input}"

        parameters = {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        if self.model_name == "bison":
            model_instance = TextGenerationModel.from_pretrained("text-bison")
            response = model_instance.predict(text, **parameters)
        elif self.model_name == "gemini":
            model_instance = GenerativeModel("gemini-1.0-pro")
            response = model_instance.generate_content(
                text, generation_config={**parameters, "top_k": self.top_k}
            )
        else:
            raise ValueError("Unsupported model name. Please use 'bison' or 'gemini'.")

        return response.text
    
    
    
    def prompt_optimizer(self, system_prompt, few_shots=None, few_shots_limit=5):
        """
        A prompt optimizer for system prompts. Uses the refine LLMs system prompts. Suggested usage with Gemini 1.5 pro or above, GPTs 4 or above, Cloude 3 or above.
        Parameters:
        - system_prompt (str):  
        - few_shots (list, optional): a list of examples for few-shot learning.
        - few_shots_limit (int, optional): a limit on the number of few shots examples accepted.
        """

        if few_shots and len(few_shots) > 0:
            few_shots_text = "\n".join([f"Example {nr}:\n{shot}########\n" for nr, shot in enumerate(few_shots[:few_shots_limit], start=1)])
            optimized_prompt = f"{system_prompt}\n{few_shots_text}"
        else:
            optimized_prompt = system_prompt

        # Here, use the class's own method to generate the optimized system prompt.
        # Assuming `system_prompter` needs to be defined or obtained dynamically.
        optimized_system_prompt = """You are a specialist in optimizing LLM system prompts for more effective responses. Please optimize the prompts you receive following this checklist:

- Introduce a specific scenario or role for the AI 
- Maximize clarity and conciseness
- Provide an appropriate level of detail
- Specify the desired output format
- Incorporate relevant keywords
- Make the prompt creative and engaging for an educated adult audience

For each optimization, describe the changes you made and why they enhance the prompt. 
Then implement those changes in the revised prompt. Always end your response with "Optimized prompt: " followed by the new and improved prompt. The goal is to craft prompts that encourage open-ended conversation and elicit the AI's most insightful and articulate responses.

Prompt to be optimized:
"""

        # Call the generate_text method with the optimized prompt
        optimized_text = self.generate_text(text_input=optimized_prompt, system_prompt=optimized_system_prompt)
        optimized_text = self.generate_text(text_input=optimized_text, system_prompt="Extract the optimized prompt (exclude 'Optimized prompt' or similar from your output) from the following text:\n")

        
        return optimized_text



def format_user_agent_responses(responses):
    """
    Formats a list of dictionaries containing user and assistant messages into a single string.
    
    Parameters:
    - responses (list of dict): A list where each element is a dictionary with a single key-value pair. 
                                The key is either 'user' or 'assistant', and the value is the message string.
                                
    Returns:
    - str: A formatted string where each message is preceded by its sender, followed by a newline.
    """
    formatted_string = ""
    for response in responses:
        for sender, message in response.items():
            formatted_string += f"{sender}: {message}\n"
    return formatted_string.strip()
