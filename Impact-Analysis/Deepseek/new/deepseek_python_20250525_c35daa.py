from jsonschema import validate, ValidationError
import json

def validate_response(llm_response: str) -> dict:
    try:
        data = json.loads(llm_response)
        validate(instance=data, schema=DEPENDENCY_SCHEMA)
        return data
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Invalid response: {e}")
        return None

def get_valid_llm_response(prompt: str, max_retries=3) -> dict:
    for _ in range(max_retries):
        response = llm_api.generate(prompt)
        validated = validate_response(response)
        if validated:
            return validated
        print("Retrying...")
    raise ValueError("Max retries reached for invalid LLM response")