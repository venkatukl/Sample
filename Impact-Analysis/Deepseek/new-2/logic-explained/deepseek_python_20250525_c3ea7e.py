# Only describe important classes (skip POJOs/DTOs)
def needs_responsibility(file_content: str) -> bool:
    return any(
        annotation in file_content 
        for annotation in ["@Service", "@RestController", "@Component"]
    )