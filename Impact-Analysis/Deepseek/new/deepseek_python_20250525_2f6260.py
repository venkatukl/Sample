# Define the EXACT schema you want (adapt as needed)
DEPENDENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "class_name": {"type": "string"},
        "package": {"type": "string"},
        "repository": {"type": "string"},
        "dependencies": {
            "type": "object",
            "properties": {
                "intra_repo": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_class": {"type": "string"},
                            "type": {"enum": ["method_call", "inheritance", "interface_impl"]}
                        }
                    }
                },
                "inter_repo": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_repo": {"type": "string"},
                            "target_service": {"type": "string"},
                            "protocol": {"enum": ["HTTP", "gRPC", "Kafka"]}
                        }
                    }
                }
            }
        }
    },
    "required": ["class_name", "dependencies"]
}