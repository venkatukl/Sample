from typing import List, Dict, Tuple
import yaml
import json
import re

def find_java_files(repo_path: Path) -> List[Path]:
    """Find all Java files in a repository"""
    return list(repo_path.glob('**/*.java'))

def find_config_files(repo_path: Path) -> Dict[str, str]:
    """Find application config files"""
    config_files = {
        'yaml': list(repo_path.glob('**/application.yml')),
        'properties': list(repo_path.glob('**/application.properties'))
    }
    return {k: v[0] if v else None for k, v in config_files.items()}

def parse_config_file(config_path: Path) -> Dict:
    """Parse YAML or properties config file"""
    if not config_path:
        return {}
    
    try:
        if config_path.suffix == '.yml':
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:  # .properties
            config = {}
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            return config
    except Exception as e:
        print(f"Error parsing {config_path}: {e}")
        return {}

def extract_class_info(java_file: Path) -> Dict:
    """Extract basic class information without full parsing"""
    with open(java_file, 'r') as f:
        content = f.read()
    
    # Simple regex-based extraction (avoids full Java parsing)
    package_match = re.search(r'package\s+([\w.]+);', content)
    class_match = re.search(r'class\s+(\w+)', content)
    interface_match = re.search(r'interface\s+(\w+)', content)
    
    return {
        'file_path': str(java_file),
        'package': package_match.group(1) if package_match else None,
        'class_name': class_match.group(1) if class_match else interface_match.group(1),
        'is_interface': bool(interface_match)
    }