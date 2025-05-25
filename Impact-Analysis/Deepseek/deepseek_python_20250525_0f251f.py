import os
import subprocess
from pathlib import Path

def clone_repositories(repo_urls, clone_dir="repos"):
    """
    Clone multiple GitHub repositories to a local directory
    """
    os.makedirs(clone_dir, exist_ok=True)
    repo_paths = []
    
    for i, url in enumerate(repo_urls):
        repo_name = url.split('/')[-1].replace('.git', '')
        repo_path = Path(clone_dir) / repo_name
        
        if not repo_path.exists():
            result = subprocess.run(['git', 'clone', url, str(repo_path)], 
                                   capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone {url}: {result.stderr}")
        
        repo_paths.append(repo_path)
    
    return repo_paths