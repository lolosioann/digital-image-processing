import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(repo_root, "src")

# Ensure repo root and src are on sys.path so imports resolve consistently
for p in (repo_root, src_path):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Run tests from repository root so relative image/resource paths resolve
os.chdir(repo_root)
