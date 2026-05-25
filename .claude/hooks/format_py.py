import os, json, subprocess, sys

try:
    data = json.loads(os.environ.get('TOOL_INPUT', '{}'))
    file_path = data.get('file_path', '')

    if file_path.endswith('.py'):
        subprocess.run(
            [sys.executable, '-m', 'black', '--quiet', '--line-length', '88', file_path],
            capture_output=True,
        )
except Exception:
    pass
