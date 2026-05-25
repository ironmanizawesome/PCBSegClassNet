import os, json, sys

DANGEROUS_PATTERNS = [
    'rm -rf /', 'rm -rf ~', 'rm -rf *', 'rm -rf .',
    ':(){', 'mkfs', 'dd if=', '> /dev/sd',
    'chmod -R 777 /', 'curl|sh', 'wget|sh', 'curl|bash', 'wget|bash',
]
PROTECTED_DIRS = ['eagle/', 'data/', '.claude/']

try:
    data = json.loads(os.environ.get('TOOL_INPUT', '{}'))
    cmd = data.get('command', '')

    for pattern in DANGEROUS_PATTERNS:
        if pattern in cmd:
            print(f'BLOCKED: dangerous command detected: {pattern}')
            sys.exit(1)

    if 'rm -rf' in cmd and any(d in cmd for d in PROTECTED_DIRS):
        print('BLOCKED: rm -rf on protected directory')
        sys.exit(1)

except Exception:
    pass
