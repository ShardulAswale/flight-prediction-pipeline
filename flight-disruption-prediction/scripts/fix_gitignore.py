"""One-off script to fix .gitignore null byte corruption."""
import os

gitignore_path = os.path.join(os.path.dirname(__file__), '..', '.gitignore')

# Read raw bytes
with open(gitignore_path, 'rb') as f:
    data = f.read()

print(f"Before: {len(data)} bytes, {data.count(b'\\x00')} null bytes")

# Remove all null bytes
clean = data.replace(b'\x00', b'')

# Decode and clean up lines
lines = clean.decode('utf-8').splitlines()

# Remove empty trailing lines
while lines and lines[-1].strip() == '':
    lines.pop()

# Check if last line is a corrupted fragment (just 'env' or similar)
if lines and lines[-1].strip() in ('env', '.env', 'e n v', 'env\r'):
    lines.pop()

# Remove any carriage-return-only trailing line
while lines and lines[-1].strip() == '':
    lines.pop()

# Add project-specific entries if missing
content = '\n'.join(lines)
additions = ['credentials.json', 'models/', 'outputs/']
for entry in additions:
    if entry not in content:
        lines.append(entry)

result = '\n'.join(lines) + '\n'

with open(gitignore_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(result)

# Verify
with open(gitignore_path, 'rb') as f:
    verify = f.read()

print(f"After: {len(verify)} bytes, {verify.count(b'\\x00')} null bytes")
print("Last 5 lines:")
for line in result.splitlines()[-5:]:
    print(f"  [{line}]")
print("DONE")
