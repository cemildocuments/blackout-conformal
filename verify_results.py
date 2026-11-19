"""Verify the deterministic numerical files against the recorded reference run."""
import pathlib,json,hashlib,sys
P=pathlib.Path(__file__).resolve().parent
checks=json.loads((P/'expected_checksums.json').read_text());bad=[]
for name,expected in checks.items():
 f=P/name
 actual=hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
 if actual!=expected:bad.append(name)
print(f'{len(checks)-len(bad)}/{len(checks)} numerical files match the reference run.')
if bad:
 print('Differing files: '+', '.join(bad));sys.exit(1)
