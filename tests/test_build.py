from pathlib import Path
import ast
import hashlib
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def test_submission_build_is_deterministic_and_embeds_core():
    builder=ROOT/'tools/build_submission.py'
    assert builder.exists(), 'Standalone builder has not been implemented'
    subprocess.run([sys.executable,str(builder)],cwd=ROOT,check=True)
    target=ROOT/'submission/portfolio_round1.py'
    before=target.read_bytes()
    subprocess.run([sys.executable,str(builder),'--check'],cwd=ROOT,check=True)
    assert target.read_bytes()==before
    text=before.decode()
    core=(ROOT/'src/portfolio_game/core.py').read_text()
    assert core in text
    tree=ast.parse(text)
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    cls=next(n for n in classes if n.name=='CVXPYPortfolio')
    assert not any(isinstance(n,ast.FunctionDef) and n.name=='__init__' for n in cls.body)
    assert any(isinstance(n,ast.FunctionDef) and n.name=='fit' for n in cls.body)
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert 'skfolio.optimization' in imports
    assert all(not (name or '').startswith(('portfolio_game','research')) for name in imports)
    assert all(not isinstance(n,ast.If) or not isinstance(n.test,ast.Compare) or '__name__' not in ast.unparse(n.test) for n in tree.body)

def test_teacher_sources_are_unchanged():
    import json
    p=ROOT/'teacher_reference/SHA256.json'
    assert p.exists(), 'Instructor source checksums missing'
    for name,expected in json.loads(p.read_text()).items():
        assert hashlib.sha256((ROOT/'teacher_reference'/name).read_bytes()).hexdigest()==expected
