"""Check public metadata, demo references and source-only release hygiene."""
from pathlib import Path
import json
import re
import tomllib


def main():
    root=Path(__file__).resolve().parents[1]
    meta=json.loads((root/'.github/repository.json').read_text(encoding='utf-8'))
    project=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']
    assert meta['name']==project['name']=='vocal-loom'
    assert meta['description']==project['description']
    assert len(meta['description'])<=350
    topics=meta['topics']
    assert len(topics)==len(set(topics))<=20
    assert all(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,49}',t) for t in topics)
    readme=(root/'README.md').read_text(encoding='utf-8')
    assert 'Imogen Heap' in readme and 'Hide and Seek' in readme
    assert '<audio' not in readme and '<video' not in readme
    for target in re.findall(r'\]\(([^)]+)\)',readme):
        if '://' in target or target.startswith(('#','mailto:')):
            continue
        assert (root/target.split('#')[0]).exists(),target
    for folder in [root,root/'docs',root/'.github']:
        files=folder.glob('*.md') if folder==root else folder.rglob('*')
        for path in files:
            if path.is_file() and path.suffix in ('.md','.json','.toml','.html','.ps1','.yml'):
                assert '\u2014' not in path.read_text(encoding='utf-8'),str(path)
    media=root/'docs/demo/media.json'
    if media.exists():
        data=json.loads(media.read_text())
        assert data['voices']==4
        assert data['mp4_bytes']<10_000_000
    for name in ['hide-and-seek-demo.mp3','hide-and-seek-notes.mp4','poster.png']:
        assert (root/'docs/demo'/name).is_file(),name
    assert (root/'docs/index.html').is_file()
    assert 'Co-authored-by:' not in (root/'docs/PUBLISHING.md').read_text()
    print(f'Release checks passed: {len(topics)} topics, valid demo files, no em dashes.')


if __name__=='__main__':
    main()
