"""Export an explicit source allowlist and the approved cover demo, never raw takes."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

PUBLIC_FILES=(
    '.gitignore','LICENSE','NOTICE','README.md','pyproject.toml','Start Vocal Loom.cmd',
    'vocal_pipeline.py','pipeline_ui.py','pipeline_midi.py','pipeline_alignment.py',
    'pipeline_cleanup.py','pipeline_comp.py','pipeline_transcript.py','pipeline_dsp.py',
    'arrangement.py','duet_part.py','synthetic_fixture.py','prepare_public_repo.py',
    'test_pipeline_midi.py','test_pipeline_alignment.py','test_pipeline_comp.py',
    'test_pipeline_cleanup.py','test_duet_part.py','test_pipeline_dsp.py','test_pipeline_e2e.py',
    '.github/repository.json','.github/workflows/tests.yml',
    'scripts/render_note_video.py','scripts/set_repository_metadata.ps1','scripts/check_release.py',
    'docs/index.html','docs/.nojekyll','docs/PUBLISHING.md','docs/SEO-NOTES.md',
    'docs/demo/hide-and-seek-demo.mp3','docs/demo/hide-and-seek-notes.mp4',
    'docs/demo/poster.png','docs/demo/media.json',
    'examples/synthetic/synthetic-vocal.wav','examples/synthetic/synthetic-score.mid',
    'examples/synthetic/fictional-lyrics.txt','examples/synthetic/fixture.json',
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return {'description':'Allowlisted source, synthetic fixtures and the explicitly approved processed cover demo; raw takes and private project data excluded',
            'files':{name:file_hash(root/name) for name in sorted(PUBLIC_FILES)}}


def export(root,destination,archive):
    root=Path(root).resolve();destination=Path(destination).resolve();archive=Path(archive).resolve()
    for target in [destination,archive]:
        if target==root or not target.is_relative_to(root):
            raise ValueError('Export paths must stay inside this checkout')
        if target.exists():
            raise FileExistsError(f'Choose a fresh export path: {target}')
    if archive.is_relative_to(destination):
        raise ValueError('The ZIP must be outside the staged folder')
    for name in PUBLIC_FILES:
        p=root/name
        if not p.is_file() or p.is_symlink():
            raise ValueError(f'Missing or unsafe release file: {name}')
    destination.mkdir(parents=True)
    for name in PUBLIC_FILES:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    data=manifest(destination)
    (destination/'PUBLIC_MANIFEST.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(PUBLIC_FILES+('PUBLIC_MANIFEST.json',)):
            z.write(destination/name,'vocal-loom/'+name)
    return {'folder':str(destination),'archive':str(archive),
            'archive_sha256':file_hash(archive),'files':len(PUBLIC_FILES)+1,'published':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',default='dist/source')
    p.add_argument('--zip',default='dist/vocal-loom-source.zip')
    p.add_argument('--refresh-manifest',action='store_true',help='Update only the checkout release manifest')
    args=p.parse_args();root=Path(__file__).resolve().parent
    if args.refresh_manifest:
        (root/'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest(root),indent=2)+'\n',encoding='utf-8')
        print('Release manifest refreshed')
    else:
        print(json.dumps(export(root,root/args.destination,root/args.zip),indent=2))


if __name__=='__main__':
    main()
