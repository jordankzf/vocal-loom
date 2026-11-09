# Publishing VocalLoom

The repository is prepared locally. These instructions do not create, push or
publish anything automatically.

## 1. Review the exact contents

```powershell
git status --short
git diff --cached --stat
git diff --cached
```

Include the application, tests, documentation, synthetic fixtures and the approved
processed cover MP3/video. Keep raw recordings, song MIDI, full lyrics, private
assessments, transcripts, environments and model weights out of the index.

The public demo is the approved final first-take master. It retains the preferred
tuning, harmonies, EQ, compression and reverb, with cleanup applied to discarded
attempts and selected mouth/breath noise. Credit the song and performance as shown
in the README and NOTICE.

## 2. Create the target repository and push the prepared commit

When publication is authorized, create `jordankzf/vocal-loom` on GitHub without an
initial README or licence so the prepared files remain authoritative. Use the
configured Git identity. The prepared local commit has no co-author trailer.

```powershell
git remote add origin https://github.com/jordankzf/vocal-loom.git
git log -1 --format=fuller
git push -u origin HEAD
```

If `origin` already exists, inspect it rather than replacing it blindly. The
current local branch can be the repository's first default branch; use its actual
name in later branch-specific settings.

## 3. Apply the description and topics

The prepared values are in `.github/repository.json`. GitHub does not read them
automatically from that file. With an authenticated GitHub CLI:

```powershell
$repo = 'jordankzf/vocal-loom'
$metadata = Get-Content -LiteralPath .github/repository.json -Raw | ConvertFrom-Json
gh repo edit $repo --description $metadata.description
foreach ($topic in $metadata.topics) {
    gh repo edit $repo --add-topic $topic
}
```

These values prioritize actual function and platform. See [SEO-NOTES.md](SEO-NOTES.md)
for the independent review and its sources.

## 4. Add the native playable README attachment

The bundled video is `docs/demo/hide-and-seek-notes.mp4`. A repo-relative MP3 or MP4
link alone does not produce GitHub's inline README player.

1. Open `README.md` in GitHub's web editor.
2. Drag the prepared MP4 into the demo section, replacing the poster between
   the `demo-player` comment markers.
3. Let GitHub finish uploading it and insert its
   `https://github.com/user-attachments/assets/...` URL.
4. Keep the inserted attachment URL on its own line. It is the real asset URL,
   not a made-up path or a placeholder.
5. Preview the README and play the beginning, a middle section and the corrected
   final phrase. Confirm audio, synchronization and credits before committing.

The upload is a separate external step from pushing source. GitHub documents
MP4/MOV/WebM support and recommends H.264 for compatibility. The video is prepared
to fit the free-plan upload limit. See [GitHub's attachment documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)
and [attachment URL behaviour](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-anonymized-urls).

## 5. Optional GitHub Pages player

The bundled `docs/index.html` provides a normal browser player. It can be opened
locally without publishing. To host it, enable GitHub Pages for the repository
with `docs/` as the publishing folder on the actual default branch, or use the
prepared Pages deployment workflow if one is included.

After deployment succeeds, verify the live player before adding a homepage link
to the repository's About settings. The intended project-site address is
`https://jordankzf.github.io/vocal-loom/`; that address is not asserted to be live
before Pages has been enabled and deployed.

GitHub Pages is optional. The native attachment player can handle the README
demo independently.
