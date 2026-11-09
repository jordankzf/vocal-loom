# Repository discovery and SEO decisions

Reviewed 9 November 2026. These are launch decisions based on current primary
documentation, not a promise of search rankings or a claim of measured keyword
volume.

## Independent review versus the older website skill

A fresh agent reviewed current GitHub and Google documentation without seeing the
local SEO skill, the repository's existing copy or the proposed SEO plan. Its
most important finding changed the priorities: ordinary GitHub repository search
uses the repository name, description and topics. README text is included when a
search explicitly uses `in:readme`. That makes accurate metadata the first launch
task, rather than adding a large keyword block to the README. See [GitHub's
repository-search documentation](https://docs.github.com/en/search-github/searching-on-github/searching-for-repositories).

| Advice from a general website workflow | Decision for this GitHub repository |
|---|---|
| Clear headings, useful explanations and readable prose | Keep. The README should explain inputs, outputs, installation and limits. |
| Relevant search terms | Use naturally in the description, topics, title and useful documentation. |
| HTML title, description and Open Graph tags | Do not add them to README Markdown. We do not control GitHub's page head. |
| JSON-LD and FAQ schema snippets | Do not add them to the README. Rendered Markdown is sanitized, including script tags. |
| Meta keywords and repeated keyword lists | Do not use. Google ignores meta keywords and warns against keyword stuffing. |
| Numerical promises about visibility increases | Do not reuse them as GitHub ranking claims. They were not validated for this repository or task. |
| A separate static project website | Optional later. It can offer a normal HTML player and its own page metadata. |

GitHub describes the README's job as explaining the project, usefulness and how
to get started. Its [README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
and [rendering pipeline](https://github.com/github/markup/blob/master/README.md)
support the documentation-first approach. [Google's SEO Starter Guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide)
supports helpful content and descriptive wording, not hidden keyword stuffing.

## Metadata selected

- Product name: **VocalLoom**.
- Repository slug: **vocal-loom**.
- Functional description: local vocal harmonizer, pitch correction, MIDI,
  solos, duets and four-voice WAV stacks.
- Fourteen topics describe the actual audio task, language, platform and engines.
  They are proposed classifications, not a high-volume keyword list.
- Imogen Heap and "Hide and Seek" appear in the description, inspiration and demo credits.
  Artist and song names are not used to fill the topic list.

The exact description and topic array are in
[`.github/repository.json`](../.github/repository.json). This file is preparation
data, not a GitHub settings mechanism. Repository metadata must be applied
through GitHub's interface or API when publishing. GitHub allows up to 20 topics,
using lowercase letters, numbers and hyphens, with at most 50 characters each.
See [GitHub's topic documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics).

## Demo and exact feature claims

The approved processed cover is useful evidence of the effect. No documented
ranking bonus is claimed for including it. The README labels the four voices as
one lead plus three harmonies, and explains that the arranger constructs parts
from score chord tones. It does not describe the application as a VST plugin,
real-time performer, voice-cloning service or text-to-singing system.

MP3, MP4 and the local player are prepared as actual media, rather than adding an
unsupported `<audio>` tag to the README. GitHub's inline video player needs a
GitHub-hosted attachment URL. GitHub recommends H.264 for video compatibility.
See the [attachment documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)
and [publishing checklist](PUBLISHING.md).

## What remains uncertain

GitHub does not disclose all relevance-ranking weights. No search-demand study,
indexing timeline, trademark clearance or guaranteed position is claimed here.
The useful launch checks are factual copy, working setup, a clear demo, suitable
topics, a detectable software licence and links from relevant project contexts.
