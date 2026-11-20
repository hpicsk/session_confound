# Sources and retrieval routes

Where the third-party material this repository reads comes from, and how to get
it. Nothing here is redistributed: the manuscript's data-availability statement
(`sec:availability`) says we ship derivations rather than other people's
material, the AI Hub terms separately restrict redistribution of what the
provider supplies, and a copy of either sitting in this repository would
contradict a section the paper stands on. What a reproducer needs is the route
to the file, not the file.

## Consulted material

| source | identifier | route |
|---|---|---|
| AI Hub dataset documentation (data manual, construction guide) | 국내 재배 아열대·열대 병해충 데이터 (*Domestically Cultivated Subtropical and Tropical Pest and Disease Data*; the English title is our translation: AI Hub lists the Korean title, and the dataset's guide gives two different English names) | Served from the dataset's own page at `https://aihub.or.kr` alongside the dataset, behind the same access gate: account registration, an application, and provider approval. At retrieval on 2026-08-15 the dataset was keyed `dataSetSn=71750`. That key is a platform-side locator rather than part of the dataset's identity, so **retrieve by title**. Cited as `aihub2023subtropical`. |
| Berrocal et al., *Multimodal deep learning model for real-time severity assessment of maize stalk rot integrating visual and soil sensing data*, Smart Agricultural Technology 14 (2026) 102038 | `doi:10.1016/j.atech.2026.102038` | Gold open access, CC-BY. The publisher serves the PDF to a browser session; a plain HTTP client is redirected to a consent page, so fetch it from a browser at `https://doi.org/10.1016/j.atech.2026.102038`. Cited as `berrocal2026`. |

Every other cited study, `cornsentry2026` and `jamun2025iot` among them, is
retrieved through the DOI that `paper/refs.bib` gives it, and is coded from its
full text in `code/survey/survey_records.csv`.

## Dataset naming, and what keeps its original identifier

The manuscript and this repository identify both datasets by their official
titles rather than by the platform that serves them. DACON is a competition
host, the way Kaggle is; the dataset it served has a name of its own, and the
CDD study gives it in full (*Applied Sciences* 14(10):4322, Section 3.1).
The mapping:

| platform key | named in this repository as |
|---|---|
| AI Hub `71750` | the AI Hub dataset (국내 재배 아열대·열대 병해충 데이터) |
| DACON `235870` | the CDD dataset (*Crop Disease Diagnosis based on Agricultural Environmental Changes*, 농업 환경 변화에 따른 작물 병해 진단) |

**The directory and artifact filenames under `code/dacon_cropdisease/` keep the
original identifier.** The rename is a notational change in the manuscript and in
reader-facing documentation, and this table is the join.

## Why no dataset fact depends on the documentation

The AI Hub dataset documentation was read once, to fact-check the dataset
description in `sec:corpora` against its primary source. Every number that survived
that pass is measured against the shipped dataset rather than read from the
documentation, which is what `sec:corpora` states. No committed result, macro, or
assertion reads it.
