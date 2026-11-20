# Macros with no data source

`params.tex` defines every number the manuscript body prints, and
`test_assertions.py` binds each one to the committed artifact that measures it.
The macros below cannot be bound, because nothing this repository ships derives
them. They are listed here so that "no source" is a recorded decision with a
reason attached rather than the default state of a number nobody checked, and so
that the enumeration in `test_params_every_macro_is_bound_or_exempt` fails on any
macro that is neither bound nor listed here.

An entry is a liability, not a licence. Adding one is the last resort, after
asking whether the value can be measured from an artifact the repository can
carry.

Two kinds of entry, and only the first is an exception to anything this paper
claims about its own numbers. A figure measured by **the CDD study** is that
study's, cited to its source; no artifact of ours could derive it, and the
data-availability statement does not claim otherwise. A figure measured by
**this paper** with no shipped table is the real exception, and the
availability statement names every one of them. That count is asserted in
`test_availability_statement_matches_the_exemptions`, so adding a row of that
kind fails until the manuscript sentence is rewritten.

| Macro | Measured by | Reason it has no committed source |
|---|---|---|
| `\paperIncrement` | the CDD study | The study's reported figure, transcribed from *Applied Sciences* 14(10):4322 Table 2. Their measurement, not ours; no artifact of ours can derive it. Cross-checked against `PAPER_INCREMENT`. |
| `\paperImageFone` | the CDD study | The study's reported image-only macro-F1, same table and the baseline the reported increment rests on. Cross-checked against `PAPER_IMAGE_F1`. |
| `\paperMultiFone` | the CDD study | The study's reported image-plus-environment macro-F1, same table and the figure the reported increment reaches. It is the figure that the criterion in `dacon_their_split.py` compares with. Cross-checked against `PAPER_MULTIMODAL_F1`. |
| `\leeYunAuroc` | Lee and Yun (2023) | The average area under the receiver operating characteristic curve that Lee and Yun (2023, *Plant Methods* 19:145, Results) report for their environment-only model. Their measurement, not ours; no artifact of ours can derive it. |
