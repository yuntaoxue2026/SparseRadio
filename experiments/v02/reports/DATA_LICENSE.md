# Data and figure license evidence

The original RadioMapSeer download is labelled **CC BY 4.0 on its official website**. The HTML places the `rel="license"` link immediately after the original Google Drive download and before the separate RadioUNet code link. This evidence was present in the earlier local snapshot and was missed in the first report; that report's uncertainty statement is superseded here.

Primary evidence:

- [Official dataset page](https://radiomapseer.github.io/index.html).
- [Official page source, line 36](https://github.com/radiomapseer/radiomapseer.github.io/blob/master/index.html#L36).
- [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
- [Author's usage and citation instructions](https://github.com/RonLevie/RadioUNet).

The relevant source contains `rel="license" href="http://creativecommons.org/licenses/by/4.0/"`. Local copies and their SHA-256/access timestamps are in `experiments/v02/reports/license_evidence/`. The original ZIP has no README/LICENSE/COPYING file. The data license evidence comes from the dataset publisher's page, **not** the paper's arXiv license or the author's code MIT license. No claim is made about later IEEE DataPort releases; this project uses only the original no-car DPM archive.

Attribution: **Çağkan Yapar, Ron Levie, Gitta Kutyniok, Giuseppe Caire**, RadioMapSeer; *Dataset of Pathloss and ToA Radio Maps With Localization Application*, arXiv:2212.11777 (2022, version 4, 2024). Dataset source: <https://radiomapseer.github.io/index.html>. Original dataset and adapted spatial displays: CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>.

Changes in displayed figures: outdoor masking, seeded observation holes/sampling, normalized-to-dB color mapping, model reconstructions, error maps, panel layout and labels. Predictions are SparseRadioMap outputs, not the dataset authors' original simulations. The dataset authors do not endorse these comparisons. Attribution and change descriptions accompany the figures in FIGURE_ATTRIBUTION.md and the spatial PNG itself.

Packaging categories:

| Category | Complete local experimental package | Minimal public candidate |
| --- | --- | --- |
| Original 3.26 GB ZIP | External read-only reference and checksum; already present locally | Excluded; download/reuse command provided |
| ZIP-derived processed arrays | Included | Excluded |
| Observational masks and raw predictions | Included for every version | Excluded |
| Per-run numerical results and summaries | Included with full history | Necessary CSV summaries and candidate evidence only |
| Dataset-derived display images | Included | Selected version-labelled, attributed displays under CC BY 4.0 |
| Detailed baseline snapshot and logs | Included | Excluded; compact baseline summaries retained |
| Virtual environments / caches / credentials | Excluded | Excluded |

Original project code is MIT © 2026 Yuntao Xue. This separate code license does not relicense the data. RadioUNet code is MIT © 2019 Ron Levie; its audit-only source is not part of our algorithm implementation. This repository publishes the minimal candidate; the original archive and derived arrays are excluded. The authors have not been contacted.

Underlying city layouts originate from OpenStreetMap, as described in the dataset paper. Map-layout credit: **© OpenStreetMap contributors**, with upstream database licensing information at https://www.openstreetmap.org/copyright (ODbL). Display figures are adapted raster presentations, not a redistribution of an OSM database; the dataset publisher's CC BY 4.0 notice and upstream map attribution are both retained.
