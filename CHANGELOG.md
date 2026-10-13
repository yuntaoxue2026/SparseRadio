# Changelog

## v0.1.0 — 2026-10-13

- Preserved the complete previously viewed baseline and its source/config/artifact checksums.
- Fixed audit first-run directory creation; added a ZIP-backed, network-free regression.
- Added source/config/data/mask-bound version manifests and strict exports; disabled legacy unversioned fitting/selection.
- Added full-observation post-update diagnostics on shared 200/1,000/3,000-step trajectories.
- Executed 48 development and 432 validation trajectories; selected low-rank 3,000 steps and retained MLP 200 using balanced validation hidden RMSE. No extra regularization search was triggered.
- Re-fit all 864 conditions as a separate exploratory follow-up on the previously viewed holdout.
- Validated installation in a genuinely clean CPU environment, a ZIP-only fresh development run and a frozen-configuration fresh holdout example; recomputed full main metrics independently.
- Corrected the missed official CC BY 4.0 notice; separated data/code licenses and attributed adapted displays.
- Clarified that virtual environments are local and not supplied; direct version pins are not a complete lock.
- Separated the complete local experimental package from the minimal public candidate. The minimal candidate is published to the authorized GitHub repository; no GitHub Release, PyPI publication or website deployment.

- Publication review: aligned software version headings and publication wording; retained frozen experiment versions and numerical evidence unchanged.

- Converted the experiment report and its generation template to English; updated README and packaging links. Numerical results and frozen scientific configuration are unchanged.
