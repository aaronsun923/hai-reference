# hai-reference

Human increment over image classifiers, against an ensemble reference and with the model shown. Locked-spec reanalysis of ImageNet-16H (Steyvers et al.) and Tejeda et al. (2022).

- `specs/`: the locked specification and dated amendments
- `pipeline/`: analysis code, numbered in run order
- `reports/`: pilot and full-run reports
- `PERMISSIONS.md`: data permission (reanalysis only, no redistribution)

Data are not included. Download the OSF files from projects `2ntrf` (ImageNet-16H) and `9xbpu` (Tejeda et al. 2022) and place them, with their original file names, under `~/Desktop/hai-reference-raw/2ntrf/` and `~/Desktop/hai-reference-raw/9xbpu/`, together with a `MANIFEST.tsv` (columns: project, osf_path, osf_sha256) as `pipeline/00_deidentify.py` expects. Step 00 writes de-identified files to `data/` (gitignored); later steps read from there and from the raw directory.

Code is MIT licensed; the data keep their owners' terms.

Archived: https://doi.org/10.5281/zenodo.22822332
