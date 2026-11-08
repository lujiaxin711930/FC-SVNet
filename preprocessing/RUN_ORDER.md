# Image-generation pipeline

The original execution order was: **compute minima and maxima → plot well-log curves → crop images → resize images → merge images**.

Run the scripts from the repository root:

```bash
python preprocessing/compute_curve_ranges.py
python preprocessing/plot_well_log_images.py
python preprocessing/crop_images.py
python preprocessing/resize_images.py
python preprocessing/merge_images.py
```

The original scripts used local Windows paths; this release uses project-relative directories by default. Override them using `FC_SVNET_EXCEL_DIR`, `FC_SVNET_PLOT_DIR`, `FC_SVNET_CROP_DIR`, `FC_SVNET_RESIZE_DIR`, or `FC_SVNET_MERGE_DIR` environment variables.

The plot script expects depth, six log curves, and a lithofacies column named `Lithofacies` with this original positional layout: depth in column 0, logs ( `GR`, `AC`, `LLD`, `LLS`, `SP`, and `CAL`) in columns 1–6, and labels in column 7. These column names remain unchanged because they are input-data keys, not code comments. The input order must be verified before use.
