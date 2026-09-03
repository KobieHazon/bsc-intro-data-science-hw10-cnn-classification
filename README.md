# BSc Intro Data Science - HW10 CNN Classification

A historical archive of my CS BSc coursework.

## Contents

Neural-network coursework covering dense and convolutional models for image classification, including evaluation and bonus model exploration.

## Provenance

- Era: CS BSc.
- Last recovered work: 2019 archive copy.
- Supplied exercise material is identified separately below.

## Files

Template or reference material:

- `assignment/HW_10 - SOL.ipynb`

My solution notebooks:

- `solutions/HW_10.ipynb`

My submitted answers:

- `results/hw10_answers.csv`

## Tech Stack

- Python notebooks.
- Main Python packages: keras, matplotlib, numpy, pandas, requests, scikit-image, scikit-learn, tensorflow, notebook.
- Jupyter-compatible local review flow.

## Dataset Notes

The original course folders for several data-science assignments contained the large `ebay_boys_girls_shirts` image dataset and tarball. Those files are not tracked in this repository. The recovered notebooks reference the course download URL and recreate the dataset folder when that URL is still available.

## Notes

- The notebook uses older TensorFlow/Keras APIs and may require a legacy Python/TensorFlow environment for full reruns.

## Validate

```bash
python3 scripts/check_notebooks.py
```

This check verifies that notebooks parse as JSON and that the removed student identifier does not remain in tracked text files.
