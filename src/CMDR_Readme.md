`CMDR.py` runs a cross-lingual consistency check for the model.

It loads aligned sentence pairs from XNLI (currently French and Vietnamese), asks the same NLI question in both languages, and compares the model outputs.

It measures:
- **Label Disagreement**: how often the predicted label changes across languages.
- **Confidence Disagreement**: how far apart the model confidence scores are across languages.

In short, it checks whether the model is stable across languages and reports mean/variance statistics to quantify that stability.
