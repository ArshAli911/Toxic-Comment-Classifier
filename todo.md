# Project Issues Checklist

- [x] Fix typo in `requirements.txt` (tdqm -> tqdm)
- [x] ~~Investigate and potentially remove redundant/incomplete training scripts (`train.py`, `train_bert.py`)~~ -> Keep `train_bert.py`, remove `train.py`.
- [x] ~~Investigate and potentially remove redundant/incomplete main scripts (`main.py`, `app.py`)~~ -> Keep `app.py`, remove `main.py`.
- [x] Address empty model files (`model_bert.pth`, `toxic_model.pth`). Need to retrain (`train_bert.py`) to generate `model_bert.pth`. Remove `toxic_model.pth`.
- [x] Address empty `run.sh` script. Determine its purpose or remove it. -> Remove `run.sh`.
- [x] ~~Investigate small `cleaned_toxic_comments.csv` file. Determine if it's placeholder or needs replacement/augmentation.~~ -> Remove `cleaned_toxic_comments.csv` (using `data/toxic_comments.csv`).
- [x] Remove temporary file `tempCodeRunnerFile.python`.
- [x] Review code in scripts for completeness, errors, and adherence to best practices. (Focus on `train_bert.py` and `app.py`).
- [x] Review notebooks in `notebook/` directory for relevance and completeness.
- [x] Ensure project structure (`app/`, `data/`, `models/`, `src/`) is logical and consistently used. -> Standardize on using root for main scripts/models for this structure.
- [x] Remove conflicting model files (`models/logistic_model.pkl`, `models/vectorizer.pkl`).
- [x] Install dependencies from `requirements.txt`.
- [ ] Run `train_bert.py` to generate `model_bert.pth`. -> **Skipped due to sandbox resource limitations. Requires local execution.**
- [ ] Validate `app.py`. -> **Skipped due to inability to train model in sandbox.**
