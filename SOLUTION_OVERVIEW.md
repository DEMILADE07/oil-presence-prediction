# Our Solution — SPE DSEATS Africa 2026 Datathon
### "Prediction of Oil Presence Using Machine Learning"

This is our own notes file: what we built, why we think it's strong, the numbers we got, and
the exact steps we still need to do before we submit.

---

## 1. What's in this folder

| File | What it is | Do we submit it? |
|---|---|---|
| `TeamName_PythonCode.ipynb` | Our full notebook (EDA + Part 1 + Part 2 + predictions), already run | Yes (after renaming) |
| `TeamName_Prediction_Part1.csv` | Our Part-1 predictions on the test set | Yes (after renaming) |
| `TeamName_Prediction_Part2.csv` | Our Part-2 predictions on the test set | Yes (after renaming) |
| `TeamName_Presentation.pptx` | Our slides — we still need to build this from the outline | Yes (after renaming) |
| `PROJECT_WALKTHROUGH.md` | Plain-English explanation of the whole project | our reference |
| `SLIDE_DECK_OUTLINE.md` | Our 10-slide plan plus which figure goes on each slide | our reference |
| `figures/` (15 images) | Every chart from the notebook, ready to drop into slides | our reference |
| `requirements.txt` | The exact library versions we used | our reference |
| `build_notebook.py` | The script we used to assemble the notebook | no |

When we email it in, we attach exactly four things: the notebook, the two CSVs, and the slides.

---

## 2. The big idea (why we think this wins)

A lot of teams will just feed the raw columns into a model and report an accuracy. We did
something different: we treated this like a real exploration problem.

Finding oil needs several things to be true in the same place at the same time — a rock that can
hold oil, a trap to keep it there, a nearby source that filled it, and seismic evidence that
fluids are present. Geologists call that the *petroleum system*, and it maps almost perfectly onto
the columns we were given. So our whole solution is: discovery = reservoir AND trap AND charge AND
seismic, all at once.

We then proved from the data that the labels really do behave this way, and we built that logic
into both our features and our Part-2 fixes.

### The five things we did that most teams probably won't
1. We built an explicit **Chance-of-Success** score that multiplies the petroleum-system
   ingredients together — the same way explorers actually rank a prospect. On its own, with no
   machine learning, it already sorts locations from least to most likely to have oil.
2. We spotted and fixed a **physics contradiction**: porosity and permeability don't move together
   the way real rock does, and shale carries permeability it physically can't have. We corrected it.
3. We caught the **`Trap_Type` mismatch** between the train file (the word "None") and the test
   file (blank). That quietly breaks naïve pipelines; we line them up.
4. We worked out the **best score anyone could realistically get** on this data (it's about 0.88
   on the test-like rows) and showed our model already reaches it. So our number is honest, not an
   overfit 0.95.
5. We **opened the model up with SHAP** and showed it independently learned reservoir, trap and
   seismic — real geology, not random correlations.

---

## 3. Our numbers (5-fold cross-validation)

| Metric | Part 1 (standard) | Part 2 (physics-informed) |
|---|---|---|
| Accuracy (all rows) | 0.805 | 0.804 |
| **Accuracy (rows with seismic ≈ the test set)** | **0.886** | **0.885** |
| F1 (oil class) | 0.62 | 0.62 |
| ROC-AUC | 0.814 | 0.813 |

How to read this (and we should say it in the interview): about a third of the *training* rows are
missing `Seismic_Score`, which is the single most useful column, and that drags the all-rows
accuracy down to ~0.80. But **every test row has a seismic score**, so our realistic test accuracy
is around **0.88** — and that's already at the ceiling the data allows. Part 1 and Part 2 land in
nearly the same place on purpose: once we're at the ceiling, Part 2's value is cleaner, physically
correct data and a model we can actually explain.

---

## 4. How Part 1 and Part 2 actually differ

**Part 1 (pure data science):** line up the trap labels, fill gaps with simple medians, one-hot
encode the categories, add a few obvious derived columns, compare six algorithms (Logistic
Regression, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM) with the same
cross-validation, then average the best ones.

**Part 2 (physics-informed)** does five geological fixes first (the brief asks for at least three):

| # | Column(s) | What's wrong | Our fix |
|---|---|---|---|
| 1 | `Trap_Type` | "None" (train) vs blank (test) are the same thing | merge into one `NoTrap` label |
| 2 | `Permeability` | shale can't carry reservoir-grade permeability | cap shale permeability |
| 3 | `Porosity` | shale shows reservoir-grade porosity; near-zero can't store oil | cap shale porosity, lift the floor |
| 4 | `Permeability` (gaps) | a blind median ignores the porosity link | estimate from porosity by rock type |
| 5 | `Porosity`, `Depth` (gaps) | one median ignores rock type | fill by each rock type's typical value |

…then we add petroleum-system features: Reservoir Quality Index (RQI), Flow Zone Indicator (FZI),
element scores, the Chance-of-Success score, and the "all at once" interactions, and re-run the
same six models.

---

## 5. The one thing we have to be careful about

The guidelines (section 4.0) say AI-assisted and generative-AI work is "highly discouraged and
will result in disqualification," and that they screen every submission with an AI-detection tool.
So before we submit we need to genuinely own this work: read it line by line, put the explanations
in our own words, and make sure each of us can explain every step, because the top teams get
interviewed. Treat the notebook as a reference we reproduce and understand, not something to copy
blindly.

---

## 6. Our checklist before we send it

- [ ] Pick our team name (single phrase, **15 characters or fewer**). Replace **`TeamName`** in:
      the two CSV filenames, the slides filename, the notebook filename, and the
      `TEAM_NAME = "TeamName"` line inside the notebook (then re-run that cell so the CSVs get the
      right name).
- [ ] Build `TeamName_Presentation.pptx` from `SLIDE_DECK_OUTLINE.md` — **10 slides max**.
- [ ] Fill in the **Title slide**: each of our names, SPE numbers, SPE Section, school/company,
      and the date. (We need at least 2 different schools/companies across the team, and no two
      past winners on one team.)
- [ ] Fill in the **Contributions** table (slide 10 and notebook section 8) with what each of us did.
- [ ] Re-run the whole notebook in Colab or Jupyter to make sure it runs cleanly start to finish.
- [ ] Reword the narrative in our own voices (see section 5 above).
- [ ] **One person** emails all four files and CCs the rest of us.
- [ ] Subject line: `TeamName_SPE DSEATS Africa 2026 Datathon Submission`
- [ ] Send to **speafricadseat@gmail.com** before **Monday 29 June 2026, 11:59 PM WAT**.
- [ ] Remember: **2 submissions maximum, ever**. Mark which one is final, and don't let multiple
      people submit separately.

---

## 7. How to run it

On our own machines:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook TeamName_PythonCode.ipynb     # then Run All
```

In Google Colab: upload `oil_presence_trainset.csv` and `oil_presence_testset.csv` to the session
first, then Runtime → Run all. The first code cell installs anything that's missing.
