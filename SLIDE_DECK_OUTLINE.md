# Our Slide Plan — `TeamName_Presentation.pptx`
### Prediction of Oil Presence Using Machine Learning
**SPE DSEATS Africa 2026 Datathon** · 10 slides max (we lose points if we go over 10)

We'll build the deck ourselves in PowerPoint. Every figure we mention below is already saved in
the `figures/` folder. Let's keep each slide to about 6–8 lines of text and let the charts do the
talking.

---

## Slide 1 — Title  (we lose points if anything here is missing)
- Caption: "Prediction of Oil Presence Using Machine Learning"
- Our team name (one phrase, 15 characters or fewer)
- Each member: full name, **SPE number**, **SPE Section in Africa**, school/company
- The date
- (We need at least 2 different schools/companies represented across the team.)

## Slide 2 — Outline and our approach
- One line on the problem: predict oil (1) or no oil (0) from geology and seismic data.
- Our angle: treat it as a petroleum-system question — reservoir, trap, charge, seismic.
- Our roadmap: data check → exploration → Part 1 (standard ML) → Part 2 (with geology) → results.
- Use the petroleum-system table from notebook section 1.

## Slide 3 — The data and its quirks
- 3,000 training rows, 2,000 test rows, 7 features, about 28% of locations have oil.
- Three things we had to handle: lots of missing values in training (up to ~42%), the `Trap_Type`
  "None" vs blank mismatch, and some deliberate odd values.
- Figures: `figures/01_target_balance.png` and `figures/02_missingness.png`.

## Slide 4 — What controls oil  (EDA slide 1)
- Shale is about 10% oil (it's a seal, not a reservoir); real traps roughly double the oil rate.
- `Seismic_Score` is the strongest single signal — more seismic, more oil.
- Figures: `figures/04_categorical_controls.png` and `figures/07_seismic_dhi.png`.

## Slide 5 — The "all at once" rule and the hidden contradiction  (EDA slide 2)
- Oil needs good rock AND a trap AND strong seismic together — about 88% oil only when all three
  line up, and it collapses if any one is missing.
- Porosity and permeability don't agree the way real rock does — that's the contradiction we fix.
- Figures: `figures/09_and_gate.png` and `figures/08_poroperm_anomaly.png`.

## Slide 6 — How we built it  (this slide must cover BOTH parts)
- Part 1: line up trap labels → fill gaps with medians → encode → a few derived columns → compare
  6 models with cross-validation → average the best.
- Part 2: 5 geology fixes on `Trap_Type`, `Porosity`, `Permeability` → add petroleum-system
  features (RQI, FZI, charge score, Chance-of-Success) → re-run the same models.
- Figure: `figures/11_corrections.png` (before/after) plus the fixes table from notebook 5.1.

## Slide 7 — Results
- Headline: about **0.88 accuracy on the test-like rows**, ROC-AUC around 0.81.
- Part 1 and Part 2 give almost the same accuracy (they agree on 99% of test rows); Part 2 adds
  physically correct data and a model we can explain.
- Show the model-comparison table and the confusion matrix.
- Figures: `figures/13_part1_vs_part2.png` and `figures/10_part1_diagnostics.png`.

## Slide 8 — Why our analysis is strong
- We estimated the best score anyone could realistically get here (~0.88 on test-like rows) and
  showed our model already reaches it, so our number is honest.
- SHAP shows the model learned reservoir, trap and seismic by itself.
- Our physics-only Chance-of-Success score already sorts locations before any ML.
- Figures: `figures/14_shap.png` and `figures/12_chance_of_success.png`.

## Slide 9 — From prediction to a drilling plan
- We don't just hand over 0s and 1s; we rank every test location into risk tiers: drill the
  "very high" ones first, appraise "moderate", skip "very low".
- Figure: `figures/15_risk_map.png`.

## Slide 10 — Contributions  (required — name what each of us did)
- A small table: member → what they worked on (EDA / Part 1 / Part 2 geology / interpretability
  and slides).
- A one-line thank-you is fine; we don't need a separate "Thank You" slide (it just wastes one).

---

### Lines we want to say out loud (the judges reward these)
1. "We treated this as a petroleum-system chance-of-success problem, the way explorers risk a prospect."
2. "We found and fixed a physics contradiction — porosity and permeability that don't agree, and
   shale carrying permeability it can't physically have."
3. "We lined up the `Trap_Type` mismatch between the train and test files that quietly breaks naïve pipelines."
4. "We worked out the realistic accuracy ceiling, so our 0.88 is honest, not an overfit 0.95."
5. "Our model is explainable — SHAP confirms it learned real geology: reservoir, trap and seismic."
