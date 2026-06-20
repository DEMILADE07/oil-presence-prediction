# Project Walkthrough — Predicting Oil From Rocks
### A plain-English guide to our SPE DSEATS Africa 2026 Datathon project

This guide explains our whole project from the very beginning, assuming you've never seen machine
learning or geology before. By the end you'll understand what we were asked to do, what the data
means, every step in our notebook, and why we made the choices we made.

Read it top to bottom. Each section builds on the one before it.

---

## Part A — The big picture

### A1. What is this competition asking us to do?

There's a table of 5,000 locations around the world. For each location we're told a few facts
about the rock and the seismic survey (things like how spongy the rock is, how deep it is, what
the seismic scan saw). For 3,000 of those locations we're also told the answer: did they find oil
there, yes or no. For the other 2,000 locations the answer is hidden, and our job is to predict it.

In machine-learning words, this is a **binary classification problem**:
- *Binary* = there are only two possible answers (oil = 1, no oil = 0).
- *Classification* = we sort each location into one of those two buckets.

The competition has two parts:
- **Part 1** — solve it as a normal data-science problem.
- **Part 2** — solve it again, but this time use oil-and-gas knowledge to clean up the data first.
  The judges care most about Part 2.

### A2. A 2-minute geology crash course

Oil doesn't sit in underground lakes. It's squeezed inside the tiny holes of certain rocks, like
water in a sponge. For a location to actually hold oil, several separate things all have to be true
at the same time. Geologists call this the **petroleum system**, and it's the heart of our whole
project:

1. **A reservoir rock** — a rock with lots of little holes (so oil can be stored) that are
   connected (so oil can flow). Sandstone and limestone are good at this. Shale is not — shale is
   dense and tight.
2. **A trap** — some shape in the rock that stops the oil from floating away. Oil is lighter than
   water, so it constantly tries to rise; a trap is what holds it in place.
3. **A charge** — a nearby "source rock" that actually cooked up the oil and pushed it into the
   reservoir. No source nearby, no oil, no matter how nice the reservoir is.
4. **A seismic indicator** — when geophysicists bounce sound waves into the ground, fluids like oil
   can create a tell-tale "bright spot." A strong seismic signal is a hint that oil is down there.

The key word is **AND**. You need a reservoir **and** a trap **and** a charge. Miss any one and you
drill an expensive dry hole. This single idea is what makes our solution different from just
throwing numbers at a computer — we'll come back to it again and again.

### A3. What each column in the data means

| Column | Plain meaning | Good sign for oil? |
|---|---|---|
| `Rock_Type` | The kind of rock: Sandstone, Limestone, or Shale | Sandstone/Limestone good, Shale bad |
| `Porosity` | How much empty space is in the rock (%), i.e. how much oil it could hold | Higher is better |
| `Permeability` | How easily fluid can flow through the rock (mD) | Higher is better |
| `Trap_Type` | The trap shape: Anticline, Dome, Fault, or None | Any real trap beats None |
| `Seismic_Score` | The seismic "bright spot" score, from 0 to 1 | Higher is better |
| `Proximity_to_Oil_Field` | Distance to the nearest known oil field (km) | Closer is better |
| `Estimated_Reservoir_Depth` | How deep the rock layer is (m) | Mixed |
| `Oil_Presence` | **The answer:** 1 = oil, 0 = no oil | This is what we predict |

---

## Part B — Tools and setup (notebook section 2)

### B1. What is a Jupyter notebook?

Our whole solution lives in a file called `TeamName_PythonCode.ipynb`. A notebook is just a
document made of **cells**. Some cells contain text (explanations like this), and some contain
Python code you can run. When you run a code cell, its output (a number, a table, a chart) appears
right underneath it. You run them top to bottom, like following a recipe.

### B2. The libraries we use (and what each one is for)

Python on its own can't do much data science, so we borrow toolkits called *libraries*:
- **pandas** — works with tables of data (think Excel, but in code). A table is called a
  *DataFrame*.
- **numpy** — fast maths on lots of numbers at once.
- **matplotlib** and **seaborn** — draw charts.
- **scikit-learn** — the main machine-learning toolkit (models, cross-validation, metrics).
- **xgboost** and **lightgbm** — two extra, very strong model types.
- **shap** — explains *why* a model made a prediction.

The first code cell tries to load each one and quietly installs anything that's missing, so the
notebook works on a fresh computer or in Google Colab without fuss.

### B3. Loading the data and checking its health

We read the two CSV files into two tables, `train_raw` (the 3,000 rows with answers) and
`test_raw` (the 2,000 rows we predict). Then we do a quick "health check" and find four important
things:

1. **Lots of gaps in the training data** (some columns are ~40% empty), but the test data has no
   gaps in its number columns. So filling gaps matters for *learning*, but never affects the
   predictions we're graded on.
2. **The `Trap_Type` column is written two different ways.** In the training file, "no trap" is the
   actual word `"None"`. In the test file it's just left blank. These mean the same thing, so we
   merge them into one label, `NoTrap`. If we forgot this, the model would get confused on the test
   set.
3. **Only ~28% of locations have oil.** This is called *class imbalance*. It matters because a lazy
   model that always guesses "no oil" would already be right 72% of the time, so 72% is the bar to
   beat, and plain accuracy can be misleading.
4. **There are deliberately weird values** in the data that we'll hunt down later.

---

## Part C — Exploring the data (notebook section 3)

Before building anything, we look at the data and let it tell us its story. We call this
**EDA** (Exploratory Data Analysis). We asked five questions, each answered with a chart.

### C1. How often is there oil? (figure 01)
About 1 location in 4 has oil. This confirms the imbalance and sets our "must beat 72%" bar.

### C2. Do the gaps mean anything? (figure 02)
Sometimes the fact that a value is *missing* is itself a clue. We checked, and here it isn't: the
oil rate is the same whether a value is present or missing. So the gaps look random, which means
it's safe to fill them in with sensible estimates.

### C3. Are the numbers physically sensible? (figure 03)
Every measurement sits inside the range it's supposed to. So the "anomalies" the competition warned
us about aren't just numbers that are too big or too small. They're *contradictions between
columns* — which we find in question 5.

### C4. What actually decides whether there's oil? (figures 04, 05, 06, 07)
- **Rock type matters a lot:** shale drops to ~10% oil, while sandstone and limestone sit near 35%.
- **Having a trap matters:** any real trap roughly doubles the oil rate compared to no trap.
- **`Seismic_Score` is the single most useful column** by far. As the seismic score rises, the oil
  rate climbs smoothly from ~10% to ~65%. That smooth climb is exactly how a good oil indicator
  should behave.
- The other numbers (porosity, permeability, proximity, depth) barely matter *on their own* in this
  particular dataset.

> A note on the word *correlation*: it's just a number from -1 to +1 that says how strongly two
> things move together. +1 = they rise together perfectly, 0 = no relationship, -1 = one rises as
> the other falls. `Seismic_Score` has about +0.42 with oil (strong here); the others are near 0.

### C5. The hidden contradiction (figure 08)
In real rocks, porosity and permeability rise together — more holes usually means easier flow. In
our data they don't; the relationship is basically flat, and shale (which should barely let fluid
through) often shows high permeability. **That's physically impossible.** It's the planted
inconsistency, and fixing it is the centerpiece of Part 2.

### C6. Proof of the "all at once" rule (figure 09)
This is our favourite chart. When we split the data by (good rock?) × (has trap?) × (strong
seismic?), oil shows up about **88% of the time only when all three are true at once**. Take away
any single ingredient and it crashes back to ~10%. This proves the ingredients **multiply**
together rather than add up — and that insight shapes our features in Part 2.

---

## Part D — Part 1: a standard machine-learning workflow (notebook section 4)

This is the "do it by the book" half — no geology tricks yet.

### D1. Preparing the data
- Merge the `Trap_Type` labels (as explained above).
- **Fill the gaps** ("imputation"): for number columns we use the median (the middle value); for
  category columns we use the most common value.
- **Encode the categories** ("one-hot encoding"): models need numbers, not words. So a column like
  `Rock_Type` becomes three yes/no columns: is it Sandstone? is it Limestone? is it Shale?
- Add a few simple derived columns (for example, seismic score multiplied by "has a trap").

### D2. What is cross-validation, and why we trust it
If we tested a model on the same data it learned from, it could just memorise the answers and look
amazing while being useless on new data. To avoid fooling ourselves, we use **5-fold
cross-validation**: split the training data into 5 equal chunks, train on 4 of them, test on the
5th, and rotate so every chunk gets a turn as the test. The scores then reflect how the model does
on locations it has *never seen*. That's the honest way to measure.

### D3. The six models we compare
We don't guess which algorithm is best — we try six and let the numbers decide:
- **Logistic Regression** — a simple, fast baseline.
- **Random Forest** and **Extra Trees** — many decision trees voting together.
- **HistGradientBoosting**, **XGBoost**, **LightGBM** — "gradient boosting," where each new tree
  fixes the mistakes of the last. These are usually the strongest on this kind of data.

We also tell every model that oil is the rarer, more important class so it doesn't lazily ignore it.

### D4. Combining the best models (the "ensemble")
The tree models all scored close together, so instead of betting on one we **average their
predictions** — this is called a soft-voting *ensemble*. Averaging several good models is usually a
little more reliable than trusting any single one.

### D5. Reading the scorecard: every evaluation metric, explained

This is how we measure whether the model is any good. The easiest way to understand all of it is
to picture each prediction as a **drilling decision**, because that's exactly what it stands for.
Saying "oil" means *drill here* (expensive); saying "dry" means *don't bother*.

**The confusion matrix (the foundation everything is built on).** Every prediction lands in one of
four boxes, depending on what we said versus what was actually true:

|  | We predicted **dry** | We predicted **oil** |
|---|---|---|
| **Actually dry** | TN — correctly skipped (money saved) | **FP** — drilled a dry hole (wasted cost) |
| **Actually oil** | **FN** — walked away from real oil (missed fortune) | TP — drilled and struck oil |

- **TP (true positive):** we said oil, and there was oil. ✅
- **TN (true negative):** we said dry, and it was dry. ✅
- **FP (false positive):** we said oil, but it was dry — a *false alarm*. ❌
- **FN (false negative):** we said dry, but it was oil — a *miss*. ❌

Our actual numbers on the test-like rows (the 1,950 training rows that have a seismic score, our
closest mirror of the real test set): **TN = 1,362, FP = 72, FN = 152, TP = 364.**

**The five headline metrics**, each one just an everyday question:

| Metric | Plain-English question | How it's worked out | Our number |
|---|---|---|---|
| **Accuracy** | Of every decision, how often were we right? | (TP+TN) ÷ everything | **0.885** |
| **Precision** | When we said "oil," how often were we right? | TP ÷ (TP+FP) | **0.835** |
| **Recall** | Of all the real oil, how much did we catch? | TP ÷ (TP+FN) | **0.705** |
| **F1 score** | One fair score balancing precision and recall | 2·(P·R) ÷ (P+R) | **0.765** |
| **ROC-AUC** | How well do we *rank* oil above dry? | area under the ROC curve | **0.831** |

What each one is really telling us here:

- **Accuracy 0.885** — we make the right call on about 89 of every 100 locations. Important
  context: because only ~28% of locations have oil, a lazy "never drill" guess already scores 72%.
  So 72% is the floor we have to beat, and we clearly do.
- **Precision 0.835** — when we get excited and drill, we strike oil about 84 times in 100; the
  other ~16 are dry holes (our 72 false positives).
- **Recall 0.705** — of all the oil that was really out there, we go and get about 70% of it. We
  leave ~30% in the ground (our 152 false negatives) because the model plays it safe.
- **F1 0.765** — a single number that stops us "cheating" either by only drilling ultra-safe bets
  (great precision, terrible recall) or by drilling everything (great recall, terrible precision).
  It's the harmonic mean, so it's only high when *both* precision and recall are decent.
- **ROC-AUC 0.831** — imagine lining up every location from most to least likely to have oil. AUC
  is the chance that a real oil location sits above a real dry one in that ranking. 0.5 is a coin
  flip, 1.0 is perfect; 0.83 is a genuinely useful drilling order.

One more idea ties all of these together: **the threshold.** The model doesn't actually output
"oil/dry" — it gives us a *probability* (say 0.62), and we turn that into a yes/no by cutting at
0.5. If we slide that cut *down*, we drill more freely: we catch more oil (recall goes up) but hit
more dry holes (precision goes down). Accuracy, precision and recall all move as we move the
threshold. ROC-AUC is the one exception, because it judges the ranking itself, so it doesn't depend
on where we put the cut.

> A note on why we don't just trust accuracy: because oil is rare here (~28%), a useless model that
> never drills already scores 72%. That's why we always report precision, recall and F1 next to it,
> so a high accuracy can't fool us into thinking a lazy model is a good one.

### D5b. The ROC curve and confusion matrix plot (figure 10)
Figure 10 shows two of these visually: the **confusion matrix** as a coloured 2×2 grid, and the
**ROC curve**, which traces the recall-vs-false-alarm trade-off as we slide the threshold from one
extreme to the other. The area under that curve is the AUC number above.

---

## Part E — Part 2: bringing in the geology (notebook section 5)

This is the half the judges weigh most heavily. We use rock physics to repair the data, then add
features built from petroleum-system thinking.

### E1. The five fixes
The brief asks us to correct inconsistencies in at least three columns. We did five:

1. **Trap labels** — merge "None" and blank into one `NoTrap` (needed for train and test to match).
2. **Shale permeability** — shale is a seal; it can't flow like a reservoir, so we cap its
   permeability at a low, tight-rock value.
3. **Shale porosity / near-zero porosity** — cap shale's porosity (it shouldn't be reservoir-grade)
   and lift values that are essentially zero (a rock with no holes can't hold oil).
4. **Filling missing permeability** — instead of a blind median, we estimate it from the rock's
   porosity using a standard rock-physics relationship (the idea behind the *Timur* and
   *Kozeny–Carman* equations: more porosity → more permeability, in a smooth curve).
5. **Filling missing porosity and depth** — use each rock type's own typical value, not one number
   for everything.

Figure 11 shows the before-and-after: after the fix, the shale points drop into the tight,
low-permeability corner where they physically belong.

### E2. The petroleum-system features we added
- **RQI (Reservoir Quality Index)** and **FZI (Flow Zone Indicator)** — real petrophysics formulas
  that combine porosity and permeability into a single "how good is this reservoir" number.
- **Element scores** — a 0-to-1 score for each ingredient: reservoir quality, trap quality, charge
  (how close the nearest oil field is), and the seismic score.
- **Chance-of-Success** — we *multiply* those element scores together, exactly the way exploration
  teams estimate the odds of a prospect. Figure 12 shows that this score, with no machine learning
  at all, already sorts locations from least to most likely to have oil. That's strong evidence our
  geological reasoning matches how the data was really built.

### E3. Re-running the models
We run the same six models again on this cleaned, geology-aware data and compare.

---

## Part F — Comparing, explaining, and being honest (notebook section 6)

### F1. Part 1 vs Part 2 (figure 13)
The two parts land at almost the same accuracy. That sounds underwhelming until you understand the
next point — and it's actually one of our strongest talking points.

### F2. How good can *any* model possibly be here?
The competition built the answers from geological rules **plus deliberate randomness** (to mimic
real-world uncertainty). That randomness puts a hard ceiling on accuracy that *no* model can cross
— this ceiling is called the **Bayes limit**. We estimate it and find it's about **0.82 across all
rows, and about 0.88 on the rows that look like the test set**. Our model already reaches it.

This is a big deal for two reasons:
1. It means the errors left over are baked into the data, not a flaw in our model.
2. If another team reports something like 95%, they've almost certainly made a mistake (leaked the
   answer or overfit). Knowing the ceiling protects us from chasing a fake number, and shows the
   judges we really understand the problem.

> Why is our test number (~0.88) higher than our all-rows number (~0.80)? Because a third of the
> *training* rows are missing the all-important `Seismic_Score`, which drags the training score
> down. **Every test row has a seismic score**, so on test-like data we do better.

### F3. Opening the black box with SHAP (figure 14)
A model can be accurate but mysterious. **SHAP** is a tool that shows which columns pushed each
prediction toward "oil" or "no oil." When we run it, the top drivers are exactly what a geologist
would expect: seismic score, having a trap, our Chance-of-Success score, and reservoir quality. In
other words, the model taught *itself* the petroleum system. That's a powerful thing to show.

---

## Part G — From predictions to a decision (notebook section 7)

### G1. Making the prediction files
We retrain our final ensembles on *all* the training data, then predict the 2,000 test locations.
We save two files, `..._Prediction_Part1.csv` and `..._Prediction_Part2.csv`, each one being the
test table with our predicted `Oil_Presence` column added on the end.

### G2. A drilling plan, not just 0s and 1s (figure 15)
Instead of bare yes/no labels, we sort the test locations into risk tiers from "very low" to "very
high." In real life you'd drill the "very high" prospects first, take a closer look at the
"moderate" ones, and skip the "very low" ones. This turns a column of numbers into something an
exploration manager could actually act on.

---

## Part H — Glossary (quick reference)

- **Feature** — an input column we use to predict (e.g. `Porosity`).
- **Target / label** — the thing we predict (`Oil_Presence`).
- **Classification** — predicting a category (here: oil or no oil).
- **Imputation** — filling in missing values with sensible estimates.
- **One-hot encoding** — turning a word column into yes/no number columns.
- **Cross-validation** — testing a model on data it didn't learn from, to get an honest score.
- **Class imbalance** — when one answer is much rarer than the other (oil is rarer here).
- **Accuracy** — the share of predictions we got right.
- **Precision** — of the locations we *called* oil, how many really were.
- **Recall** — of the locations that *really* had oil, how many we caught.
- **F1** — a single score that balances precision and recall.
- **ROC-AUC** — how well the model ranks oil above dry (0.5 = guessing, 1.0 = perfect).
- **Ensemble** — combining several models (we average their predictions).
- **SHAP** — a method that explains which features drove each prediction.
- **Bayes limit** — the best accuracy *any* model could reach, given the randomness in the data.
- **Petroleum system** — the geologist's checklist (reservoir + trap + charge + seal) that all has
  to be present for oil to accumulate.
- **Porosity** — the empty space in a rock (storage). **Permeability** — how easily fluid flows
  through it.

---

## Part I — How to run the whole thing

On your own computer:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook TeamName_PythonCode.ipynb
```
Then choose "Run All" and read it top to bottom. In Google Colab, upload the two CSV files first,
then Runtime → Run all.

If you can follow this walkthrough and explain figure 09 (the "all at once" chart) and the Bayes
ceiling idea in your own words, you understand the most important parts of our project.
