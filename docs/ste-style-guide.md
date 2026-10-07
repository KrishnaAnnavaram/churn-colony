# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of churn-colony and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the churn-colony documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **customer table** | The input CSV with one row for each customer and the 21 Telco columns | dataset (alone), file (alone) |
| **column contract** | The list of columns, types and allowed values in `schema.py` | schema file, spec |
| **raw column** | One of the 19 input columns that the model can use (`FEATURE_COLUMNS`) | input feature, attribute |
| **engineered feature** | A column that `TelcoFeatures` adds from the raw columns of one row | derived variable, new feature |
| **encoded column** | One column after the preprocessor (a scaled number or one one-hot level) | dummy, model feature |
| **target** | The `Churn` column, 1 for a customer who left | label (in prose), outcome, class |
| **churner** | A customer with target 1 | leaver, positive |
| **leak** | A column that contains information about the target that is not known at prediction time | leakage feature, cheat column |
| **leakage screen** | The check that flags a column whose single-column AUC is 0.95 or more | leak test, leak detector |
| **split** | One of the three disjoint parts: train, validation, test | partition, subset (for rows) |
| **pipeline** | The scikit-learn `Pipeline`: features, preprocessor, selector, model | workflow, chain |
| **selector** | The feature selection step: `none`, `l1`, `rfe` or `aco` | filter, picker |
| **ant colony selector** | `AntColonySelector`, the ACO selector | ACO wrapper, swarm |
| **ant** | One random column mask in one ACO iteration | agent, particle |
| **pheromone** | The value for each encoded column that sets its chance to be in a mask | weight (for ACO), trail |
| **fitness** | Inner CV score minus the size penalty of a mask | objective, reward |
| **size penalty** | `size_penalty` times the share of selected columns | complexity cost, regularisation |
| **resampling** | The class balance method: `none`, `smote`, `random` or `class_weight` | oversampling (for all methods), balancing |
| **candidate** | One model that `train` fits and scores on the validation split | contender, option |
| **churn score** | The predicted probability of the target for one customer | risk, prediction (alone), propensity |
| **threshold** | The churn score at or above which a customer goes on the contact list | cut-off, decision boundary |
| **contact list** | The customers whose churn score is at or above the threshold | target list, campaign list |
| **cost model** | The contact cost, the offer success rate and the value months | business model, cost matrix |
| **profit** | The expected saved value minus the contact cost of a contact list | gain, return, ROI |
| **run folder** | The folder with `model.joblib`, `metrics.json` and `model_card.md` | output, artefact folder |
| **model card** | The `model_card.md` report of one run | report (alone), datasheet |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **validate** | Check a customer table against the column contract |
| **split** | Divide the rows into the train, validation and test splits |
| **select** | Keep a subset of the encoded columns |
| **resample** | Add minority rows inside `fit` |
| **fit** | Learn the parameters of a pipeline step from training rows |
| **score** | Give a churn score to each customer |
| **tune** | Search hyperparameters with cross-validation |
| **evaluate** | Measure a saved pipeline on labelled rows |
| **explain** | Measure the effect of each raw column on a churn score |
