You are an expert Python and machine learning engineer. I am giving you a Jupyter notebook (`zomato_csao_datagen_v3.ipynb`) and a README file (`CHANGES_README.md`). Your job is to implement every single change described in the README into the notebook. This is a large task — take your time, be thorough, and do not stop until everything is done.

---

## CONTEXT

The notebook generates a synthetic dataset for a food delivery recommendation system (Zomato Cart Super Add-On / CSAO). It simulates users adding items to a cart and generates training data for a LightGBM ranker. The dataset has known bugs, realism gaps, and missing features that are all documented in the README.

---

## YOUR TASK

Read `CHANGES_README.md` in full before writing any code. Then implement every change in the order they appear in the README:

1. **Section 1 — Bug Fixes** (7 bugs): Fix cart_momentum, tautology in features, restaurant repeat behavior, PMI sparsity, cold start users, hour distribution in sessions, and user-restaurant ordering mismatch.

2. **Section 2 — User Data Realism** (5 improvements): Power-law order distribution, temporal concentration of orders, veg constraint hardening in three places, user_addon_rate_by_slot interaction feature, candidate_slot_urgency replacing binary fills_missing_slot.

3. **Section 3 — Revenue Optimization Features** (7 features): margin_score on menu items, candidate_revenue_potential, user_price_upgrade_tendency, revenue_weighted_label as a second training target, aov_lift_if_added, rest_avg_margin, candidate_in_price_sweet_spot. These features encode the business objective — the system must optimize for both user acceptance AND revenue simultaneously.

4. **Section 4 — Validation Cell**: Add a new cell with id `validate` between gen-sessions and save that runs 9 quality checks and prints a full report with OK/WARNING status for each.

5. **Section 5 — Save Cell Updates**: Update the save cell to include all new columns in each CSV, and print the full column list of cart_sessions.csv.

6. **Section 6 — Code Quality**: Add header comments to each major cell, remove dead code, ensure idempotency, add tqdm to the lift computation loop, add running stats every 5000 sessions.

---

## RULES

- Read the README completely before starting. Every change has exact code provided — use it.
- Implement changes in the cells specified. Cell IDs are given for every change.
- The cells you must NOT touch are: `llm-helpers`, `run-generation`, `fallback`, `inspect`, `restaurant-profiles`, `load-model`, `install`, `imports`. Do not modify these under any circumstances.
- All other cells are in scope: `gen-users`, `gen-orders`, `gen-features`, `gen-sessions`, `build-menu-table`, `save`.
- The new `validate` cell must be inserted between `gen-sessions` and `save` with cell id `validate`.
- Do not rename any existing columns — only add new ones or fix existing logic.
- Do not change the LLM generation logic.
- Every change from the README must appear in the final notebook. Do not skip or summarize — implement everything.
- When the README says "copy from gen-orders", literally copy the function definition into the other cell so it is self-contained.
- The notebook must be fully runnable top-to-bottom on a Colab A100 GPU after your changes.
- Where the README provides exact code snippets, use them exactly as written. Do not paraphrase or simplify.

---

## PROCESS

Work through the README section by section. For each change:
1. Identify the target cell by its id.
2. Read the full current source of that cell.
3. Make the precise change described — no more, no less.
4. Move to the next change.

Do not make changes that are not in the README. Do not refactor code that is not broken. Do not rename variables. Do not restructure cells. Only implement what is specified.

---

## VERIFICATION

After implementing all changes, verify the following before finishing:

- [ ] cart_momentum formula uses MEAL_SLOTS_NEEDED (not the old set division formula)
- [ ] user_addon_rate_30d, user_drink_rate_30d, user_dessert_rate_30d have Gaussian noise added in session_rows
- [ ] gen-orders has user_last_restaurant and user_restaurant_counts dicts
- [ ] pmi() function is removed, lift() and pair_seen_before() are present
- [ ] users_df has is_cold_start column
- [ ] cold start users have -1 for rate features in user_feat
- [ ] gen-sessions uses user_hour_dist(pref_slots) not hour_dist()
- [ ] gen-sessions picks uid first, then rid via pick_restaurant(uinfo)
- [ ] user_order_weights (power law) computed in both gen-orders and gen-sessions
- [ ] sample_order_date() function present in gen-orders, sim_end = datetime(2025,1,1)
- [ ] combo_prob first line is the veg hard filter
- [ ] slot_addon_wide computed and merged into user_feat
- [ ] candidate_slot_urgency present in session_rows (candidate_fills_missing_slot removed)
- [ ] margin_score column present in menu_df (built in build-menu-table cell)
- [ ] item_margin_lookup dict built in gen-sessions
- [ ] candidate_margin_score, candidate_revenue_potential, candidate_in_price_sweet_spot in session_rows
- [ ] revenue_weighted_label and aov_lift_if_added in session_rows
- [ ] user_price_upgrade_tendency computed in gen-features and in session_rows
- [ ] rest_avg_margin in rest_feat and in session_rows
- [ ] validate cell exists between gen-sessions and save with all 9 checks
- [ ] save cell prints full column list of cart_sessions.csv
- [ ] save cell includes all new columns for each CSV
- [ ] header comment blocks added to gen-users, gen-orders, gen-features, gen-sessions, validate, save
- [ ] tqdm added to lift computation in gen-features
- [ ] running stats printed every 5000 sessions in gen-sessions

Only submit when all checkboxes above are satisfied.
