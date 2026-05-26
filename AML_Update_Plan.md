# **AML Extended Build Remediation Specification**

## **1\. Document purpose**

This specification defines the required remediation work to stabilise and improve the AML Rules and ML Triage extended build across four focused sprints:

1. **Sprint 1:** Fix alert and band sizing.  
2. **Sprint 2:** Fix graph and cycle logic.  
3. **Sprint 3:** Fix case consolidation.  
4. **Sprint 4:** Improve explainability and reporting.

The objective is to convert the extended build from a technically complete prototype into a more reliable, decision-ready challenger that can be compared fairly against the MVP champion.

## **1.1 Implementation decisions and defaults**

The remediation build will be implemented as an **additive v2 layer** on top of the current MVP and extended pipeline. Existing MVP artifacts and commands must remain backward compatible unless this document explicitly says otherwise.

### **Implementation principles**

| Decision | Implementation detail |
| ----- | ----- |
| Backward compatibility | Existing commands such as `run-rules`, `train-model`, `train-extended-model`, `compare-models`, `build-report`, and `build-extended-report` remain valid. |
| v2 artifacts | Remediation outputs use explicit v2 or remediation file names where behavior changes materially. |
| Source of truth | All thresholds, caps, and report gates are config-driven. Hard-coded thresholds are allowed only as defaults loaded from config templates. |
| No suppression | No alert is auto-closed, deleted, suppressed, or hidden. Remediation changes queue priority, diagnostics, and case grouping only. |
| Evaluation split | Model and graph comparison decisions use validation for tuning/selection and test for final reporting. |
| Champion policy | MVP remains champion unless remediation evidence passes the explicit promotion gates in Sprint 4. |

### **Current pipeline order to preserve**

The one-command workflow must run in this dependency order:

```bash
pixi run run-workflow --preset full
```

Equivalent expanded order:

```bash
pixi run validate-data
pixi run create-splits
pixi run run-rules
pixi run tune-rules
pixi run build-features
pixi run train-model
pixi run prioritize-alerts
pixi run build-report
pixi run extended-stress-test
pixi run build-graph-features
pixi run run-graph-rules
pixi run train-extended-model
pixi run compare-models
pixi run consolidate-cases
pixi run explain-alerts
pixi run calibrate-scores
pixi run build-extended-report
```

Remediation commands will be added as a separate preset:

```bash
pixi run run-remediation
```

Equivalent target order:

```bash
pixi run diagnose-alerts
pixi run rebuild-priority-bands
pixi run build-graph-v2
pixi run run-graph-ablation
pixi run consolidate-cases-v2
pixi run generate-reason-codes
pixi run build-remediation-report
```

### **Canonical data sources**

| Input | Source path |
| ----- | ----- |
| Split transactions | `data/processed/base_transactions_with_splits.parquet` |
| Rule hits | `data/processed/rule_hits.parquet` |
| Consolidated alerts | `data/processed/alerts.parquet` |
| MVP features | `data/processed/alert_features.parquet` |
| MVP scores | `data/processed/scored_alerts.parquet` |
| Extended graph features | `data/processed/graph_features.parquet` |
| Extended alert features | `data/processed/extended_alert_features.parquet` |
| Extended scores | `data/processed/extended_scored_alerts.parquet` |
| Graph rule hits | `data/processed/graph_rule_hits.parquet` |
| Calibrated scores | `data/processed/calibrated_scores.parquet` |

### **Shared remediation config defaults**

The following defaults should be used unless later evidence requires change:

| Config area | Default |
| ----- | ----- |
| Daily investigators | `3` |
| Alerts per investigator per day | `40` |
| Total daily capacity | `120` |
| Primary model comparison metric | `precision_at_k` |
| Primary model comparison K | `1000` |
| Minimum extended uplift for promotion | `5% relative Precision@1000 uplift` |
| Case max alerts | `100` |
| Case max accounts | `50` |
| Case max transactions | `200` |
| Mega-component threshold | `1000 accounts or alerts` |
| Graph ablation decision | Keep graph group only if Precision@1000 delta is non-negative and no major runtime issue occurs. |
| Forced graph features | Disabled by default during remediation; re-enable only for groups with `decision=keep` from ablation. |

---

## **2\. Current-state issues to address**

| Area | Current finding | Impact | Sprint |
| ----- | ----- | ----- | ----- |
| Alert generation | 3,365,497 alerts from 6,924,049 transactions; alert rate 48.6% | Rule layer is too broad and creates excessive investigation load | Sprint 1 |
| P1 sizing | 2,316,065 P1 alerts | P1 is not operationally meaningful | Sprint 1 |
| Model uplift | Extended model underperforms MVP at Precision@100, @500, @1,000 | Extended model should not replace MVP | Sprint 1 and 4 |
| Cycle logic | 36,900 cycle candidates from 50,000 evaluated alerts, but 0 labelled laundering cases | Cycle logic creates weak or irrelevant signals | Sprint 2 |
| Case consolidation | Max alerts per case around 3.15M | Case linkage is too broad; mega-case problem | Sprint 3 |
| SHAP reason codes | Reason codes are technical, not AML-investigator friendly | Weak usability for investigation and governance | Sprint 4 |
| Report output | Report is artifact-heavy but decision-light | Needs executive interpretation and traffic-light conclusion | Sprint 4 |

---

# **Sprint 1 Specification: Fix Alert and Band Sizing**

## **1.1 Objective**

Reduce alert inflation and redesign priority bands so that P1 to P4 reflect operational investigation capacity, not broad score quantiles or excessive rule overrides.

The current P1 population is too large to be useful: more than **2.31M alerts** are assigned to P1, while P1 precision is below 0.1%.

---

## **1.2 Scope**

### **Included**

| Item | Description |
| ----- | ----- |
| Rule-level alert contribution analysis | Identify which rules drive alert volume and low precision. |
| Alert contribution waterfall | Show cumulative alert volume and label capture by rule. |
| P1/P2/P3/P4 redesign | Redefine bands using capacity-aware thresholds. |
| Critical rule override review | Prevent broad overrides from flooding P1. |
| Capacity simulation | Simulate daily review capacity and top-K performance. |
| Top-K rerun | Recalculate Precision@100, @500, @1,000, @5,000 and Recall@K. |

### **Excluded**

| Item | Reason |
| ----- | ----- |
| Auto-closing alerts | Not permitted at this stage. |
| Removing rules entirely without evidence | Rules must be tuned or reweighted through governance. |
| Production queue integration | Remains out of scope. |

---

## **1.3 Required artifacts**

| Artifact | File path | Format |
| ----- | ----- | ----- |
| Rule contribution table | `outputs/metrics/rule_alert_contribution.csv` | CSV |
| Rule contribution waterfall | `outputs/charts/rule_alert_waterfall.png` | PNG |
| Revised band config | `config/priority_band_config.yaml` | YAML |
| Band summary | `outputs/metrics/priority_band_summary.csv` | CSV |
| Capacity simulation | `outputs/metrics/daily_capacity_simulation.csv` | CSV |
| Top-K comparison | `outputs/metrics/top_k_after_band_fix.csv` | CSV |
| Sprint report section | `outputs/reports/sections/sprint_1_alert_band_sizing.html` | HTML |

---

## **1.4 Functional requirements**

### **FR1. Rule-level alert contribution**

The system must produce a rule-level table with the following columns:

| Column | Description |
| ----- | ----- |
| `rule_id` | Rule identifier, for example `R1_AMOUNT`, `R3_VELOCITY`. |
| `rule_name` | Human-readable rule name. |
| `alert_count` | Number of alerts triggered by the rule. |
| `unique_alert_count` | Number of unique consolidated alerts where the rule appears. |
| `label_count` | Number of labelled laundering alerts captured by the rule. |
| `precision` | `label_count / unique_alert_count`. |
| `recall_contribution` | Label count captured by this rule divided by total labelled alerts. |
| `incremental_label_count` | Additional labels captured after previously ranked rules. |
| `incremental_alert_count` | Additional alerts added after previously ranked rules. |
| `overlap_rate` | Percentage of this rule’s alerts also triggered by another rule. |
| `recommended_action` | `keep`, `tighten`, `downgrade`, `review`, or `retire_candidate`. |

### **FR2. Alert contribution waterfall**

The system must produce a waterfall showing:

1. Total transactions.  
2. Rule-triggered alert count by rule.  
3. De-duplicated consolidated alerts.  
4. Alerts by priority band.  
5. Labels captured at each stage.

The waterfall should highlight rules that add large alert volume but low incremental label capture.

### **FR3. Priority band redesign**

The priority banding logic must support both **score-based thresholds** and **capacity-based thresholds**.

Recommended configuration:

priority\_bands:

  method: hybrid\_capacity\_score

  daily\_capacity\_assumption:

    investigators: 3

    alerts\_per\_investigator\_per\_day: 40

    total\_daily\_capacity: 120

  bands:

    P1:

      description: "Critical queue for immediate review"

      max\_daily\_alerts: 120

      score\_percentile\_floor: 99.9

      critical\_rule\_override: true

    P2:

      description: "High-priority queue"

      max\_daily\_alerts: 500

      score\_percentile\_floor: 99.5

    P3:

      description: "Medium-priority monitoring queue"

      score\_percentile\_floor: 95.0

    P4:

      description: "Low-priority retained alerts"

      score\_percentile\_floor: 0.0

### **FR4. Critical rule override control**

Critical rule override must not automatically promote broad rule hits to P1 unless at least one additional condition is met.

Example logic:

Promote to P1 if:

    critical\_rule\_hit \= 1

AND (

    calibrated\_score \>= P1\_score\_threshold

    OR rule\_count \>= 2

    OR graph\_high\_risk\_flag \= 1

    OR amount \>= segment\_p99\_threshold

)

### **FR5. Capacity simulation**

The system must simulate the following scenarios:

| Scenario | Description |
| ----- | ----- |
| Top 100 alerts | High-capacity daily urgent review queue. |
| Top 500 alerts | Weekly or expanded review queue. |
| Top 1,000 alerts | Benchmark against current report. |
| Top 5,000 alerts | Broader monitoring benchmark. |
| Daily capacity | Based on investigator count and expected alerts per investigator. |

Required metrics:

| Metric | Formula |
| ----- | ----- |
| `precision_at_k` | Labelled alerts in top K / K |
| `recall_at_k` | Labelled alerts in top K / total labelled alerts |
| `lift_at_k` | Precision@K / base label rate |
| `alerts_per_day` | Band alert count divided by number of days |
| `labels_per_day` | Band label count divided by number of days |
| `coverage_at_capacity` | Recall captured within daily capacity |

---

## **1.5 Non-functional requirements**

| Requirement | Description |
| ----- | ----- |
| Reproducibility | Banding must run from saved config. |
| Auditability | All thresholds and overrides must be logged. |
| Explainability | Every P1 alert must have a reason for P1 assignment. |
| Safety | No alert should be auto-closed. |
| Performance | Banding should run within the existing report pipeline. |

---

## **1.6 Acceptance criteria**

Sprint 1 is complete when:

| Criterion | Minimum standard |
| ----- | ----- |
| Alert contribution | Rule-level contribution table generated. |
| Waterfall | Alert waterfall chart available in report. |
| Band sizing | P1 no longer contains millions of alerts unless explicitly configured for research mode. |
| Top-K rerun | Precision@100, @500, @1,000, and @5,000 recalculated. |
| Champion comparison | MVP versus extended comparison updated after band fix. |
| Governance | All P1 override logic documented. |

## **1.7 Implementation details**

### **Modules and commands**

Implement Sprint 1 in:

| Component | Detail |
| ----- | ----- |
| Diagnostics module | `src/aml_mvp/diagnostics/rule_contribution.py` |
| Waterfall module | `src/aml_mvp/diagnostics/alert_waterfall.py` |
| Band sizing module | `src/aml_mvp/calibration/band_sizing.py` |
| Capacity module | Extend existing `src/aml_mvp/triage/capacity_simulation.py` or add `src/aml_mvp/calibration/capacity_simulation.py` if v2 logic diverges. |
| CLI | Add `diagnose-alerts` and `rebuild-priority-bands`. |
| Pixi tasks | Add `diagnose-alerts` and `rebuild-priority-bands`. |

### **Config: `config/priority_band_config.yaml`**

Required keys:

```yaml
artifacts:
  scored_alerts_path: data/processed/extended_scored_alerts.parquet
  fallback_scored_alerts_path: data/processed/scored_alerts.parquet
  rule_hits_path: data/processed/rule_hits.parquet
  alerts_path: data/processed/alerts.parquet
  transactions_path: data/processed/base_transactions_with_splits.parquet
  output_priority_alerts_path: data/processed/priority_alerts_v2.parquet
  rule_contribution_path: outputs/metrics/rule_alert_contribution.csv
  waterfall_chart_path: outputs/charts/rule_alert_waterfall.png
  band_summary_path: outputs/metrics/priority_band_summary.csv
  capacity_simulation_path: outputs/metrics/daily_capacity_simulation.csv
  top_k_after_band_fix_path: outputs/metrics/top_k_after_band_fix.csv

priority_bands:
  method: hybrid_capacity_score
  score_column: calibrated_score
  fallback_score_column: model_score
  daily_capacity_assumption:
    investigators: 3
    alerts_per_investigator_per_day: 40
  bands:
    P1:
      max_daily_alerts: 120
      score_percentile_floor: 99.9
      critical_rule_override: true
    P2:
      max_daily_alerts: 500
      score_percentile_floor: 99.5
    P3:
      score_percentile_floor: 95.0
    P4:
      score_percentile_floor: 0.0
  critical_override_conditions:
    min_rule_count: 2
    require_score_floor: true
    allow_graph_high_risk_flag: true
    allow_segment_p99_amount: true
```

### **Band output schema**

`data/processed/priority_alerts_v2.parquet` must include:

| Column | Description |
| ----- | ----- |
| `alert_id` | Existing alert id. |
| `transaction_id` | Source transaction id. |
| `priority_band` | Existing or fallback band if present. |
| `priority_band_v2` | New P1-P4 assignment. |
| `priority_rank_v2` | P1=1, P2=2, P3=3, P4=4. |
| `priority_score_v2` | Score used for ranking. |
| `p1_override_reason` | Empty unless promoted by override. |
| `band_assignment_reason` | Human-readable reason. |
| `is_laundering` or `target` | Label for evaluation. |

### **Implementation notes**

- `diagnose-alerts` must not mutate rule outputs; it only reads saved artifacts and writes diagnostics.
- `rebuild-priority-bands` must prefer `calibrated_score` if present, otherwise use `model_score`.
- Critical overrides must never promote all hits from broad rules. A critical rule hit needs at least one additional configured condition.
- Alert waterfall can use `matplotlib` because it is already in `pixi.toml`.
- `build-report` and `build-extended-report` should read Sprint 1 outputs if present, but must still render when Sprint 1 outputs are absent.

### **Sprint 1 tests**

Add:

| Test | Required assertions |
| ----- | ----- |
| `tests/test_rule_contribution.py` | Rule counts, unique alert counts, overlap rate, incremental labels are correct on toy data. |
| `tests/test_priority_band_sizing.py` | P1 respects daily capacity and broad critical overrides are constrained. |
| `tests/test_capacity_simulation.py` | Precision@K, Recall@K, lift, and daily capacity rows match formulas. |
| `tests/test_remediation_report.py` | Sprint 1 section renders when artifacts exist and report still renders when absent. |

Run:

```bash
python -m pytest tests/test_rule_contribution.py tests/test_priority_band_sizing.py tests/test_capacity_simulation.py
pixi run diagnose-alerts
pixi run rebuild-priority-bands
```

---

# **Sprint 2 Specification: Fix Graph and Cycle Logic**

## **2.1 Objective**

Improve graph intelligence so that graph features and graph rules contribute meaningful AML detection value, rather than increasing noise.

The current cycle logic is not useful: **36,900 cycle candidates** were identified among **50,000 evaluated alert transactions**, but **0 labelled positives** were captured.

---

## **2.2 Scope**

### **Included**

| Item | Description |
| ----- | ----- |
| Revised gather-scatter rule | Improve sequencing and reduce false candidates. |
| Revised cycle candidate logic | Add amount, timing, and path constraints. |
| Graph ablation testing | Measure whether graph features improve performance. |
| Forced feature review | Remove forced graph features unless validated. |
| Graph diagnostics | Report graph feature distribution and correlation. |

### **Excluded**

| Item | Reason |
| ----- | ----- |
| Full GNN modelling | Out of scope for this remediation cycle. |
| Real-time graph engine | Not required for batch prototype. |
| Full entity resolution | Dataset limitations remain. |

---

## **2.3 Required artifacts**

| Artifact | File path | Format |
| ----- | ----- | ----- |
| Revised graph rule config | `config/graph_rule_config.yaml` | YAML |
| Graph feature table | `data/processed/graph_features_v2.parquet` | Parquet |
| Graph rule hits | `data/processed/graph_rule_hits_v2.parquet` | Parquet |
| Cycle candidate table | `outputs/metrics/cycle_candidates_v2.csv` | CSV |
| Graph ablation results | `outputs/metrics/graph_ablation_results.csv` | CSV |
| Graph diagnostic section | `outputs/reports/sections/sprint_2_graph_diagnostics.html` | HTML |

---

## **2.4 Functional requirements**

### **FR1. Point-in-time graph snapshots**

Graph features must be computed using only edges observed **before** or **at** the alert timestamp, excluding the current alert transaction where necessary.

Required feature windows:

| Window | Purpose |
| ----- | ----- |
| 1 day | Capture bursts and short-term layering. |
| 7 days | Capture campaign-level mule behaviour. |
| 30 days | Capture broader network behaviour. |
| Lifetime-to-date | Capture long-term centrality and account role. |

### **FR2. Gather-scatter v2 logic**

A gather-scatter candidate should require:

receiver receives from multiple unique senders within T1

AND same receiver sends to multiple unique receivers within T2 after receiving funds

AND outgoing amount is materially related to incoming amount

AND activity occurs within configured temporal window

Recommended parameters:

| Parameter | Suggested initial value |
| ----- | ----- |
| `min_unique_sources` | 3 |
| `min_unique_destinations` | 3 |
| `max_gather_window_hours` | 24 |
| `max_scatter_window_hours` | 48 |
| `min_in_out_amount_ratio` | 0.5 |
| `max_in_out_amount_ratio` | 1.5 |
| `min_total_amount_percentile` | P75 by segment |

### **FR3. Cycle detection v2 logic**

A cycle candidate should be flagged only when:

A directed path returns to an earlier account

AND path length is between 3 and 5

AND all transactions occur within configured time window

AND amount decay is within tolerance

AND path contains at least one high-risk signal

Recommended parameters:

| Parameter | Suggested initial value |
| ----- | ----- |
| `min_path_length` | 3 |
| `max_path_length` | 5 |
| `max_cycle_window_hours` | 72 |
| `max_amount_decay_ratio` | 0.5 |
| `min_total_cycle_amount_percentile` | P75 |
| `require_high_risk_rule_flag` | true |

### **FR4. Graph ablation testing**

The model pipeline must compare:

| Model | Feature set |
| ----- | ----- |
| A | MVP features only |
| B | MVP \+ degree features |
| C | MVP \+ degree \+ component features |
| D | MVP \+ degree \+ component \+ PageRank |
| E | MVP \+ all graph features including cycle involvement |

Required output columns:

| Column | Description |
| ----- | ----- |
| `model_variant` | A, B, C, D, or E |
| `feature_group_added` | Graph feature group added |
| `pr_auc` | PR-AUC |
| `precision_at_100` | Precision@100 |
| `precision_at_500` | Precision@500 |
| `precision_at_1000` | Precision@1,000 |
| `recall_at_1000` | Recall@1,000 |
| `winner_vs_baseline` | true/false |
| `decision` | keep, remove, or research\_only |

### **FR5. Forced graph feature control**

No graph feature should be forced into the model unless:

feature has AML rationale

AND feature is stable across train/test or HI/LI split

AND ablation test shows non-negative top-K impact

---

## **2.5 Non-functional requirements**

| Requirement | Description |
| ----- | ----- |
| Runtime control | Graph rules must complete within defined processing time. |
| Memory safety | Graph calculations should support chunking or windowed processing. |
| Auditability | Graph paths and rule evidence must be saved. |
| Reproducibility | Graph feature generation must be config-driven. |

---

## **2.6 Acceptance criteria**

Sprint 2 is complete when:

| Criterion | Minimum standard |
| ----- | ----- |
| Gather-scatter v2 | Generates fewer but higher-quality candidates. |
| Cycle v2 | Candidate count reduced and label capture tested. |
| Ablation | Graph feature contribution is quantified. |
| Forced features | Forced graph features removed unless validated. |
| Report | Graph diagnostics added to HTML report. |

## **2.7 Implementation details**

### **Modules and commands**

Implement Sprint 2 in additive v2 modules:

| Component | Detail |
| ----- | ----- |
| Feature builder | `src/aml_mvp/graph/graph_features_v2.py` |
| Gather-scatter rule | `src/aml_mvp/graph/gather_scatter_v2.py` |
| Cycle rule | `src/aml_mvp/graph/cycle_detection_v2.py` |
| Ablation | `src/aml_mvp/graph/graph_ablation.py` |
| CLI | Add `build-graph-v2` and `run-graph-ablation`. |
| Pixi tasks | Add `build-graph-v2` and `run-graph-ablation`. |

### **Config: `config/graph_rule_config.yaml`**

Required keys:

```yaml
artifacts:
  transactions_path: data/processed/base_transactions_with_splits.parquet
  alerts_path: data/processed/alerts.parquet
  alert_features_path: data/processed/alert_features.parquet
  graph_features_v2_path: data/processed/graph_features_v2.parquet
  extended_alert_features_v2_path: data/processed/extended_alert_features_v2.parquet
  graph_rule_hits_v2_path: data/processed/graph_rule_hits_v2.parquet
  cycle_candidates_v2_path: outputs/metrics/cycle_candidates_v2.csv
  graph_ablation_results_path: outputs/metrics/graph_ablation_results.csv

graph_features:
  windows: [1d, 7d, 30d, lifetime]
  include_degree: true
  include_weighted_degree: true
  include_component_size: true
  include_pagerank_proxy: true
  include_cycle_flags: true

gather_scatter_v2:
  min_unique_sources: 3
  min_unique_destinations: 3
  max_gather_window_hours: 24
  max_scatter_window_hours: 48
  min_in_out_amount_ratio: 0.5
  max_in_out_amount_ratio: 1.5
  min_total_amount_percentile: 0.75

cycle_v2:
  min_path_length: 3
  max_path_length: 5
  max_cycle_window_hours: 72
  max_amount_decay_ratio: 0.5
  min_total_cycle_amount_percentile: 0.75
  require_high_risk_rule_flag: true

ablation:
  k_values: [100, 500, 1000, 5000]
  decision_metric: precision_at_1000
  keep_min_delta: 0.0
```

### **Graph feature v2 schema**

`data/processed/graph_features_v2.parquet` must include one row per alert transaction and:

| Column pattern | Description |
| ----- | ----- |
| `alert_id`, `transaction_id`, `alert_timestamp` | Join keys. |
| `graph_<window>_sender_out_degree` | Unique receivers before alert. |
| `graph_<window>_receiver_in_degree` | Unique senders before alert. |
| `graph_<window>_sender_weighted_out_amount` | Prior outgoing amount. |
| `graph_<window>_receiver_weighted_in_amount` | Prior incoming amount. |
| `graph_<window>_component_size` | Component size in the configured window. |
| `graph_<window>_pagerank_proxy_sender` | Degree-normalized proxy unless full PageRank is made efficient. |
| `graph_cycle_v2_flag` | 1 if validated cycle criteria are met. |
| `graph_gather_scatter_v2_flag` | 1 if validated gather-scatter criteria are met. |

### **Graph rule hit v2 evidence**

`data/processed/graph_rule_hits_v2.parquet` follows the existing rule-hit schema and must include evidence JSON with:

| Field | Description |
| ----- | ----- |
| `path_accounts` | Ordered account path for cycle hits. |
| `path_transaction_ids` | Ordered transactions for cycle or gather-scatter path. |
| `window_start`, `window_end` | Time window supporting hit. |
| `total_in_amount`, `total_out_amount` | Gather-scatter evidence. |
| `amount_ratio` | Out/in ratio or path amount decay ratio. |
| `high_risk_rule_flags` | Existing rules that support graph hit. |

### **Graph ablation behavior**

- Train/evaluate ablation variants using the same balancing, feature selection, Optuna tuning, and validation/test policy as current model training.
- Do not force graph features into the extended model by default during remediation.
- After ablation, write a machine-readable decision for each graph group:
  - `keep` if Precision@1000 delta is non-negative and no runtime/data quality issue is flagged.
  - `remove` if Precision@1000 delta is negative.
  - `research_only` if candidate count is high but label evidence is weak.
- `train-extended-model` may force graph features only when the corresponding graph group has `decision=keep`.

### **Sprint 2 tests**

Add:

| Test | Required assertions |
| ----- | ----- |
| `tests/test_graph_rules_v2.py` | Gather-scatter and cycle toy paths satisfy strict timing/amount/path constraints. |
| `tests/test_graph_ablation.py` | Ablation variants A-E are generated and decisions follow configured metric deltas. |
| `tests/test_graph_feature_leakage.py` | Current alert transaction is not used as historical evidence for its own features. |
| `tests/test_graph_runtime_config.py` | Window and cap settings are read from config. |

Run:

```bash
python -m pytest tests/test_graph_rules_v2.py tests/test_graph_ablation.py tests/test_graph_feature_leakage.py
pixi run build-graph-v2
pixi run run-graph-ablation
```

---

# **Sprint 3 Specification: Fix Case Consolidation**

## **3.1 Objective**

Improve operational case quality by preventing mega-cases, reducing duplicate alerts, and creating meaningful case typology labels.

The current report shows **160,799 cases**, average **20.9 alerts per case**, but a maximum of approximately **3.15M alerts in one case**, indicating that linkage logic is too broad.

---

## **3.2 Scope**

### **Included**

| Item | Description |
| ----- | ----- |
| Case-size cap | Prevent operationally unusable mega-cases. |
| Mega-component split logic | Split large graph components into manageable cases. |
| Case-level precision and recall | Evaluate case quality, not just alert quality. |
| Case typology labels | Assign case type based on dominant pattern. |
| Case-level priority | Score and rank consolidated cases. |

### **Excluded**

| Item | Reason |
| ----- | ----- |
| Full case management integration | Outside prototype scope. |
| Investigator assignment workflow | Not required for current build. |
| Manual investigation feedback loop | Future enhancement. |

---

## **3.3 Required artifacts**

| Artifact | File path | Format |
| ----- | ----- | ----- |
| Consolidation config | `config/case_consolidation_config.yaml` | YAML |
| Consolidated case table | `data/processed/consolidated_cases_v2.parquet` | Parquet |
| Alert-to-case mapping | `data/processed/alert_case_mapping_v2.parquet` | Parquet |
| Case quality metrics | `outputs/metrics/case_quality_metrics.csv` | CSV |
| Mega-case diagnostics | `outputs/metrics/mega_case_diagnostics.csv` | CSV |
| Case typology summary | `outputs/metrics/case_typology_summary.csv` | CSV |
| Case report section | `outputs/reports/sections/sprint_3_case_consolidation.html` | HTML |

---

## **3.4 Functional requirements**

### **FR1. Case linkage hierarchy**

The system should consolidate alerts using the following hierarchy:

| Priority | Linkage criterion | Condition |
| ----- | ----- | ----- |
| 1 | Same transaction | Always consolidate duplicate rule hits. |
| 2 | Same sender-receiver pair | Consolidate within configured time window. |
| 3 | Same account plus typology | Consolidate if same account participates in same typology. |
| 4 | Gather-scatter group | Consolidate linked fan-in/fan-out alerts. |
| 5 | Cycle group | Consolidate short-cycle path alerts. |
| 6 | Graph component | Use only if component size is below cap. |

### **FR2. Case-size cap**

Recommended configuration:

case\_consolidation:

  max\_alerts\_per\_case: 100

  max\_accounts\_per\_case: 50

  max\_transactions\_per\_case: 200

  max\_component\_size\_for\_auto\_case: 1000

  split\_large\_components: true

If a case exceeds the cap, the system must split it by:

1. Time window.  
2. Dominant account.  
3. Payment format.  
4. Typology.  
5. Community or connected subcomponent.

### **FR3. Mega-component split logic**

Mega-components must not become investigation cases. They should be labelled as network intelligence objects.

If component\_size \> max\_component\_size\_for\_auto\_case:

    create network\_cluster\_id

    split into operational cases using time \+ typology \+ account role

    do not assign entire component as one case

### **FR4. Case-level schema**

Required case table:

| Column | Description |
| ----- | ----- |
| `case_id` | Unique case identifier. |
| `case_created_at` | First alert timestamp in case. |
| `case_end_at` | Last alert timestamp in case. |
| `case_typology` | Dominant typology label. |
| `case_priority_band` | P1, P2, P3, or P4. |
| `case_score` | Aggregated risk score. |
| `alert_count` | Number of alerts in case. |
| `transaction_count` | Number of transactions. |
| `account_count` | Number of unique accounts. |
| `total_amount` | Total amount in case. |
| `max_transaction_amount` | Highest transaction amount. |
| `rule_ids` | Rules represented in the case. |
| `has_laundering_label` | 1 if any alert in case is labelled laundering. |
| `label_count` | Number of labelled alerts in case. |
| `case_rationale` | Human-readable case summary. |

### **FR5. Case typology labelling**

Case typology should be assigned using a rule hierarchy:

| Typology | Assignment rule |
| ----- | ----- |
| Cycle | Case contains validated cycle path. |
| Gather-scatter | Multiple sources and multiple destinations through hub account. |
| Fan-in | Many source accounts to one receiver. |
| Fan-out | One sender to many receivers. |
| Pass-through | Incoming and outgoing funds within short time window. |
| Structuring | Repeated small or threshold-adjacent transactions. |
| High amount | Dominated by high-value transaction. |
| Mixed | Multiple typologies without clear dominant pattern. |

### **FR6. Case-level metrics**

Required metrics:

| Metric | Description |
| ----- | ----- |
| `case_count` | Total number of cases. |
| `avg_alerts_per_case` | Mean case size. |
| `p95_alerts_per_case` | Tail case size indicator. |
| `max_alerts_per_case` | Must be below configured cap unless flagged as network object. |
| `case_precision` | Labelled cases / total cases. |
| `case_recall` | Labels captured in cases / total labelled alerts. |
| `case_reduction_ratio` | 1 \- case count / alert count. |
| `p1_case_count` | Number of P1 cases. |
| `p1_case_precision` | Precision of P1 cases. |

---

## **3.5 Non-functional requirements**

| Requirement | Description |
| ----- | ----- |
| Traceability | Every alert must map to exactly one operational case or one network-intelligence cluster. |
| Interpretability | Every case must have a rationale. |
| Operational usability | Cases must be small enough for analyst review. |
| Configurability | Case caps and split rules must be configurable. |

---

## **3.6 Acceptance criteria**

Sprint 3 is complete when:

| Criterion | Minimum standard |
| ----- | ----- |
| Mega-case fixed | No operational case contains millions of alerts. |
| Case cap | Case size cap applied and tested. |
| Case metrics | Case-level precision/recall reported. |
| Case typology | Every case has a typology label. |
| Traceability | Alert-to-case mapping complete. |
| Report | Case quality section added to HTML report. |

## **3.7 Implementation details**

### **Modules and commands**

Implement Sprint 3 in:

| Component | Detail |
| ----- | ----- |
| Case consolidation | `src/aml_mvp/cases/case_consolidation_v2.py` |
| Mega split | `src/aml_mvp/cases/mega_component_splitter.py` |
| Typology | `src/aml_mvp/cases/case_typology.py` |
| Metrics | `src/aml_mvp/cases/case_metrics.py` |
| CLI | Add `consolidate-cases-v2`. |
| Pixi task | Add `consolidate-cases-v2`. |

### **Config: `config/case_consolidation_config.yaml`**

Required keys:

```yaml
artifacts:
  transactions_path: data/processed/base_transactions_with_splits.parquet
  alerts_path: data/processed/alerts.parquet
  rule_hits_path: data/processed/rule_hits.parquet
  graph_rule_hits_path: data/processed/graph_rule_hits_v2.parquet
  scored_alerts_path: data/processed/extended_scored_alerts.parquet
  fallback_scored_alerts_path: data/processed/scored_alerts.parquet
  output_cases_path: data/processed/consolidated_cases_v2.parquet
  alert_case_mapping_path: data/processed/alert_case_mapping_v2.parquet
  network_clusters_path: data/processed/network_clusters_v2.parquet
  case_quality_metrics_path: outputs/metrics/case_quality_metrics.csv
  mega_case_diagnostics_path: outputs/metrics/mega_case_diagnostics.csv
  case_typology_summary_path: outputs/metrics/case_typology_summary.csv

case_consolidation:
  max_alerts_per_case: 100
  max_accounts_per_case: 50
  max_transactions_per_case: 200
  max_component_size_for_auto_case: 1000
  split_large_components: true
  pair_window_hours: 72
  same_account_typology_window_hours: 168
  time_split_hours_for_mega_components: 24
```

### **Alert-to-case mapping schema**

`data/processed/alert_case_mapping_v2.parquet` must include:

| Column | Description |
| ----- | ----- |
| `alert_id` | Alert id. |
| `transaction_id` | Source transaction id. |
| `case_id` | Operational case id if mapped to a case. |
| `network_cluster_id` | Network intelligence cluster id if not suitable as an operational case. |
| `mapping_type` | `operational_case` or `network_cluster`. |
| `mapping_reason` | Linkage reason used. |

### **Network cluster schema**

`data/processed/network_clusters_v2.parquet` must include:

| Column | Description |
| ----- | ----- |
| `network_cluster_id` | Cluster id. |
| `component_size` | Number of linked alerts/accounts. |
| `alert_count` | Alerts in cluster. |
| `account_count` | Unique accounts. |
| `split_reason` | Why it was not an operational case. |
| `recommended_use` | `network_intelligence_only`. |

### **Implementation notes**

- Every alert must appear exactly once in `alert_case_mapping_v2`.
- A row can have either `case_id` or `network_cluster_id`, not both.
- Mega-components must be split before operational case aggregation.
- Case-level score should default to `max(model_score)` or `max(calibrated_score)` where available, with `avg_score` retained as a secondary metric.
- Case priority should use the highest alert priority in the case unless case score places it into a higher band.
- Existing `consolidate-cases` remains available; `consolidate-cases-v2` writes new artifacts.

### **Sprint 3 tests**

Add:

| Test | Required assertions |
| ----- | ----- |
| `tests/test_case_consolidation_v2.py` | Caps are respected and every alert maps once. |
| `tests/test_mega_component_splitter.py` | Mega-components become network clusters and smaller operational cases. |
| `tests/test_case_typology.py` | Typology hierarchy assigns cycle, gather-scatter, fan-in/out, pass-through, structuring, high amount, or mixed. |
| `tests/test_case_metrics.py` | Case precision, recall, reduction ratio, and P1 metrics match formulas. |

Run:

```bash
python -m pytest tests/test_case_consolidation_v2.py tests/test_mega_component_splitter.py tests/test_case_typology.py tests/test_case_metrics.py
pixi run consolidate-cases-v2
```

---

# **Sprint 4 Specification: Improve Explainability and Reporting**

## **4.1 Objective**

Make the extended build decision-ready for AML stakeholders by improving reason codes, executive reporting, champion-challenger conclusions, and traffic-light findings.

The current report contains useful technical outputs, including SHAP features and tuning trials, but the narrative needs to clearly tell management whether the extended build should replace, challenge, or remain behind the MVP model. The report already concluded that the MVP should be retained because the extended model had negative delta at the primary top-K metric.

---

## **4.2 Scope**

### **Included**

| Item | Description |
| ----- | ----- |
| AML business reason-code dictionary | Translate technical features into investigator-friendly explanations. |
| Executive summary page | Summarise decision, evidence, and next actions. |
| Champion-challenger conclusion | Formalise MVP versus extended decision. |
| HTML traffic-light dashboard | Red/amber/green status by module. |
| Report quality controls | Ensure metrics reconcile and decisions are visible. |

### **Excluded**

| Item | Reason |
| ----- | ----- |
| PDF automation | Optional unless required later. |
| Interactive dashboard hosting | Not required. |
| Investigator feedback UI | Future enhancement. |

---

## **4.3 Required artifacts**

| Artifact | File path | Format |
| ----- | ----- | ----- |
| Reason-code dictionary | `config/reason_code_dictionary.yaml` | YAML |
| Reason-code output | `outputs/metrics/alert_reason_codes.csv` | CSV |
| Executive summary | `outputs/reports/sections/executive_summary.html` | HTML |
| Champion decision file | `outputs/metrics/champion_challenger_decision.json` | JSON |
| Traffic-light findings | `outputs/metrics/traffic_light_findings.csv` | CSV |
| Final report | `outputs/reports/aml_extended_remediation_report.html` | HTML |

---

## **4.4 Functional requirements**

### **FR1. AML business reason-code dictionary**

The reason-code dictionary must map technical feature names to:

| Field | Description |
| ----- | ----- |
| `feature_name` | Technical feature name. |
| `business_reason_code` | Investigator-friendly reason. |
| `risk_direction` | `risk_increasing`, `risk_reducing`, or `contextual`. |
| `typology_mapping` | Fan-in, fan-out, pass-through, structuring, cycle, etc. |
| `plain_language_template` | Alert explanation sentence. |
| `investigator_note` | What the investigator should check. |

Example:

feature\_sender\_prior\_unique\_receivers:

  business\_reason\_code: "Sender distributed funds to many counterparties"

  risk\_direction: "risk\_increasing"

  typology\_mapping: "fan-out / layering"

  plain\_language\_template: "The sender has previously transferred funds to many unique receivers, which may indicate distribution or layering behaviour."

  investigator\_note: "Review whether receivers are related, newly observed, or part of a rapid movement pattern."

### **FR2. SHAP-to-reason conversion**

The system must convert top SHAP drivers into investigator-readable explanations.

Required output:

| Column | Description |
| ----- | ----- |
| `alert_id` | Alert identifier. |
| `case_id` | Case identifier if available. |
| `reason_rank` | Rank of reason. |
| `feature_name` | Source feature. |
| `contribution` | SHAP contribution. |
| `risk_direction` | Increasing, reducing, or contextual. |
| `business_reason_code` | Business reason. |
| `plain_language_reason` | Full explanation. |
| `investigator_note` | Suggested review angle. |

### **FR3. Executive summary page**

The report must start with a one-page executive summary containing:

| Section | Content |
| ----- | ----- |
| Overall decision | Keep MVP champion / promote extended / continue challenger. |
| Key evidence | Top-K comparison, alert volume, P1 size, graph/case findings. |
| Risk implication | Whether extended model is safe for prioritisation. |
| Operational implication | Expected queue size and investigator workload. |
| Required remediation | Remaining items before promotion. |
| Next approval gate | Conditions for model promotion. |

### **FR4. Champion-challenger decision logic**

Recommended decision rule:

Promote extended model only if:

    Precision@1000 improves by \>= 5% relative uplift

AND Recall@1000 does not decrease

AND PR-AUC improves

AND P1 band size is within operational capacity

AND case consolidation has no mega-case issue

AND reason codes are generated for P1 alerts

Decision output:

{

  "champion\_model": "mvp",

  "challenger\_model": "extended",

  "decision": "keep\_mvp",

  "primary\_reason": "Extended model does not improve Precision@1000",

  "promotion\_ready": false,

  "failed\_gates": \[

    "precision\_at\_1000",

    "p1\_band\_size",

    "case\_consolidation"

  \]

}

### **FR5. Traffic-light dashboard**

Required traffic-light statuses:

| Area | Green condition | Amber condition | Red condition |
| ----- | ----- | ----- | ----- |
| Alert volume | Alert rate within target | Slightly above target | Excessive alert rate |
| P1 sizing | Within daily/weekly capacity | Slightly above capacity | Operationally impossible |
| Top-K uplift | Extended beats MVP | Mixed results | Extended underperforms |
| Graph features | Positive ablation | Neutral | Negative or noisy |
| Cycle detection | Captures labels with low volume | Low evidence | High volume and no labels |
| Case consolidation | No mega-cases | Some large cases | Mega-case issue |
| Calibration | Monotonic bands | Minor non-monotonicity | Unusable bands |
| Explainability | Business reason codes complete | Partial | Technical only |
| Report automation | One-command complete | Minor manual steps | Broken or incomplete |

---

## **4.5 Non-functional requirements**

| Requirement | Description |
| ----- | ----- |
| Audience readiness | Report must be readable by AML, analytics, and management. |
| Traceability | Every executive conclusion must link to supporting metric. |
| Consistency | HTML report must reconcile with CSV/JSON artifacts. |
| Portability | HTML report must open locally without external dependencies. |

---

## **4.6 Acceptance criteria**

Sprint 4 is complete when:

| Criterion | Minimum standard |
| ----- | ----- |
| Reason-code dictionary | Covers all top 30 model features and all rule flags. |
| P1 reason codes | 100% of P1 alerts have at least one business reason. |
| Executive summary | Appears on first report page. |
| Champion decision | Explicit promote/keep/challenger decision is stated. |
| Traffic lights | All major modules have red/amber/green status. |
| Final report | HTML report is standalone and generated from artifacts. |

## **4.7 Implementation details**

### **Modules and commands**

Implement Sprint 4 in:

| Component | Detail |
| ----- | ----- |
| Reason dictionary loader | `src/aml_mvp/explainability/reason_code_dictionary.py` |
| SHAP reason mapper | `src/aml_mvp/explainability/shap_reason_mapper.py` |
| Traffic lights | `src/aml_mvp/diagnostics/traffic_lights.py` |
| Executive summary | `src/aml_mvp/reporting/executive_summary.py` |
| Remediation report | `src/aml_mvp/reporting/remediation_report.py` |
| CLI | Add `generate-reason-codes` and `build-remediation-report`. |
| Pixi tasks | Add `generate-reason-codes`, `build-remediation-report`, and `run-remediation`. |

### **Config: `config/reason_code_dictionary.yaml`**

Required structure:

```yaml
defaults:
  unknown_feature:
    business_reason_code: "Model feature contributed to risk score"
    risk_direction: contextual
    typology_mapping: unknown
    plain_language_template: "A model feature contributed to the alert score."
    investigator_note: "Review transaction context and linked rule evidence."

features:
  feature_sender_prior_unique_receivers:
    business_reason_code: "Sender distributed funds to many counterparties"
    risk_direction: risk_increasing
    typology_mapping: fan-out / layering
    plain_language_template: "The sender has previously transferred funds to many unique receivers, which may indicate distribution or layering behaviour."
    investigator_note: "Review whether receivers are related, newly observed, or part of rapid movement."

rule_flags:
  feature_rule_r4_pass_through_flag:
    business_reason_code: "Rapid pass-through activity"
    risk_direction: risk_increasing
    typology_mapping: pass-through
    plain_language_template: "Funds appear to move through the account shortly after receipt."
    investigator_note: "Compare incoming and outgoing timing, amounts, and counterparties."
```

### **Config: remediation report additions**

Add to `config/report_config.yaml` or a new `config/remediation_report_config.yaml`:

```yaml
remediation_report:
  output_file: outputs/reports/aml_extended_remediation_report.html
  include_sections:
    - executive_summary
    - alert_band_sizing
    - graph_diagnostics
    - case_consolidation
    - champion_challenger
    - explainability
    - traffic_lights
```

### **Champion decision output**

`outputs/metrics/champion_challenger_decision.json` must include:

| Field | Description |
| ----- | ----- |
| `champion_model` | Current production-like champion, default `mvp`. |
| `challenger_model` | Default `extended`. |
| `decision` | `keep_mvp`, `promote_extended`, or `continue_challenger`. |
| `promotion_ready` | Boolean. |
| `primary_reason` | Human-readable decision reason. |
| `failed_gates` | List of failed promotion gates. |
| `passed_gates` | List of passed promotion gates. |
| `supporting_metrics` | Key values used in the decision. |

### **Traffic-light thresholds**

Use these defaults:

| Area | Green | Amber | Red |
| ----- | ----- | ----- | ----- |
| Alert volume | Alert rate <= 5% | >5% and <=15% | >15% |
| P1 sizing | P1 daily alerts <= capacity | <= 2x capacity | > 2x capacity |
| Top-K uplift | Precision@1000 relative uplift >= 5% | between 0% and 5% | < 0% |
| Graph features | Ablation decision `keep` | `research_only` | `remove` |
| Cycle detection | Label count > 0 and candidate rate <= 10% | Label count > 0 or candidate rate <= 20% | 0 labels with high candidate rate |
| Case consolidation | Max operational alerts <= cap | <= 2x cap | > 2x cap or mega-case present |
| Calibration | P1 precision >= P2 >= P3 >= P4 | One adjacent inversion | Multiple inversions |
| Explainability | 100% P1 reason coverage | 90%-99% | <90% |
| Report automation | All sections rendered | Missing optional artifact | Missing decision or core section |

### **Reason-code behavior**

- Use `outputs/extended/reason_codes.csv` or `data/processed/shap_values.parquet` as SHAP source.
- Join `alert_case_mapping_v2.parquet` when available to populate `case_id`.
- Generate reasons for at least all P1 alerts in `priority_alerts_v2.parquet`; if P1 v2 is unavailable, use current priority artifacts.
- Unknown features must use the dictionary default and must not break report generation.

### **Sprint 4 tests**

Add:

| Test | Required assertions |
| ----- | ----- |
| `tests/test_reason_codes.py` | Known and unknown features map to business-readable reason rows. |
| `tests/test_traffic_lights.py` | Red/amber/green statuses follow default thresholds. |
| `tests/test_champion_decision.py` | Promotion gates pass/fail correctly. |
| `tests/test_remediation_report.py` | Final report renders with executive summary and no unresolved placeholders. |

Run:

```bash
python -m pytest tests/test_reason_codes.py tests/test_traffic_lights.py tests/test_champion_decision.py tests/test_remediation_report.py
pixi run generate-reason-codes
pixi run build-remediation-report
```

---

# **Cross-Sprint Technical Design**

## **5.1 Updated folder structure**

```text
src/aml_mvp/
  diagnostics/
    __init__.py
    rule_contribution.py
    alert_waterfall.py
    traffic_lights.py
  graph/
    graph_builder.py
    graph_features_v2.py
    gather_scatter_v2.py
    cycle_detection_v2.py
    graph_ablation.py
  cases/
    case_consolidation_v2.py
    mega_component_splitter.py
    case_typology.py
    case_metrics.py
  explainability/
    shap_reason_mapper.py
    reason_code_dictionary.py
  calibration/
    band_sizing.py
    capacity_simulation.py
  reporting/
    executive_summary.py
    remediation_report.py
    report_sections/
      alert_band_sizing.html.j2
      graph_diagnostics.html.j2
      case_consolidation.html.j2
      explainability.html.j2
      champion_challenger.html.j2
```

Each new module must expose one top-level function used by the CLI. That keeps the CLI thin and gives tests a stable import target.

| Module | Public function |
| ----- | ----- |
| `diagnostics/rule_contribution.py` | `build_rule_contribution(config: dict, root: Path, logger: logging.Logger) -> pd.DataFrame` |
| `diagnostics/alert_waterfall.py` | `build_alert_waterfall(config: dict, root: Path, logger: logging.Logger) -> dict` |
| `calibration/band_sizing.py` | `rebuild_priority_bands(config: dict, root: Path, logger: logging.Logger) -> pd.DataFrame` |
| `calibration/capacity_simulation.py` | `run_capacity_simulation(config: dict, priority_alerts: pd.DataFrame) -> pd.DataFrame` |
| `graph/graph_features_v2.py` | `build_graph_features_v2(config: dict, root: Path, logger: logging.Logger) -> pd.DataFrame` |
| `graph/gather_scatter_v2.py` | `detect_gather_scatter_v2(transactions: pd.DataFrame, config: dict) -> pd.DataFrame` |
| `graph/cycle_detection_v2.py` | `detect_cycles_v2(transactions: pd.DataFrame, config: dict) -> pd.DataFrame` |
| `graph/graph_ablation.py` | `run_graph_ablation(config: dict, root: Path, logger: logging.Logger) -> pd.DataFrame` |
| `cases/case_consolidation_v2.py` | `consolidate_cases_v2(config: dict, root: Path, logger: logging.Logger) -> dict` |
| `cases/mega_component_splitter.py` | `split_mega_components(alert_links: pd.DataFrame, config: dict) -> dict` |
| `cases/case_typology.py` | `assign_case_typology(case_alerts: pd.DataFrame) -> pd.Series` |
| `cases/case_metrics.py` | `compute_case_metrics(cases: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame` |
| `explainability/reason_code_dictionary.py` | `load_reason_code_dictionary(path: Path) -> dict` |
| `explainability/shap_reason_mapper.py` | `generate_alert_reason_codes(config: dict, root: Path, logger: logging.Logger) -> pd.DataFrame` |
| `diagnostics/traffic_lights.py` | `build_traffic_light_findings(config: dict, root: Path) -> pd.DataFrame` |
| `reporting/remediation_report.py` | `build_remediation_report(config: dict, root: Path, logger: logging.Logger) -> Path` |

---

## **5.2 New CLI commands and Pixi tasks**

All remediation commands must support the existing logging flags:

```bash
--log-level INFO|DEBUG|WARNING
--log-file outputs/run_logs/custom.log
```

### **Direct Python commands**

```bash
# Sprint 1
python -m aml_mvp.cli diagnose-alerts --config config/priority_band_config.yaml
python -m aml_mvp.cli rebuild-priority-bands --config config/priority_band_config.yaml

# Sprint 2
python -m aml_mvp.cli build-graph-v2 --config config/graph_rule_config.yaml
python -m aml_mvp.cli run-graph-ablation --config config/graph_rule_config.yaml

# Sprint 3
python -m aml_mvp.cli consolidate-cases-v2 --config config/case_consolidation_config.yaml

# Sprint 4
python -m aml_mvp.cli generate-reason-codes --config config/reason_code_dictionary.yaml
python -m aml_mvp.cli build-remediation-report --config config/report_config.yaml
```

### **Pixi task contract**

Add these tasks to `pixi.toml`:

```toml
diagnose-alerts = "python -m aml_mvp.cli diagnose-alerts --config config/priority_band_config.yaml"
rebuild-priority-bands = "python -m aml_mvp.cli rebuild-priority-bands --config config/priority_band_config.yaml"
build-graph-v2 = "python -m aml_mvp.cli build-graph-v2 --config config/graph_rule_config.yaml"
run-graph-ablation = "python -m aml_mvp.cli run-graph-ablation --config config/graph_rule_config.yaml"
consolidate-cases-v2 = "python -m aml_mvp.cli consolidate-cases-v2 --config config/case_consolidation_config.yaml"
generate-reason-codes = "python -m aml_mvp.cli generate-reason-codes --config config/reason_code_dictionary.yaml"
build-remediation-report = "python -m aml_mvp.cli build-remediation-report --config config/report_config.yaml"
run-remediation = "python -m aml_mvp.cli run-remediation --config config/remediation_workflow.yaml"
```

The wrapper command `run-remediation` must allow selective execution:

```bash
pixi run run-remediation
python -m aml_mvp.cli run-remediation --config config/remediation_workflow.yaml --steps diagnose-alerts,rebuild-priority-bands
python -m aml_mvp.cli run-remediation --config config/remediation_workflow.yaml --skip build-graph-v2,run-graph-ablation
python -m aml_mvp.cli run-remediation --config config/remediation_workflow.yaml --from-step consolidate-cases-v2
```

The wrapper should fail fast by default. If `continue_on_error: true` is set in config, it should record failed steps in `outputs/metrics/remediation_workflow_status.json` and continue to later independent report steps only when their required inputs exist.

---

## **5.3 Regression test suite**

| Test file | Purpose |
| ----- | ----- |
| `tests/test_rule_contribution.py` | Validate rule contribution and overlap logic. |
| `tests/test_priority_band_sizing.py` | Validate P1/P2/P3/P4 definitions and overrides. |
| `tests/test_capacity_simulation.py` | Validate top-K and daily capacity metrics. |
| `tests/test_graph_rules_v2.py` | Validate gather-scatter and cycle detection on toy graphs. |
| `tests/test_graph_ablation.py` | Validate model variant comparison output. |
| `tests/test_case_consolidation_v2.py` | Validate case caps and mega-component splitting. |
| `tests/test_case_typology.py` | Validate case typology assignment. |
| `tests/test_reason_codes.py` | Validate SHAP-to-business-reason mapping. |
| `tests/test_traffic_lights.py` | Validate red/amber/green status logic. |
| `tests/test_remediation_report.py` | Validate final HTML report rendering. |

---

## **5.4 Remediation workflow config**

Create `config/remediation_workflow.yaml`:

```yaml
workflow:
  name: remediation
  continue_on_error: false
  status_path: outputs/metrics/remediation_workflow_status.json
  default_steps:
    - diagnose-alerts
    - rebuild-priority-bands
    - build-graph-v2
    - run-graph-ablation
    - consolidate-cases-v2
    - generate-reason-codes
    - build-remediation-report

steps:
  diagnose-alerts:
    command: diagnose-alerts
    config: config/priority_band_config.yaml
    required_inputs:
      - data/processed/rule_hits.parquet
      - data/processed/alerts.parquet
    expected_outputs:
      - outputs/metrics/rule_alert_contribution.csv
      - outputs/charts/rule_alert_waterfall.png

  rebuild-priority-bands:
    command: rebuild-priority-bands
    config: config/priority_band_config.yaml
    required_inputs:
      - data/processed/alerts.parquet
    expected_outputs:
      - data/processed/priority_alerts_v2.parquet
      - outputs/metrics/priority_band_summary.csv
      - outputs/metrics/daily_capacity_simulation.csv

  build-graph-v2:
    command: build-graph-v2
    config: config/graph_rule_config.yaml
    required_inputs:
      - data/processed/base_transactions_with_splits.parquet
    expected_outputs:
      - data/processed/graph_features_v2.parquet
      - data/processed/graph_rule_hits_v2.parquet
      - outputs/metrics/cycle_candidates_v2.csv

  run-graph-ablation:
    command: run-graph-ablation
    config: config/graph_rule_config.yaml
    required_inputs:
      - data/processed/alert_features.parquet
      - data/processed/graph_features_v2.parquet
    expected_outputs:
      - outputs/metrics/graph_ablation_results.csv

  consolidate-cases-v2:
    command: consolidate-cases-v2
    config: config/case_consolidation_config.yaml
    required_inputs:
      - data/processed/alerts.parquet
      - data/processed/rule_hits.parquet
    expected_outputs:
      - data/processed/consolidated_cases_v2.parquet
      - data/processed/alert_case_mapping_v2.parquet
      - data/processed/network_clusters_v2.parquet
      - outputs/metrics/case_quality_metrics.csv

  generate-reason-codes:
    command: generate-reason-codes
    config: config/reason_code_dictionary.yaml
    required_inputs:
      - data/processed/priority_alerts_v2.parquet
    expected_outputs:
      - outputs/metrics/alert_reason_codes.csv

  build-remediation-report:
    command: build-remediation-report
    config: config/report_config.yaml
    required_inputs:
      - outputs/metrics/champion_challenger_decision.json
    expected_outputs:
      - outputs/reports/aml_extended_remediation_report.html
```

The wrapper should resolve paths relative to project root and should write elapsed seconds, status, and artifact existence checks for each step.

---

## **5.5 Dependency order and implementation gates**

Implementation should proceed in this order:

| Order | Work item | Required before starting | Exit gate |
| ----- | ----- | ----- | ----- |
| 1 | Add config files and CLI stubs | Current tests pass | `python -m aml_mvp.cli <new-command> --help` works for every command. |
| 2 | Sprint 1 diagnostics and band sizing | Existing rule and score artifacts | Sprint 1 tests pass and `priority_alerts_v2.parquet` is generated. |
| 3 | Sprint 2 graph v2 and ablation | Base transactions and feature matrix | Graph v2 tests pass and ablation output has variants A-E. |
| 4 | Sprint 3 case v2 | Alert table and graph v2 hits | Every alert maps exactly once and no operational case violates configured cap. |
| 5 | Sprint 4 reasons and report | Priority v2, case v2, comparison metrics | Final report renders with executive summary, decision, and traffic lights. |
| 6 | Workflow wrapper | All individual commands implemented | `pixi run run-remediation` runs end to end and status JSON marks all steps complete. |

Do not update champion promotion logic until Sprint 1 band outputs and Sprint 2 ablation outputs exist. Otherwise the decision file will encode incomplete evidence.

---

## **5.6 Artifact compatibility and report integration**

The remediation pipeline must keep the existing MVP and extended artifacts unchanged. New report sections should be optional readers:

| Report | Required behavior |
| ----- | ----- |
| `build-report` | Continue rendering MVP report. If Sprint 1 outputs exist, include a compact band-sizing appendix. |
| `build-extended-report` | Continue rendering the current extended report. If v2 graph, case, reason, or decision outputs exist, include a compact remediation status table. |
| `build-remediation-report` | Require the remediation decision artifacts and render the full executive remediation report. |

When an optional artifact is missing, the report should show `not_generated` in a status table rather than failing with a traceback. Core artifacts for `build-remediation-report` are `champion_challenger_decision.json`, `traffic_light_findings.csv`, and `priority_band_summary.csv`.

---

## **5.7 Logging and runtime expectations**

Every new command must use the existing logging utility and log:

| Event | Required log fields |
| ----- | ----- |
| Command start | command name, config path, project root, log file |
| Input load | artifact path, row count, columns used |
| Major stage start/end | stage name, elapsed seconds |
| Output write | artifact path, row count or file size |
| Command end | total elapsed seconds, success or failure |

Long-running graph and case stages must log chunk progress at `INFO` every 100,000 rows or every completed window, whichever is easier to implement. Per-group detail remains `DEBUG`.

---

## **5.8 Open implementation questions**

No blocking questions remain for implementation. The following decisions are assumptions that can be changed later through config:

| Topic | Current assumption |
| ----- | ----- |
| Capacity | 3 investigators, 40 alerts per investigator per day. |
| Promotion gate | Extended model needs at least 5% relative Precision@1000 uplift and no recall decrease. |
| Graph features | Forced graph features are disabled in remediation unless ablation supports them. |
| Mega-components | Components above 1,000 accounts or alerts become network intelligence objects, not operational cases. |
| Reason-code coverage | Unknown model features use a default reason-code template instead of blocking report generation. |

---

# **Implementation Schedule**

## **Sprint 1: 1 week**

| Day | Activity | Output |
| ----- | ----- | ----- |
| Day 1 | Build rule contribution diagnostics | Rule contribution CSV |
| Day 2 | Build alert waterfall and overlap analysis | Waterfall chart |
| Day 3 | Redesign P1/P2/P3/P4 config | New band config |
| Day 4 | Run capacity simulation | Capacity simulation CSV |
| Day 5 | Re-run top-K and update report section | Sprint 1 report section |

---

## **Sprint 2: 1 to 1.5 weeks**

| Day | Activity | Output |
| ----- | ----- | ----- |
| Day 1 | Implement graph snapshot windows | Graph feature base |
| Day 2 | Implement gather-scatter v2 | Graph rule hits |
| Day 3 | Implement cycle v2 | Cycle candidates |
| Day 4 | Run graph ablation | Ablation results |
| Day 5 | Remove invalid forced graph features | Updated feature set |
| Day 6–7 | Optimise runtime and finalise diagnostics | Sprint 2 report section |

---

## **Sprint 3: 1 week**

| Day | Activity | Output |
| ----- | ----- | ----- |
| Day 1 | Implement case linkage hierarchy | Case logic |
| Day 2 | Implement case caps | Case table v2 |
| Day 3 | Implement mega-component split | Mega-case diagnostics |
| Day 4 | Implement case typology labels | Case typology summary |
| Day 5 | Compute case-level metrics | Sprint 3 report section |

---

## **Sprint 4: 1 week**

| Day | Activity | Output |
| ----- | ----- | ----- |
| Day 1 | Build reason-code dictionary | YAML dictionary |
| Day 2 | Map SHAP to AML business reasons | Reason-code CSV |
| Day 3 | Build executive summary and champion decision | Decision JSON |
| Day 4 | Build traffic-light dashboard | Traffic-light table |
| Day 5 | Generate final remediation HTML report | Final HTML report |

---

# **Final Definition of Done**

The remediation cycle is complete when:

| Area | Done condition |
| ----- | ----- |
| Alert sizing | Alert and P1 volumes are within explicit target ranges or explained as research mode. |
| Rule diagnostics | Rule contribution waterfall identifies volume drivers. |
| Graph logic | Gather-scatter and cycle logic produce validated signals or are downgraded to research-only. |
| Case consolidation | No operational case contains excessive alerts; mega-components are split or flagged separately. |
| Model comparison | MVP versus extended decision is rerun after remediation. |
| Explainability | Business reason codes are available for all P1 alerts. |
| Reporting | HTML report begins with executive decision page and traffic-light findings. |
| Governance | No model-based auto-closure or suppression is introduced. |
| Reproducibility | All outputs are generated through CLI and config-driven pipeline. |

Final promotion of the extended model should require clear evidence that it improves **Precision@K and Recall@K at realistic investigation capacity** without creating operationally unmanageable alert or case volumes.
