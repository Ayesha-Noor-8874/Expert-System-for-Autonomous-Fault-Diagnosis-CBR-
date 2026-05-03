import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

FILE_PATH  = r"C:\Users\ALLAH\Downloads\industrial_fault_detection_data_1000.csv"
REPORT_DIR = "cbr_reports"

Features        = []
Features_Weight = {}   # fixed typo: was Feautres_Weight
K = 3

Fault_Map = {
    0: "Normal Operation (No Fault)",
    1: "Mechanical Vibration / Bearing Issue",
    2: "System Overheating / Thermal Stress",
}

Rules = {}


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_case_library(path):
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    print(f"[SYSTEM] Loaded {len(df)} cases from: {path}")
    return df


def setup_features(library):
    global Features, Features_Weight, Rules

    cols = library.columns.tolist()

    vib_col  = next(c for c in cols if 'Vibration' in c and '(' in c)
    temp_col = next(c for c in cols if 'Temp'      in c and '(' in c)
    pres_col = next(c for c in cols if 'Pressure'  in c and '(' in c)

    Features = [vib_col, temp_col, pres_col]

    # Vibration is most diagnostic, temperature second, pressure least
    Features_Weight = {
        vib_col:  2.0,
        temp_col: 1.5,
        pres_col: 1.0,
    }

    Rules = {
        'high_temp':      {'feature': temp_col, 'threshold': 130, 'op': '>', 'label': 2},
        'high_vibration': {'feature': vib_col,  'threshold': 2.5, 'op': '>', 'label': 1},
        'low_vibration':  {'feature': vib_col,  'threshold': 0.3, 'op': '<', 'label': 0},
    }

    print(f"[SYSTEM] Features detected : {Features}")
    print(f"[SYSTEM] Weights           : {Features_Weight}")


# ─────────────────────────────────────────────
#  NORMALIZATION
# ─────────────────────────────────────────────

def compute_normalization_params(library):
    params = {}
    for feature in Features:
        params[feature] = {
            'min': library[feature].min(),
            'max': library[feature].max(),
        }
    return params


def normalize(values_dict, norm_params):
    normalized = []
    for feature in Features:
        mn  = norm_params[feature]['min']
        mx  = norm_params[feature]['max']
        val = values_dict[feature]
        normalized.append((val - mn) / (mx - mn) if mx != mn else 0.0)
    return np.array(normalized)


def normalize_library(library, norm_params):
    norm_rows = []
    for _, row in library.iterrows():
        values = {f: row[f] for f in Features}
        norm_rows.append(normalize(values, norm_params))
    return np.array(norm_rows)


# ─────────────────────────────────────────────
#  ANOMALY DETECTION
#  Runs BEFORE retrieve so CBR is never misled
#  by readings outside the training range.
# ─────────────────────────────────────────────

def detect_anomaly(query_values, norm_params):
    anomalies = []

    for feature in Features:
        val = query_values[feature]
        mn  = norm_params[feature]['min']
        mx  = norm_params[feature]['max']

        if val < mn:
            pct = ((mn - val) / mn * 100) if mn != 0 else float('inf')
            anomalies.append({
                'feature':   feature,
                'value':     val,
                'min':       mn,
                'max':       mx,
                'deviation': f"{pct:.1f}% below minimum",
                'severity':  'HIGH' if pct > 50 else 'MEDIUM',
            })
        elif val > mx:
            pct = ((val - mx) / mx * 100) if mx != 0 else float('inf')
            anomalies.append({
                'feature':   feature,
                'value':     val,
                'min':       mn,
                'max':       mx,
                'deviation': f"{pct:.1f}% above maximum",
                'severity':  'HIGH' if pct > 50 else 'MEDIUM',
            })

    if not anomalies:
        return False, "All sensors within normal range.", None

    lines = [
        "",
        "!" * 52,
        "  ANOMALY DETECTED",
        "  Sensor readings outside training range!",
        "!" * 52,
    ]
    for a in anomalies:
        lines += [
            f"  * {a['feature']}",
            f"    Value    : {a['value']:.3f}",
            f"    Range    : [{a['min']:.3f}, {a['max']:.3f}]",
            f"    Deviation: {a['deviation']}",
            f"    Severity : {a['severity']}",
        ]
    lines += [
        "!" * 52,
        "  RECOMMENDATION: Manual inspection required.",
        "  CBR results may be unreliable for this input.",
        "!" * 52,
        "",
    ]
    return True, '\n'.join(lines), anomalies


# ─────────────────────────────────────────────
#  STAGE 1 — RETRIEVE
# ─────────────────────────────────────────────

def retrieve_top_k(library, norm_library, query_values, norm_params, k=K):
    weights    = np.array([Features_Weight[f] for f in Features])
    query_norm = normalize(query_values, norm_params)
    diff       = norm_library - query_norm
    distances  = np.sqrt((diff ** 2 * weights).sum(axis=1))

    top_k_indices = np.argsort(distances)[:k]
    top_cases     = library.iloc[top_k_indices].copy()
    top_distances = distances[top_k_indices]

    confidence = 1.0 / (1.0 + top_distances[0])
    return top_cases, top_distances, confidence


# ─────────────────────────────────────────────
#  STAGE 2 — REUSE
# ─────────────────────────────────────────────

def reuse_majority_vote(top_cases):
    votes = top_cases['Fault Label'].astype(int).value_counts()
    return int(votes.idxmax())


# ─────────────────────────────────────────────
#  STAGE 3 — REVISE
# ─────────────────────────────────────────────

def revise(suggested_label, query_values):
    for rule_name, rule in Rules.items():
        val       = query_values[rule['feature']]
        threshold = rule['threshold']
        op        = rule['op']
        triggered = (op == '>' and val > threshold) or \
                    (op == '<' and val < threshold)
        if triggered:
            return rule['label'], True, rule_name
    return suggested_label, False, None


# ─────────────────────────────────────────────
#  STAGE 4 — RETAIN
# ─────────────────────────────────────────────

def retain_new_case(library, query_values, fault_label, path, tolerance=0.01):
    norm_params = compute_normalization_params(library)
    norm_lib    = normalize_library(library, norm_params)
    query_norm  = normalize(query_values, norm_params)
    distances   = np.linalg.norm(norm_lib - query_norm, axis=1)

    if distances.min() < tolerance:
        print("[RETAIN] Case too similar to existing entry — skipped.")
        return library, False

    new_row = {'Timestamp': pd.Timestamp.now(), 'Fault Label': fault_label}
    new_row.update(query_values)

    updated = pd.concat([library, pd.DataFrame([new_row])], ignore_index=True)
    updated.to_csv(path, index=False)
    print(f"[RETAIN] New case saved. Library now has {len(updated)} cases.")
    return updated, True


# ─────────────────────────────────────────────
#  GENERATE REPORT
#  Saves a full audit trail (.txt) for every
#  diagnosis cycle.  Called at end of diagnose().
# ─────────────────────────────────────────────

def generate_report(query_values, top_cases, distances,
                    suggested_label, final_label,
                    was_revised, rule_name,
                    confidence, was_retained,
                    is_anomaly=False, anomaly_report=None):

    os.makedirs(REPORT_DIR, exist_ok=True)

    ts       = datetime.now()
    filename = f"cbr_report_{ts.strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(REPORT_DIR, filename)

    severity = {
        0: "LOW    — Normal operation",
        1: "MEDIUM — Mechanical fault detected",
        2: "HIGH   — Thermal fault detected",
    }.get(final_label, "UNKNOWN")

    actions = {
        0: "No action required. Continue normal monitoring.",
        1: "Inspect bearings and mechanical components. Schedule maintenance.",
        2: "Check cooling system immediately. Reduce load if temperature persists.",
    }.get(final_label, "Consult engineer.")

    lines = []

    def section(title):
        lines.extend(["", "=" * 56, f"  {title}", "=" * 56])

    def row(label, value):
        lines.append(f"  {label:<28} {value}")

    lines += ["=" * 56,
              "  CBR EXPERT SYSTEM — FAULT DIAGNOSIS REPORT",
              "=" * 56]
    row("Generated",   ts.strftime("%Y-%m-%d  %H:%M:%S"))
    row("Report file", filename)

    if is_anomaly:
        lines += ["", "!" * 56,
                  "  ANOMALY DETECTED — RESULTS MAY BE UNRELIABLE",
                  "!" * 56]
        if anomaly_report:
            lines.append(anomaly_report)

    section("STAGE 1 — SENSOR READINGS (Query)")
    for feature, value in query_values.items():
        row(feature, f"{value:.4f}")

    section("STAGE 2 — RETRIEVE  (top-k similar cases)")
    row("k neighbours", str(len(top_cases)))
    lines.append("")
    for i, (_, case_row) in enumerate(top_cases.iterrows()):
        label   = int(case_row['Fault Label'])
        tag     = "CLOSEST MATCH" if i == 0 else f"Match {i + 1}"
        lines.append(f"  [{tag}]")
        row("  Timestamp",   str(case_row['Timestamp'])[:19])
        row("  Distance",    f"{distances[i]:.6f}")
        row("  Fault label", f"{label}  —  {Fault_Map.get(label, 'Unknown')}")
        lines.append("")

    section("STAGE 3 — REUSE  (majority vote)")
    row("Suggested label", f"{suggested_label}  —  {Fault_Map.get(suggested_label, 'Unknown')}")
    row("Confidence",      f"{confidence * 100:.1f}%")
    if confidence < 0.4:
        row("WARNING", "Low confidence — manual inspection advised")

    section("STAGE 4 — REVISE  (rule-based check)")
    if was_revised:
        row("Rule triggered", rule_name)
        row("CBR suggested",  f"{suggested_label}  —  {Fault_Map.get(suggested_label, 'Unknown')}")
        row("Revised to",     f"{final_label}  —  {Fault_Map.get(final_label, 'Unknown')}")
    else:
        row("Rule triggered", "None — CBR result accepted")

    section("FINAL DIAGNOSIS")
    row("Fault label", str(final_label))
    row("Condition",   Fault_Map.get(final_label, "Unknown"))
    row("Severity",    severity)
    row("Confidence",  f"{confidence * 100:.1f}%")
    row("Method",      "Rule override" if was_revised else "CBR majority vote")
    if is_anomaly:
        row("CAUTION", "Anomalous input — diagnosis may be invalid")

    section("STAGE 5 — RETAIN")
    if is_anomaly:
        row("Status", "NOT SAVED — anomalous case rejected")
        row("Reason", "Input outside training range")
    elif was_retained:
        row("Status",         "Case saved to library")
        row("Label retained", f"{final_label}  —  {Fault_Map.get(final_label, 'Unknown')}")
    else:
        row("Status", "Case NOT saved (too similar to existing entry)")

    section("RECOMMENDED ACTION")
    if is_anomaly:
        lines += ["  IMMEDIATE ACTION REQUIRED:",
                  "    1. Verify sensor calibration",
                  "    2. Inspect equipment manually",
                  "    3. Do not rely on automated diagnosis"]
    else:
        lines.append(f"  {actions}")

    lines += ["", "=" * 56, "  END OF REPORT", "=" * 56, ""]

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[REPORT] Saved -> {filepath}")
    return filepath


# ─────────────────────────────────────────────
#  DIAGNOSE
#  Orchestrates all 4 CBR stages + report.
# ─────────────────────────────────────────────

def diagnose(library, norm_library, norm_params, query_values, retain=False):
    print("\n" + "=" * 52)
    print("  SENSOR READINGS")
    print("=" * 52)
    for f, v in query_values.items():
        print(f"  {f:<28} {v}")

    # Anomaly check before retrieve
    is_anomaly, anomaly_msg, _ = detect_anomaly(query_values, norm_params)
    if is_anomaly:
        print(anomaly_msg)
        print("  Continuing with diagnosis — results may be unreliable.\n")

    # STAGE 1 — RETRIEVE
    top_cases, distances, confidence = retrieve_top_k(
        library, norm_library, query_values, norm_params
    )

    print(f"\n  Top-{K} similar cases retrieved:")
    for i, (_, row) in enumerate(top_cases.iterrows()):
        label = Fault_Map.get(int(row['Fault Label']), 'Unknown')
        print(f"    [{i+1}] dist={distances[i]:.4f}  ->  {label}  ({row['Timestamp']})")

    # STAGE 2 — REUSE
    suggested_label = reuse_majority_vote(top_cases)

    # STAGE 3 — REVISE
    final_label, was_revised, rule_name = revise(suggested_label, query_values)

    print("\n" + "-" * 52)
    if was_revised:
        print(f"  [REVISE] Rule triggered: '{rule_name}'")
        print(f"  CBR suggested : {Fault_Map[suggested_label]}")
    print(f"  DIAGNOSIS     : {Fault_Map[final_label]}")
    print(f"  CONFIDENCE    : {confidence * 100:.1f}%")
    if confidence < 0.4:
        print("  WARNING: Low confidence — consider manual inspection")
    if is_anomaly:
        print("  WARNING: Anomalous input — diagnosis may be invalid")
    print("-" * 52)

    # STAGE 4 — RETAIN (skipped for anomalous inputs)
    was_retained = False
    if retain:
        if is_anomaly:
            print("[RETAIN] Skipped — anomalous case not saved to library.")
        else:
            library, was_retained = retain_new_case(
                library, query_values, final_label, FILE_PATH
            )
            norm_params  = compute_normalization_params(library)
            norm_library = normalize_library(library, norm_params)

    # REPORT — saves full audit trail
    generate_report(
        query_values    = query_values,
        top_cases       = top_cases,
        distances       = distances,
        suggested_label = suggested_label,
        final_label     = final_label,
        was_revised     = was_revised,
        rule_name       = rule_name,
        confidence      = confidence,
        was_retained    = was_retained,
        is_anomaly      = is_anomaly,
        anomaly_report  = anomaly_msg if is_anomaly else None,
    )

    return library, norm_library, norm_params, final_label


# ─────────────────────────────────────────────
#  EVALUATION  (80/20 split)
# ─────────────────────────────────────────────

def evaluate(library):
    print("\n" + "=" * 52)
    print("  EVALUATION MODE  (80/20 split)")
    print("=" * 52)

    shuffled  = library.sample(frac=1, random_state=42).reset_index(drop=True)
    split     = int(len(shuffled) * 0.8)
    train_lib = shuffled.iloc[:split].reset_index(drop=True)
    test_lib  = shuffled.iloc[split:].reset_index(drop=True)

    norm_params = compute_normalization_params(train_lib)
    norm_train  = normalize_library(train_lib, norm_params)

    y_true, y_pred = [], []
    for _, row in test_lib.iterrows():
        query     = {f: row[f] for f in Features}
        top_cases, _, _ = retrieve_top_k(train_lib, norm_train, query, norm_params)
        suggested = reuse_majority_vote(top_cases)
        final, _, _ = revise(suggested, query)
        y_true.append(int(row['Fault Label']))
        y_pred.append(final)

    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred) * 100

    labels = sorted(Fault_Map.keys())
    n  = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    print(f"\n  Test samples : {len(test_lib)}")
    print(f"  Accuracy     : {accuracy:.2f}%\n")
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    header = "              " + "  ".join([f"Pred {l}" for l in labels])
    print(f"  {header}")
    for i, l in enumerate(labels):
        row_str = "  ".join([f"{cm[i][j]:>6}" for j in range(n)])
        print(f"  Actual {l}  :  {row_str}")

    print("\n  Per-class metrics:")
    for i, l in enumerate(labels):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) \
             if (precision + recall) > 0 else 0
        print(f"    Fault {l} ({Fault_Map[l][:30]:<30})"
              f"  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Pred {l}" for l in labels], rotation=30, ha='right')
    ax.set_yticklabels([f"Actual {l}" for l in labels])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i][j]),
                    ha='center', va='center',
                    color='white' if cm[i][j] > cm.max() / 2 else 'black')
    ax.set_title(f"Confusion Matrix  (Accuracy: {accuracy:.1f}%)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\n  [EVAL] Confusion matrix saved as confusion_matrix.png")
    plt.show()
    return accuracy


# ─────────────────────────────────────────────
#  FAST EVALUATION  (no plots, used internally)
# ─────────────────────────────────────────────

def evaluate_fast(library, k_override=None):
    shuffled  = library.sample(frac=1, random_state=42).reset_index(drop=True)
    split     = int(len(shuffled) * 0.8)
    train_lib = shuffled.iloc[:split].reset_index(drop=True)
    test_lib  = shuffled.iloc[split:].reset_index(drop=True)

    norm_params = compute_normalization_params(train_lib)
    norm_train  = normalize_library(train_lib, norm_params)
    k_use = k_override if k_override is not None else K

    y_true, y_pred = [], []
    for _, row in test_lib.iterrows():
        query     = {f: row[f] for f in Features}
        top_cases, _, _ = retrieve_top_k(train_lib, norm_train, query, norm_params, k=k_use)
        suggested = reuse_majority_vote(top_cases)
        final, _, _ = revise(suggested, query)
        y_true.append(int(row['Fault Label']))
        y_pred.append(final)

    return np.mean(np.array(y_true) == np.array(y_pred)) * 100


# ─────────────────────────────────────────────
#  FIND OPTIMAL K
# ─────────────────────────────────────────────

def find_optimal_k(library, max_k=10):
    print("\n" + "=" * 70)
    print("  OPTIMAL K VALUE ANALYSIS")
    print(f"  Testing k=1 through k={max_k}")
    print("=" * 70)

    results = {}
    for k in range(1, max_k + 1):
        print(f"\n  Testing k={k}...")
        acc = evaluate_fast(library, k_override=k)
        results[k] = acc
        print(f"    -> Accuracy: {acc:.2f}%")

    best_k   = max(results, key=results.get)
    best_acc = results[best_k]

    print("\n" + "-" * 70)
    print("  SUMMARY")
    print("-" * 70)
    for k in range(1, max_k + 1):
        marker = " * BEST" if k == best_k else ""
        print(f"  k={k:<3}  ->  Accuracy: {results[k]:6.2f}%{marker}")
    print("-" * 70)
    print(f"\n  Best K: {best_k}  (Accuracy: {best_acc:.2f}%)")
    print(f"  Current K: {K}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(results.keys()), list(results.values()),
            marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.scatter(best_k, best_acc, color='red', s=150, zorder=5,
               label=f'Best k={best_k} ({best_acc:.1f}%)')
    for k, acc in results.items():
        ax.annotate(f'{acc:.1f}%', (k, acc),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9)
    ax.set_xlabel('K Value (Number of Neighbours)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Optimal K Value Selection', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, max_k + 1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("optimal_k_analysis.png", dpi=150)
    print("  Plot saved as optimal_k_analysis.png")
    plt.show()

    return best_k, results


# ─────────────────────────────────────────────
#  COMPARE SIMILARITY METRICS
#  Safe version: never overwrites global
#  retrieve_top_k — uses a local closure instead.
# ─────────────────────────────────────────────

def compare_similarity_metrics(library):
    print("\n" + "=" * 70)
    print("  SIMILARITY METRICS COMPARISON")
    print("  Euclidean vs Cosine vs Manhattan")
    print("=" * 70)

    # Distance functions — all accept (vec_a, vec_b, weights)
    def euclidean(a, b, w):
        return np.sqrt(((a - b) ** 2 * w).sum())

    def cosine(a, b, w):
        aw, bw = a * np.sqrt(w), b * np.sqrt(w)
        denom  = np.linalg.norm(aw) * np.linalg.norm(bw)
        return 1.0 - (np.dot(aw, bw) / denom) if denom > 1e-9 else 1.0

    def manhattan(a, b, w):
        return np.sum(np.abs(a - b) * w)

    metrics = {'Euclidean': euclidean, 'Cosine': cosine, 'Manhattan': manhattan}

    # One self-contained evaluation per metric — never touches global functions
    def run_metric(distance_func):
        shuffled  = library.sample(frac=1, random_state=42).reset_index(drop=True)
        split     = int(len(shuffled) * 0.8)
        train_lib = shuffled.iloc[:split].reset_index(drop=True)
        test_lib  = shuffled.iloc[split:].reset_index(drop=True)
        norm_p    = compute_normalization_params(train_lib)
        norm_tr   = normalize_library(train_lib, norm_p)
        weights   = np.array([Features_Weight[f] for f in Features])

        y_true, y_pred = [], []
        for _, row in test_lib.iterrows():
            query      = {f: row[f] for f in Features}
            q_norm     = normalize(query, norm_p)
            dists      = np.array([distance_func(q_norm, c, weights) for c in norm_tr])
            top_idx    = np.argsort(dists)[:K]
            top_cases  = train_lib.iloc[top_idx]
            suggested  = reuse_majority_vote(top_cases)
            final, _, _ = revise(suggested, query)
            y_true.append(int(row['Fault Label']))
            y_pred.append(final)

        return np.mean(np.array(y_true) == np.array(y_pred)) * 100

    results = {}
    for name, func in metrics.items():
        print(f"\n  Testing {name}...")
        results[name] = run_metric(func)
        print(f"    -> Accuracy: {results[name]:.2f}%")

    best_metric = max(results, key=results.get)

    print("\n" + "-" * 70)
    print(f"  {'Metric':<15} {'Accuracy':>10}   Rank")
    print("-" * 70)
    for rank, (m, a) in enumerate(sorted(results.items(), key=lambda x: -x[1]), 1):
        print(f"  {m:<15} {a:>9.2f}%   #{rank}")
    print("-" * 70)
    print(f"\n  Best metric: {best_metric}  ({results[best_metric]:.2f}%)")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = ['#2ecc71' if m == best_metric else '#95a5a6' for m in results]
    bars    = ax.bar(list(results.keys()), list(results.values()),
                     color=colors, edgecolor='black', linewidth=1)
    for bar, acc in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Similarity Metric', fontsize=12)
    ax.set_title('Similarity Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("similarity_metrics_comparison.png", dpi=150)
    print("  Plot saved as similarity_metrics_comparison.png")
    plt.show()

    return results, best_metric


# ─────────────────────────────────────────────
#  COMPREHENSIVE ANALYSIS
#  Runs find_optimal_k + compare_similarity_metrics
#  and writes a summary report.
# ─────────────────────────────────────────────

def run_comprehensive_analysis(library):
    print("\n" + "=" * 70)
    print("  CBR SYSTEM COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    best_k,      k_results      = find_optimal_k(library, max_k=10)
    metric_results, best_metric = compare_similarity_metrics(library)

    print("\n" + "=" * 70)
    print("  FINAL RECOMMENDATIONS")
    print("=" * 70)
    print(f"\n  Best K value     : {best_k}")
    print(f"  Best metric      : {best_metric}")
    print(f"  Current config   : K={K}, Euclidean")
    print(f"\n  Recommended changes:")
    print(f"    Set K = {best_k}")
    print(f"    Use {best_metric} distance")

    with open("cbr_optimization_analysis.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CBR SYSTEM OPTIMIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write("K-VALUE OPTIMIZATION RESULTS:\n")
        f.write("-" * 40 + "\n")
        for k, acc in k_results.items():
            f.write(f"k={k}: {acc:.2f}%\n")
        f.write(f"\nBest K: {best_k}  (Accuracy: {k_results[best_k]:.2f}%)\n\n")
        f.write("SIMILARITY METRICS COMPARISON:\n")
        f.write("-" * 40 + "\n")
        for m, acc in metric_results.items():
            f.write(f"{m}: {acc:.2f}%\n")
        f.write(f"\nBest metric: {best_metric}  ({metric_results[best_metric]:.2f}%)\n\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"1. Change K from {K} to {best_k}\n")
        f.write(f"2. Use {best_metric} distance\n")

    print("\n  Full analysis saved to cbr_optimization_analysis.txt")
    return best_k, best_metric


# ─────────────────────────────────────────────
#  PLOT FEATURE SPACE
# ─────────────────────────────────────────────

def plot_feature_space(library, query_values=None):
    if len(Features) < 2:
        print("[PLOT] Need at least 2 features.")
        return

    vib_col  = Features[0]
    temp_col = Features[1]
    colors_map = {0: 'green', 1: 'orange', 2: 'red'}

    fig, ax = plt.subplots(figsize=(10, 8))

    for label, color in colors_map.items():
        mask = library['Fault Label'] == label
        ax.scatter(library.loc[mask, vib_col], library.loc[mask, temp_col],
                   c=color, label=Fault_Map[label],
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    if query_values is not None:
        qv = query_values[vib_col]
        qt = query_values[temp_col]
        ax.scatter(qv, qt, c='black', marker='X', s=200,
                   edgecolors='white', linewidth=2, zorder=5, label='Query Point')
        ax.annotate(f'Query\n(V={qv:.2f}, T={qt:.1f})',
                    (qv, qt), xytext=(10, 10), textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    if 'high_vibration' in Rules:
        ax.axvline(Rules['high_vibration']['threshold'],
                   color='orange', linestyle='--', alpha=0.5,
                   label='Vibration threshold')
    if 'high_temp' in Rules:
        ax.axhline(Rules['high_temp']['threshold'],
                   color='red', linestyle='--', alpha=0.5,
                   label='Temperature threshold')

    ax.set_xlabel(vib_col,  fontsize=12)
    ax.set_ylabel(temp_col, fontsize=12)
    ax.set_title("Feature Space — Vibration vs Temperature",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_space_plot.png", dpi=150)
    print("  [PLOT] Saved as feature_space_plot.png")
    plt.show()


# ─────────────────────────────────────────────
#  LIBRARY HEALTH CHECK
# ─────────────────────────────────────────────

def library_health_check(library, tolerance=0.01):
    print("\n" + "=" * 60)
    print("  LIBRARY HEALTH CHECK")
    print("=" * 60)
    print(f"\n  Total cases : {len(library)}")

    norm_params = compute_normalization_params(library)
    norm_lib    = normalize_library(library, norm_params)

    # 1. Near-duplicate detection
    print("\n" + "-" * 40)
    print("  NEAR-DUPLICATE DETECTION")
    print("-" * 40)
    duplicates = []
    for i in range(len(norm_lib)):
        for j in range(i + 1, len(norm_lib)):
            dist = np.linalg.norm(norm_lib[i] - norm_lib[j])
            if dist < tolerance:
                duplicates.append((i, j, dist))

    if duplicates:
        print(f"  Found {len(duplicates)} near-duplicate pairs (distance < {tolerance}):")
        for i, j, dist in duplicates[:10]:
            print(f"    Case {i} <-> Case {j}: distance={dist:.6f}"
                  f"  labels={library.iloc[i]['Fault Label']}"
                  f"/{library.iloc[j]['Fault Label']}")
        if len(duplicates) > 10:
            print(f"    ... and {len(duplicates) - 10} more")
        print("  RECOMMENDATION: Remove duplicates to avoid overfitting")
    else:
        print(f"  No near-duplicate cases found (tolerance={tolerance})")

    # 2. Class distribution
    print("\n" + "-" * 40)
    print("  CLASS DISTRIBUTION")
    print("-" * 40)
    class_counts = {}
    class_pcts   = {}
    for label, name in Fault_Map.items():
        count = (library['Fault Label'] == label).sum()
        pct   = count / len(library) * 100
        class_counts[label] = count
        class_pcts[label]   = pct
        bar = '█' * int(pct / 2)
        print(f"  {label} - {name[:25]:<25}: {count:>4} ({pct:5.1f}%)  {bar}")

    imbalanced = [l for l, p in class_pcts.items() if p < 10 and p > 0]
    if imbalanced:
        print(f"\n  WARNING: Imbalanced classes:")
        for l in imbalanced:
            print(f"    Fault {l}: only {class_pcts[l]:.1f}% of library")
        print("  RECOMMENDATION: Collect more minority-class data")
    else:
        print("\n  Class distribution is reasonably balanced")

    # 3. Missing values
    print("\n" + "-" * 40)
    print("  DATA QUALITY")
    print("-" * 40)
    missing = library[Features].isnull().sum()
    if missing.sum() > 0:
        for f in Features:
            if missing[f] > 0:
                print(f"  Missing in {f}: {missing[f]}")
        print("  RECOMMENDATION: Impute or remove missing values")
    else:
        print("  No missing values detected")

    # 4. Health score
    health = 100
    issues = []
    if duplicates:
        penalty = min(20, len(duplicates) * 2)
        health -= penalty
        issues.append(f"  {len(duplicates)} duplicate pairs (-{penalty})")
    if imbalanced:
        health -= 15
        issues.append("  Class imbalance (-15)")
    if missing.sum() > 0:
        health -= 10
        issues.append("  Missing values (-10)")

    grade = ("POOR  — needs immediate attention" if health < 60 else
             "FAIR  — some improvements recommended" if health < 80 else
             "GOOD  — library is healthy")

    print("\n" + "=" * 60)
    print(f"  Health Score : {health}/100")
    print(f"  Status       : {grade}")
    if issues:
        print("  Issues:")
        for issue in issues:
            print(issue)
    print("=" * 60)

    return {'duplicates': len(duplicates), 'class_counts': class_counts,
            'class_pcts': class_pcts, 'health_score': health}


# ─────────────────────────────────────────────
#  SIMULATION MODE
# ─────────────────────────────────────────────

def simulate(library, n_rows=20, delay=0.3):
    print("\n" + "=" * 52)
    print(f"  SIMULATION MODE  ({n_rows} readings, {delay}s interval)")
    print("=" * 52)

    norm_params = compute_normalization_params(library)
    norm_lib    = normalize_library(library, norm_params)
    history     = []

    for i in range(min(n_rows, len(library))):
        row   = library.iloc[i]
        query = {f: row[f] for f in Features}
        top_cases, distances, confidence = retrieve_top_k(
            library, norm_lib, query, norm_params
        )
        suggested = reuse_majority_vote(top_cases)
        final, was_revised, _ = revise(suggested, query)
        status = "REVISED" if was_revised else "CBR"
        flag   = " !" if final != 0 else ""
        print(f"  [{i+1:>3}] V={query[Features[0]]:.2f}  "
              f"T={query[Features[1]]:.1f}  "
              f"P={query[Features[2]]:.2f}  ->  "
              f"Fault {final} ({status}) conf={confidence*100:.0f}%{flag}")

        history.append({
            'index':       i + 1,
            'vibration':   query[Features[0]],
            'temperature': query[Features[1]],
            'pressure':    query[Features[2]],
            'fault':       final,
            'confidence':  confidence,
        })
        time.sleep(delay)

    hist_df = pd.DataFrame(history)
    colors  = hist_df['fault'].map({0: 'green', 1: 'orange', 2: 'red'})

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Sensor Trends — Simulation", fontsize=13)

    for ax, col, color, ylabel in [
        (axes[0], 'vibration',   'steelblue',    'Vibration (mm/s)'),
        (axes[1], 'temperature', 'tomato',        'Temperature (°C)'),
        (axes[2], 'pressure',    'mediumpurple',  'Pressure (bar)'),
    ]:
        ax.plot(hist_df['index'], hist_df[col], color=color, linewidth=1.5)
        ax.scatter(hist_df['index'], hist_df[col], c=colors, zorder=5, s=40)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    axes[2].set_xlabel('Reading #')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',  label='Normal',    markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Mechanical', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',    label='Overheat',   markersize=8),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig("simulation_trends.png", dpi=150)
    print("\n  [SIM] Trend chart saved as simulation_trends.png")
    plt.show()


# ─────────────────────────────────────────────
#  CASE LIBRARY STATISTICS
# ─────────────────────────────────────────────

def print_library_stats(library):
    print("\n" + "=" * 52)
    print("  CASE LIBRARY STATISTICS")
    print("=" * 52)
    print(f"  Total cases   : {len(library)}")

    valid_ts = library['Timestamp'].dropna()
    if len(valid_ts) > 0:
        print(f"  Date range    : {valid_ts.min().date()} -> {valid_ts.max().date()}")
    else:
        print("  Date range    : N/A")

    print()
    for label, name in Fault_Map.items():
        count = (library['Fault Label'] == label).sum()
        pct   = count / len(library) * 100
        bar   = '█' * int(pct / 5)
        print(f"  Fault {label}  {name[:35]:<35}  {count:>4} ({pct:5.1f}%)  {bar}")

    print()
    for f in Features:
        mn   = library[f].min()
        mx   = library[f].max()
        mean = library[f].mean()
        print(f"  {f:<28}  min={mn:.2f}  max={mx:.2f}  mean={mean:.2f}")


# ─────────────────────────────────────────────
#  ANOMALY DEMO
# ─────────────────────────────────────────────

def demo_anomaly_detection(library, norm_params, norm_lib):
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTION DEMO")
    print("=" * 70)

    cases = [
        ("Normal input (within range)",        {Features[0]: 0.85, Features[1]: 115.2, Features[2]: 8.1}),
        ("Anomalous vibration (9.0 >> max)",   {Features[0]: 9.0,  Features[1]: 115.2, Features[2]: 8.1}),
        ("Multiple anomalies",                 {Features[0]: 9.0,  Features[1]: 250.0, Features[2]: 2.0}),
    ]

    for label, query in cases:
        print(f"\n  TEST: {label}")
        diagnose(library, norm_lib, norm_params, query, retain=False)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Load and prepare
    df = load_case_library(FILE_PATH)
    df = df.dropna(subset=['Vibration (mm/s)']).reset_index(drop=True)
    setup_features(df)
    print_library_stats(df)

    norm_params = compute_normalization_params(df)
    norm_lib    = normalize_library(df, norm_params)

    print("\n  Normal operating ranges (from training data):")
    for f in Features:
        print(f"  {f:<28} [{norm_params[f]['min']:.2f}, {norm_params[f]['max']:.2f}]")

    # Feature space plot
    plot_feature_space(df)

    # Optional interactive steps
    if input("\nRun library health check? (y/n): ").lower() == 'y':
        library_health_check(df)

    if input("\nRun anomaly detection demo? (y/n): ").lower() == 'y':
        demo_anomaly_detection(df, norm_params, norm_lib)

    if input("\nRun comprehensive optimization analysis? (y/n): ").lower() == 'y':
        run_comprehensive_analysis(df)

    # Standard diagnoses
    print("\n" + "=" * 52)
    print("  RUNNING STANDARD DIAGNOSES")
    print("=" * 52)

    test_cases = [
        {Features[0]: 0.85, Features[1]: 115.2, Features[2]: 8.1},
        {Features[0]: 2.8,  Features[1]: 72.0,  Features[2]: 6.5},
        {Features[0]: 0.4,  Features[1]: 145.0, Features[2]: 9.2},
        {Features[0]: 0.2,  Features[1]: 60.0,  Features[2]: 7.0},
    ]

    for query in test_cases:
        df, norm_lib, norm_params, _ = diagnose(
            df, norm_lib, norm_params, query, retain=True
        )

    # Evaluation
    evaluate(df)

    # Simulation
    simulate(df, n_rows=20, delay=0.3)