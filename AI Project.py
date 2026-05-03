import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# INITIALIZATION: Loading your specific CSV
# Using the path you provided
FILE_PATH = r"C:\Users\ALLAH\Downloads\industrial_fault_detection_data_1000.csv"

# Feature weights - higher important for similarity
# Will be set automatically after loading CSV
Features = []
Feautres_Weight = {}
K = 3

# Representation: Knowledge Mapping
Fault_Map = {
    0: "Normal Operation (No Fault)",
    1: "Mechanical Vibration / Bearing Issue",
    2: "System Overheating / Thermal Stress"
}

Rules = {}

# Report directory
REPORT_DIR = "cbr_reports"


def load_case_library(path):
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    print(f"[SYSTEM] Loaded {len(df)} cases from: {path}")
    return df


def setup_features(library):
    global Features, Feautres_Weight, Rules

    cols = library.columns.tolist()

    vib_col = next(c for c in cols if 'Vibration' in c and '(' in c)
    temp_col = next(c for c in cols if 'Temp' in c and '(' in c)
    pres_col = next(c for c in cols if 'Pressure' in c and '(' in c)

    Features = [vib_col, temp_col, pres_col]
    Feautres_Weight = {
        vib_col: 2.0,
        temp_col: 1.5,
        pres_col: 1.0
    }
    Rules = {
        'high_temp': {'feature': temp_col, 'threshold': 130, 'op': '>', 'label': 2},
        'high_vibration': {'feature': vib_col, 'threshold': 2.5, 'op': '>', 'label': 1},
        'low_vibration': {'feature': vib_col, 'threshold': 0.3, 'op': '<', 'label': 0},
    }
    print(f"[SYSTEM] Features detected: {Features}")


# STEP 1: NORMALIZE (fixes scale imbalance)
def compute_normalization_params(library):
    params = {}
    for feature in Features:
        params[feature] = {
            'min': library[feature].min(),
            'max': library[feature].max()
        }
    return params


def normalize(values_dict, norm_params):
    normalized = []
    for feature in Features:
        mn = norm_params[feature]['min']
        mx = norm_params[feature]['max']
        val = values_dict[feature]
        norm_val = (val - mn) / (mx - mn) if mx != mn else 0.0
        normalized.append(norm_val)
    return np.array(normalized)


def normalize_library(library, norm_params):
    norm_rows = []
    for _, row in library.iterrows():
        values = {f: row[f] for f in Features}
        norm_rows.append(normalize(values, norm_params))
    return np.array(norm_rows)


# ANOMALY DETECTION - Check if sensor readings are outside training range
def detect_anomaly(query_values, norm_params):
    """
    Detects if any sensor reading is outside the training data range.
    Returns: (is_anomaly, anomaly_report)
    """
    anomalies = []
    
    for feature in Features:
        val = query_values[feature]
        mn = norm_params[feature]['min']
        mx = norm_params[feature]['max']
        
        # Calculate how far outside the range (as percentage)
        if val < mn:
            percent_off = ((mn - val) / mn) * 100 if mn != 0 else float('inf')
            anomalies.append({
                'feature': feature,
                'value': val,
                'min': mn,
                'max': mx,
                'deviation': f"{percent_off:.1f}% below minimum",
                'severity': 'HIGH' if percent_off > 50 else 'MEDIUM'
            })
        elif val > mx:
            percent_off = ((val - mx) / mx) * 100 if mx != 0 else float('inf')
            anomalies.append({
                'feature': feature,
                'value': val,
                'min': mn,
                'max': mx,
                'deviation': f"{percent_off:.1f}% above maximum",
                'severity': 'HIGH' if percent_off > 50 else 'MEDIUM'
            })
    
    is_anomaly = len(anomalies) > 0
    
    # Generate anomaly report
    if is_anomaly:
        report = "\n" + "!" * 52
        report += "\n  ⚠  ANOMALY DETECTED  ⚠"
        report += "\n  Sensor readings outside training range!"
        report += "\n" + "!" * 52
        for a in anomalies:
            report += f"\n  • {a['feature']}"
            report += f"\n    Value: {a['value']:.3f}"
            report += f"\n    Range: [{a['min']:.3f}, {a['max']:.3f}]"
            report += f"\n    Status: {a['deviation']}"
            report += f"\n    Severity: {a['severity']}"
        report += "\n" + "!" * 52
        report += "\n  RECOMMENDATION: Manual inspection required!"
        report += "\n  CBR results may be unreliable for this input."
        report += "\n" + "!" * 52 + "\n"
        return True, report, anomalies
    else:
        return False, "All sensors within normal range.", None


# Retrieve
def retrieve_top_k(library, norm_library, query_values, norm_params, k=K):
    weights = np.array([Feautres_Weight[f] for f in Features])
    query_norm = normalize(query_values, norm_params)
    diff = norm_library - query_norm
    distances = np.sqrt((diff ** 2 * weights).sum(axis=1))
    top_k_indices = np.argsort(distances)[:k]
    top_cases = library.iloc[top_k_indices].copy()
    top_distances = distances[top_k_indices]

    # Confidence: inverse of closest distance, scaled to 0-1
    confidence = 1.0 / (1.0 + top_distances[0])
    return top_cases, top_distances, confidence


# Revise
def revise(suggested_label, query_values):
    for rule_name, rule in Rules.items():
        feature = rule['feature']
        val = query_values[feature]
        threshold = rule['threshold']
        op = rule['op']
        triggered = (op == '>' and val > threshold) or \
                    (op == '<' and val < threshold)
        if triggered:
            return rule['label'], True, rule_name
    return suggested_label, False, None


# RETAIN: Saving new knowledge back to the CSV file
def retain_new_case(library, query_values, fault_label, path, tolerance=0.01):
    # Check for near-duplicate
    norm_params = compute_normalization_params(library)
    norm_lib = normalize_library(library, norm_params)
    query_norm = normalize(query_values, norm_params)
    distances = np.linalg.norm(norm_lib - query_norm, axis=1)

    if distances.min() < tolerance:
        print("[RETAIN] Case too similar to existing entry — skipped.")
        return library, False

    new_row = {
        'Timestamp': pd.Timestamp.now(),
        'Fault Label': fault_label
    }
    new_row.update(query_values)

    updated = pd.concat([library, pd.DataFrame([new_row])], ignore_index=True)
    updated.to_csv(path, index=False)
    print(f"[RETAIN] New case saved. Library now has {len(updated)} cases.")
    return updated, True


def reuse_majority_vote(top_cases):
    votes = top_cases['Fault Label'].astype(int).value_counts()
    return int(votes.idxmax())


# ─────────────────────────────────────────────
#  GENERATE REPORT FUNCTION
# ─────────────────────────────────────────────

def generate_report(
    query_values,
    top_cases,
    distances,
    suggested_label,
    final_label,
    was_revised,
    rule_name,
    confidence,
    was_retained,
    is_anomaly=False,
    anomaly_report=None,
    report_dir=REPORT_DIR
):
    os.makedirs(report_dir, exist_ok=True)

    ts = datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    filename = f"cbr_report_{ts_str}.txt"
    filepath = os.path.join(report_dir, filename)

    if final_label == 0:
        severity = "LOW    — Normal operation"
    elif final_label == 1:
        severity = "MEDIUM — Mechanical fault detected"
    else:
        severity = "HIGH   — Thermal fault detected"

    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 56)
        lines.append(f"  {title}")
        lines.append("=" * 56)

    def row(label, value):
        lines.append(f"  {label:<28} {value}")

    lines.append("=" * 56)
    lines.append("  CBR EXPERT SYSTEM — FAULT DIAGNOSIS REPORT")
    lines.append("=" * 56)
    row("Generated", ts.strftime("%Y-%m-%d  %H:%M:%S"))
    row("Report file", filename)

    if is_anomaly:
        lines.append("")
        lines.append("!" * 56)
        lines.append("  ⚠  ANOMALY DETECTED - RESULTS MAY BE UNRELIABLE  ⚠")
        lines.append("!" * 56)
        if anomaly_report:
            lines.append(anomaly_report)
        lines.append("!" * 56)

    section("STAGE 1 — SENSOR READINGS (Query)")
    for feature, value in query_values.items():
        row(feature, f"{value:.4f}")

    section("STAGE 2 — RETRIEVE  (top-k similar cases)")
    row("Cases searched", "full library")
    row("k neighbours", str(len(top_cases)))
    lines.append("")
    for i, (_, case_row) in enumerate(top_cases.iterrows()):
        label = int(case_row['Fault Label'])
        label_str = Fault_Map.get(label, "Unknown")
        ts_case = str(case_row['Timestamp'])[:19]
        dist = distances[i]
        match_str = "CLOSEST MATCH" if i == 0 else f"Match {i+1}"
        lines.append(f"  [{match_str}]")
        row("  Timestamp", ts_case)
        row("  Distance", f"{dist:.6f}")
        row("  Fault label", f"{label}  —  {label_str}")
        lines.append("")

    section("STAGE 3 — REUSE  (majority vote)")
    suggested_str = Fault_Map.get(suggested_label, "Unknown")
    row("Suggested label", f"{suggested_label}  —  {suggested_str}")
    row("Confidence", f"{confidence * 100:.1f}%")
    if confidence < 0.4:
        row("WARNING", "Low confidence — manual inspection advised")

    section("STAGE 4 — REVISE  (rule-based check)")
    if was_revised:
        row("Rule triggered", rule_name)
        row("CBR suggested", f"{suggested_label}  —  {suggested_str}")
        row("Revised to", f"{final_label}  —  {Fault_Map[final_label]}")
    else:
        row("Rule triggered", "None — CBR result accepted")

    section("FINAL DIAGNOSIS")
    row("Fault label", f"{final_label}")
    row("Condition", Fault_Map.get(final_label, "Unknown"))
    row("Severity", severity)
    row("Confidence", f"{confidence * 100:.1f}%")
    row("Method", "Rule override" if was_revised else "CBR majority vote")
    if is_anomaly:
        row("⚠ CAUTION", "Anomalous input - diagnosis may be invalid")

    section("STAGE 5 — RETAIN")
    if is_anomaly:
        row("Status", "NOT SAVED - Anomalous case rejected")
        row("Reason", "Input outside training range - requires validation")
    elif was_retained:
        row("Status", "Case saved to library")
        row("Label retained", f"{final_label}  —  {Fault_Map[final_label]}")
    else:
        row("Status", "Case NOT saved (too similar to existing entry)")

    section("RECOMMENDED ACTION")
    if is_anomaly:
        lines.append("  🚨 IMMEDIATE ACTION REQUIRED:")
        lines.append("     1. Verify sensor calibration")
        lines.append("     2. Inspect equipment manually")
        lines.append("     3. Do not rely on automated diagnosis")
    else:
        actions = {
            0: "No action required. Continue normal monitoring.",
            1: "Inspect bearings and mechanical components. Schedule maintenance.",
            2: "Check cooling system immediately. Reduce load if temperature persists.",
        }
        lines.append(f"  {actions.get(final_label, 'Consult engineer.')}")

    lines.append("")
    lines.append("=" * 56)
    lines.append("  END OF REPORT")
    lines.append("=" * 56)
    lines.append("")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[REPORT] Saved → {filepath}")
    return filepath


# DIAGNOSE FUNCTION
def diagnose(library, norm_library, norm_params, query_values, retain=False):
    print("\n" + "=" * 52)
    print("  SENSOR READINGS")
    print("=" * 52)
    for f, v in query_values.items():
        print(f"  {f:<28} {v}")

    is_anomaly, anomaly_msg, anomaly_details = detect_anomaly(query_values, norm_params)
    
    if is_anomaly:
        print(anomaly_msg)
        print("\n  ⚠  Continuing with diagnosis, but results may be unreliable.")
        print("  Recommendation: Manual inspection required!\n")

    top_cases, distances, confidence = retrieve_top_k(
        library, norm_library, query_values, norm_params
    )

    print(f"\n  Top-{K} similar cases retrieved:")
    for i, (_, row) in enumerate(top_cases.iterrows()):
        label = Fault_Map.get(int(row['Fault Label']), 'Unknown')
        print(f"    [{i+1}] dist={distances[i]:.4f}  →  {label}  ({row['Timestamp']})")

    suggested_label = reuse_majority_vote(top_cases)
    final_label, was_revised, rule_name = revise(suggested_label, query_values)

    print("\n" + "-" * 52)
    if was_revised:
        print(f"  [REVISE] Rule triggered: '{rule_name}'")
        print(f"  CBR suggested : {Fault_Map[suggested_label]}")
    print(f"  DIAGNOSIS     : {Fault_Map[final_label]}")
    print(f"  CONFIDENCE    : {confidence * 100:.1f}%")

    if confidence < 0.4:
        print("  ⚠  Low confidence — consider manual inspection")
    
    if is_anomaly:
        print("  ⚠⚠  ANOMALOUS INPUT - Diagnosis may be invalid!  ⚠⚠")

    print("-" * 52)

    was_retained = False
    if retain:
        if is_anomaly:
            print("[RETAIN] Skipped - Anomalous case not saved to library.")
            print("         Manual review required before adding to knowledge base.")
        else:
            library, was_retained = retain_new_case(
                library, query_values, final_label, FILE_PATH
            )
            norm_params = compute_normalization_params(library)
            norm_library = normalize_library(library, norm_params)

    generate_report(
        query_values=query_values,
        top_cases=top_cases,
        distances=distances,
        suggested_label=suggested_label,
        final_label=final_label,
        was_revised=was_revised,
        rule_name=rule_name,
        confidence=confidence,
        was_retained=was_retained,
        is_anomaly=is_anomaly,
        anomaly_report=anomaly_msg if is_anomaly else None,
    )

    return library, norm_library, norm_params, final_label


# FAST EVALUATION (no plots, just accuracy)
def evaluate_fast(library):
    shuffled_lib = library.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(shuffled_lib) * 0.8)
    train_lib = shuffled_lib.iloc[:split].reset_index(drop=True)
    test_lib = shuffled_lib.iloc[split:].reset_index(drop=True)
    
    norm_params = compute_normalization_params(train_lib)
    norm_train = normalize_library(train_lib, norm_params)
    
    y_true, y_pred = [], []
    
    for _, row in test_lib.iterrows():
        query = {f: row[f] for f in Features}
        top_cases, _, _ = retrieve_top_k(train_lib, norm_train, query, norm_params)
        suggested = reuse_majority_vote(top_cases)
        final, _, _ = revise(suggested, query)
        y_true.append(int(row['Fault Label']))
        y_pred.append(final)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred) * 100
    
    return accuracy


# FIND OPTIMAL K VALUE
def find_optimal_k(library, max_k=10):
    print("\n" + "=" * 70)
    print("  OPTIMAL K VALUE ANALYSIS")
    print("  Testing k=1 through k={}".format(max_k))
    print("=" * 70)
    
    results = {}
    k_values = range(1, max_k + 1)
    
    global K
    original_k = K
    
    for k in k_values:
        print(f"\n  Testing k={k}...")
        K = k
        accuracy = evaluate_fast(library)
        results[k] = accuracy
        print(f"    → Accuracy: {accuracy:.2f}%")
    
    K = original_k
    
    best_k = max(results, key=results.get)
    best_accuracy = results[best_k]
    
    print("\n" + "-" * 70)
    print("  SUMMARY")
    print("-" * 70)
    for k in k_values:
        marker = " ★ BEST" if k == best_k else ""
        print(f"  k={k:<3}  →  Accuracy: {results[k]:6.2f}%{marker}")
    print("-" * 70)
    print(f"\n  Best K Value: {best_k} (Accuracy: {best_accuracy:.2f}%)")
    print(f"  Current K: {original_k}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(results.keys()), list(results.values()), 
            marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('K Value (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Optimal K Value Selection for CBR System', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    ax.scatter(best_k, best_accuracy, color='red', s=150, zorder=5, 
               label=f'Best K={best_k} ({best_accuracy:.1f}%)')
    ax.legend()
    
    for k, acc in results.items():
        ax.annotate(f'{acc:.1f}%', (k, acc), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("optimal_k_analysis.png", dpi=150)
    print(f"\n  Plot saved as: optimal_k_analysis.png")
    plt.show()
    
    return best_k, results


# COMPARE SIMILARITY METRICS
def compare_similarity_metrics(library):
    print("\n" + "=" * 70)
    print("  SIMILARITY METRICS COMPARISON")
    print("  Euclidean vs Cosine vs Manhattan Distance")
    print("=" * 70)
    
    results = {}
    
    def euclidean_distance(vec1, vec2, weights):
        diff = vec1 - vec2
        return np.sqrt((diff ** 2 * weights).sum())
    
    def cosine_distance(vec1, vec2, weights):
        vec1_w = vec1 * np.sqrt(weights)
        vec2_w = vec2 * np.sqrt(weights)
        dot_product = np.dot(vec1_w, vec2_w)
        norm1 = np.linalg.norm(vec1_w)
        norm2 = np.linalg.norm(vec2_w)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        similarity = dot_product / (norm1 * norm2)
        return 1 - similarity
    
    def manhattan_distance(vec1, vec2, weights):
        return np.sum(np.abs(vec1 - vec2) * weights)
    
    metrics = {
        'Euclidean': euclidean_distance,
        'Cosine': cosine_distance,
        'Manhattan': manhattan_distance
    }
    
    original_retrieve = retrieve_top_k
    
    for metric_name, distance_func in metrics.items():
        print(f"\n  Testing {metric_name} distance...")
        
        def retrieve_with_metric(library, norm_library, query_values, norm_params, k=K):
            weights = np.array([Feautres_Weight[f] for f in Features])
            query_norm = normalize(query_values, norm_params)
            distances = np.array([
                distance_func(query_norm, case_norm, weights) 
                for case_norm in norm_library
            ])
            top_k_indices = np.argsort(distances)[:k]
            top_cases = library.iloc[top_k_indices].copy()
            top_distances = distances[top_k_indices]
            confidence = 1.0 / (1.0 + top_distances[0])
            return top_cases, top_distances, confidence
        
        globals()['retrieve_top_k'] = retrieve_with_metric
        accuracy = evaluate_fast(library)
        results[metric_name] = accuracy
        print(f"    → Accuracy: {accuracy:.2f}%")
    
    globals()['retrieve_top_k'] = original_retrieve
    
    print("\n" + "-" * 70)
    print("  COMPARISON RESULTS")
    print("-" * 70)
    print(f"  {'Metric':<15} {'Accuracy':<15} {'Rank':<10}")
    print("-" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (metric, accuracy) in enumerate(sorted_results, 1):
        print(f"  {metric:<15} {accuracy:>6.2f}%{'':<8} #{rank}")
    
    print("-" * 70)
    
    best_metric = max(results, key=results.get)
    best_accuracy = results[best_metric]
    
    print(f"\n  Best Similarity Metric: {best_metric}")
    print(f"  (Accuracy: {best_accuracy:.2f}%)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_list = list(results.keys())
    accuracies = list(results.values())
    colors = ['#2ecc71' if i == np.argmax(accuracies) else '#e74c3c' for i in range(len(metrics_list))]
    bars = ax.bar(metrics_list, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Similarity Metric', fontsize=12)
    ax.set_title('Comparison of Similarity Metrics for CBR System', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    best_idx = np.argmax(accuracies)
    ax.text(best_idx, accuracies[best_idx] - 15, '★ BEST', 
            ha='center', fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("similarity_metrics_comparison.png", dpi=150)
    print(f"\n  Plot saved as: similarity_metrics_comparison.png")
    plt.show()
    
    return results, best_metric


# COMPREHENSIVE ANALYSIS
def run_comprehensive_analysis(library):
    print("\n" + "=" * 70)
    print("  CBR SYSTEM COMPREHENSIVE ANALYSIS")
    print("  Optimizing Parameters for Fault Diagnosis")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    best_k, k_results = find_optimal_k(library, max_k=10)
    
    print("\n" + "=" * 70)
    metric_results, best_metric = compare_similarity_metrics(library)
    
    print("\n" + "=" * 70)
    print("  FINAL RECOMMENDATIONS")
    print("=" * 70)
    print(f"\n  Best K Value: {best_k}")
    print(f"  Best Similarity Metric: {best_metric}")
    print(f"\n  Current Configuration: K={K}, Euclidean distance")
    print(f"\n  Recommended Changes:")
    print(f"    - Set K = {best_k}")
    print(f"    - Use {best_metric} distance")
    
    with open("cbr_optimization_analysis.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CBR SYSTEM OPTIMIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("K-VALUE OPTIMIZATION RESULTS:\n")
        f.write("-" * 40 + "\n")
        for k, acc in k_results.items():
            f.write(f"k={k}: {acc:.2f}%\n")
        f.write(f"\nBest K: {best_k} (Accuracy: {k_results[best_k]:.2f}%)\n\n")
        
        f.write("SIMILARITY METRICS COMPARISON:\n")
        f.write("-" * 40 + "\n")
        for metric, acc in metric_results.items():
            f.write(f"{metric}: {acc:.2f}%\n")
        f.write(f"\nBest Metric: {best_metric} (Accuracy: {metric_results[best_metric]:.2f}%)\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"1. Change K from {K} to {best_k}\n")
        f.write(f"2. Use {best_metric} distance for similarity calculation\n")
    
    print("\n  Analysis saved to: cbr_optimization_analysis.txt")
    
    return best_k, best_metric


# Evaluation
def evaluate(library):
    print("\n" + "=" * 52)
    print("  EVALUATION MODE  (80/20 split)")
    print("=" * 52)

    library = library.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(library) * 0.8)
    train_lib = library.iloc[:split].reset_index(drop=True)
    test_lib = library.iloc[split:].reset_index(drop=True)

    norm_params = compute_normalization_params(train_lib)
    norm_train = normalize_library(train_lib, norm_params)

    y_true, y_pred = [], []

    for _, row in test_lib.iterrows():
        query = {f: row[f] for f in Features}
        top_cases, _, _ = retrieve_top_k(train_lib, norm_train, query, norm_params)
        suggested = reuse_majority_vote(top_cases)
        final, _, _ = revise(suggested, query)
        y_true.append(int(row['Fault Label']))
        y_pred.append(final)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred) * 100

    labels = sorted(Fault_Map.keys())
    n = len(labels)
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
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
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


# SIMULATION MODE
def simulate(library, n_rows=20, delay=0.5):
    print("\n" + "=" * 52)
    print(f"  SIMULATION MODE  ({n_rows} readings, {delay}s interval)")
    print("=" * 52)
    norm_params = compute_normalization_params(library)
    norm_lib = normalize_library(library, norm_params)
    history = []
    for i in range(min(n_rows, len(library))):
        row = library.iloc[i]
        query = {f: row[f] for f in Features}
        top_cases, distances, confidence = retrieve_top_k(
            library, norm_lib, query, norm_params
        )
        suggested = reuse_majority_vote(top_cases)
        final, was_revised, rule_name = revise(suggested, query)
        status = "REVISED" if was_revised else "CBR"
        flag = " ⚠" if final != 0 else ""
        print(f"  [{i+1:>3}] V={query[Features[0]]:.2f}  "
              f"T={query[Features[1]]:.1f}  "
              f"P={query[Features[2]]:.2f}  →  "
              f"Fault {final} ({status}) conf={confidence*100:.0f}%{flag}")

        history.append({
            'index': i + 1,
            'vibration': query['Vibration (mm/s)'],
            'temperature': query['Temperature (°C)'],
            'pressure': query['Pressure (bar)'],
            'fault': final,
            'confidence': confidence
        })

        time.sleep(delay)

    hist_df = pd.DataFrame(history)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Sensor Trends — Live Simulation", fontsize=13)

    fault_colors = {0: 'green', 1: 'orange', 2: 'red'}
    colors = hist_df['fault'].map(fault_colors)

    axes[0].plot(hist_df['index'], hist_df['vibration'], color='steelblue', linewidth=1.5)
    axes[0].scatter(hist_df['index'], hist_df['vibration'], c=colors, zorder=5, s=40)
    axes[0].set_ylabel('Vibration (mm/s)')
    axes[0].set_title('Vibration')

    axes[1].plot(hist_df['index'], hist_df['temperature'], color='tomato', linewidth=1.5)
    axes[1].scatter(hist_df['index'], hist_df['temperature'], c=colors, zorder=5, s=40)
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Temperature')

    axes[2].plot(hist_df['index'], hist_df['pressure'], color='mediumpurple', linewidth=1.5)
    axes[2].scatter(hist_df['index'], hist_df['pressure'], c=colors, zorder=5, s=40)
    axes[2].set_ylabel('Pressure (bar)')
    axes[2].set_xlabel('Reading #')
    axes[2].set_title('Pressure')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Normal', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Mechanical Fault', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Overheating', markersize=8),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig("simulation_trends.png", dpi=150)
    print("\n  [SIM] Trend chart saved as simulation_trends.png")
    plt.show()


# CASE LIBRARY STATISTICS
def print_library_stats(library):
    print("\n" + "=" * 52)
    print("  CASE LIBRARY STATISTICS")
    print("=" * 52)
    print(f"  Total cases   : {len(library)}")

    valid_ts = library['Timestamp'].dropna()
    if len(valid_ts) > 0:
        print(f"  Date range    : {valid_ts.min().date()} → {valid_ts.max().date()}")
    else:
        print(f"  Date range    : N/A")

    print()
    for label, name in Fault_Map.items():
        count = (library['Fault Label'] == label).sum()
        pct = count / len(library) * 100
        bar = '█' * int(pct / 5)
        print(f"  Fault {label}  {name[:35]:<35}  {count:>4} ({pct:5.1f}%)  {bar}")
    print()
    for f in Features:
        mn = library[f].min()
        mx = library[f].max()
        mean = library[f].mean()
        print(f"  {f:<28}  min={mn:.2f}  max={mx:.2f}  mean={mean:.2f}")


# DEMO FUNCTION
def demo_anomaly_detection(library, norm_params, norm_lib):
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTION DEMO")
    print("=" * 70)
    
    print("\n▶ TEST 1: Normal input (within training range)")
    normal_query = {
        Features[0]: 0.85,
        Features[1]: 115.2,
        Features[2]: 8.1
    }
    diagnose(library, norm_lib, norm_params, normal_query, retain=False)
    
    print("\n\n▶ TEST 2: Anomalous input (outside training range)")
    anomaly_query = {
        Features[0]: 9.0,
        Features[1]: 115.2,
        Features[2]: 8.1
    }
    diagnose(library, norm_lib, norm_params, anomaly_query, retain=False)
    
    print("\n\n▶ TEST 3: Multiple anomalies")
    multi_anomaly = {
        Features[0]: 9.0,
        Features[1]: 250.0,
        Features[2]: 2.0
    }
    diagnose(library, norm_lib, norm_params, multi_anomaly, retain=False)


# MAIN
if __name__ == "__main__":

    df = load_case_library(FILE_PATH)
    print_library_stats(df)

    df = df.dropna(subset=['Vibration (mm/s)']).reset_index(drop=True)
    setup_features(df)
    norm_params = compute_normalization_params(df)
    norm_lib = normalize_library(df, norm_params)

    print("\n" + "=" * 52)
    print("  NORMAL OPERATING RANGES (from training data)")
    print("=" * 52)
    for f in Features:
        print(f"  {f:<28} [{norm_params[f]['min']:.2f}, {norm_params[f]['max']:.2f}]")
    
    print("\n" + "=" * 52)
    choice = input("Run anomaly detection demo? (y/n): ").lower()
    if choice == 'y':
        demo_anomaly_detection(df, norm_params, norm_lib)
    
    print("\n" + "=" * 52)
    choice2 = input("Run comprehensive optimization analysis? (y/n): ").lower()
    if choice2 == 'y':
        best_k, best_metric = run_comprehensive_analysis(df)
    
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

    evaluate(df)
    simulate(df, n_rows=20, delay=0.3)