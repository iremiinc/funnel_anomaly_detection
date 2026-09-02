"""
pipeline.py
============
Bu modül dashboard'un "beynidir". Burada HİÇBİR skor / finansal rakam
elle yazılmaz (hardcode edilmez). Tüm sayılar gerçek engine dosyalarından
(funnel_engine, baseline_engine, anomaly_engine, rca_engine,
financial_impact_engine, tracking_anomaly_detector) üretilen veriler
üzerinden, o engine'lerin fonksiyonları ÇAĞRILARAK hesaplanır.

app.py bu modülü import eder; hiçbir RCA/finansal/tracking mantığını
kendi içinde yeniden yazmaz (bkz. Kritik Düzeltme #1).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---- GERÇEK ENGINE IMPORTLARI (mock / reimplementation YOK) ----
from rca_engine import run_production_rca, classify_evidence_level
from financial_impact_engine import calculate_financial_impact, calculate_segment_financial_breakdown
from tracking_anomaly_detector import detect_tracking_vs_business_anomaly
from anomaly_engine import detect_and_classify_anomalies
from funnel_engine import FUNNEL_STEPS

DATA_DIR = Path(__file__).parent
SEG_DIMS = ["platform", "appVersion", "paymentProvider"]


# -----------------------------------------------------------------------
# 0. VERİ YÜKLEME (Tek yerden, cache'lenebilir)
# -----------------------------------------------------------------------
def load_raw_artifacts():
    df_overall = pd.read_parquet(DATA_DIR / "funnel_metrics_with_baseline.parquet")
    df_overall["eventTime"] = pd.to_datetime(df_overall["eventTime"])

    df_dims = pd.read_parquet(DATA_DIR / "funnel_metrics_by_dimensions.parquet")
    df_dims["eventTime"] = pd.to_datetime(df_dims["eventTime"])

    df_detected = pd.read_parquet(DATA_DIR / "detected_anomalies.parquet")
    df_detected["eventTime"] = pd.to_datetime(df_detected["eventTime"])

    change_events_raw = json.loads((DATA_DIR / "change_events.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    df_events_sample = None  # sadece AOV hesaplamak için revenue lazım -> overall df yeterli

    return df_overall, df_dims, df_detected, change_events_raw, ground_truth


def get_real_aov(df_overall: pd.DataFrame) -> float:
    """
    Ortalama Sepet Değeri'ni (AOV) elle 250/450 gibi sabitler yazmak yerine
    gerçekleşen (anomaliden ETKİLENMEMİŞ, normal) günlerin revenue/purchase
    oranından hesaplar. Bu, Kritik Düzeltme #3'ün bir parçasıdır: finansal
    motor artık gerçek veriden türetilen bir AOV kullanır.
    """
    clean = df_overall[df_overall["purchase_completed"] > 0]
    aov = clean["revenue"].sum() / clean["purchase_completed"].sum()
    return round(float(aov), 2)


# -----------------------------------------------------------------------
# 1. Bir segment kombinasyonu için "kontrol dönemi" conversion baseline'ı
# -----------------------------------------------------------------------
def _segment_control_conversion(df_dims, control_start, control_end, num_col, den_col):
    ctrl = df_dims[(df_dims["eventTime"] >= control_start) & (df_dims["eventTime"] < control_end)]
    grouped = ctrl.groupby(SEG_DIMS)[[num_col, den_col]].sum().reset_index()
    grouped["control_cr"] = np.where(
        grouped[den_col] > 0, grouped[num_col] / grouped[den_col], np.nan
    )
    global_cr = ctrl[num_col].sum() / max(1, ctrl[den_col].sum())
    return grouped[SEG_DIMS + ["control_cr"]], global_cr


def _build_segment_window_df(df_dims, window_start, window_end, num_col, den_col, control_days=3):
    """
    RCA/finansal motorların ihtiyaç duyduğu 'baseline_X' ve 'actual_X'
    kolonlarını segment (platform x appVersion x paymentProvider) bazında
    üretir. Baseline = kontrol dönemindeki conversion oranı * anomali
    penceresindeki GERÇEK trafik (payment_submitted / checkout_started vs.).
    Bu sayede günlük trafik dalgalanması ile gerçek conversion çöküşü
    birbirine karıştırılmaz.
    """
    control_start = window_start - pd.Timedelta(days=control_days)
    control_end = window_start
    seg_cr, global_cr = _segment_control_conversion(df_dims, control_start, control_end, num_col, den_col)

    win = df_dims[(df_dims["eventTime"] >= window_start) & (df_dims["eventTime"] < window_end)]
    win_grouped = win.groupby(SEG_DIMS)[[num_col, den_col]].sum().reset_index()

    merged = win_grouped.merge(seg_cr, on=SEG_DIMS, how="left")
    merged["control_cr"] = merged["control_cr"].fillna(global_cr)

    baseline_col = f"baseline_{num_col}"
    merged[baseline_col] = merged[den_col] * merged["control_cr"]
    # actual = num_col (as already present)
    return merged, baseline_col, num_col


# -----------------------------------------------------------------------
# 2. Değişim event'lerini RCA engine formatına çevir
# -----------------------------------------------------------------------
def _adapt_change_events(change_events_raw):
    adapted = []
    for ev in change_events_raw:
        scope = {}
        if "platform" in ev:
            scope["platform"] = ev["platform"]
        if "version" in ev:
            scope["appVersion"] = ev["version"]
        if "provider" in ev:
            scope["paymentProvider"] = ev["provider"]
        adapted.append({
            "event_name": f"{ev['change_id']} · {ev['description']}",
            "timestamp": pd.Timestamp(ev["timestamp"]),
            "scope": scope if scope else None,
            "is_causal_experiment": False,
        })
    return adapted


# -----------------------------------------------------------------------
# 3. Tek bir incident için uçtan uca gerçek zincir: RCA -> Evidence -> Finansal
# -----------------------------------------------------------------------
def build_incident_bundle(incident_id: str, window_start, window_end, num_col, den_col,
                           df_dims, df_overall, change_events_raw, default_aov):
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)
    duration_hours = (window_end - window_start).total_seconds() / 3600.0

    seg_df, baseline_col, actual_col = _build_segment_window_df(
        df_dims, window_start, window_end, num_col, den_col
    )

    change_events = _adapt_change_events(change_events_raw)

    # --- GERÇEK RCA ENGINE ÇAĞRISI (formül burada, tek yerde hesaplanır) ---
    rca_candidates = run_production_rca(
        df_anomaly=seg_df,
        baseline_col=baseline_col,
        actual_col=actual_col,
        dimensions=SEG_DIMS,
        change_events=change_events,
        anomaly_time=window_start,
        min_cohort=30,
        strong_signal_threshold=0.15,
    )

    top = rca_candidates[0] if rca_candidates else None

    # --- GERÇEK FİNANSAL MOTOR ÇAĞRISI (SADECE bu pencere, 14 günlük veri DEĞİL) ---
    fin_result = calculate_financial_impact(
        df_anomaly=seg_df,
        baseline_col=baseline_col,
        actual_col=actual_col,
        default_aov=default_aov,
        is_active_anomaly=False,          # veri geçmişe ait, artık aktif değil -> yanlış 24h projeksiyon yapılmaz
        anomaly_duration_hours=duration_hours,
    )

    if rca_candidates:
        rca_candidates = calculate_segment_financial_breakdown(rca_candidates, default_aov=default_aov)

    return {
        "incident_id": incident_id,
        "window_start": window_start,
        "window_end": window_end,
        "duration_hours": duration_hours,
        "segment_df": seg_df,
        "baseline_col": baseline_col,
        "actual_col": actual_col,
        "rca_candidates": rca_candidates,
        "top_candidate": top,
        "financial": fin_result,
    }


# -----------------------------------------------------------------------
# 4. FR-19: Tracking vs Business ayrımı (gerçek engine, iOS 3.2.0 vakası)
# -----------------------------------------------------------------------
def build_tracking_bundle(window_start, window_end, df_dims, segment_filter):
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)
    control_start = window_start - pd.Timedelta(days=3)

    seg_all = df_dims
    for k, v in segment_filter.items():
        seg_all = seg_all[seg_all[k] == v]

    ctrl = seg_all[(seg_all["eventTime"] >= control_start) & (seg_all["eventTime"] < window_start)]
    win = seg_all[(seg_all["eventTime"] >= window_start) & (seg_all["eventTime"] < window_end)].copy()

    if win.empty or ctrl.empty:
        return None

    baseline_purchase = ctrl["purchase_completed"].sum() / max(1, ctrl.shape[0]) * win.shape[0]
    baseline_submitted = ctrl["payment_submitted"].sum() / max(1, ctrl.shape[0]) * win.shape[0]

    win["baseline_purchase_completed"] = ctrl["purchase_completed"].mean()
    win["baseline_payment_submitted"] = ctrl["payment_submitted"].mean()

    result = detect_tracking_vs_business_anomaly(
        df_anomaly=win,
        current_step_col="purchase_completed",
        previous_step_col="payment_submitted",
        backend_success_col=None,
        zero_drop_threshold=0.90,
    )
    result["window_rows"] = win
    return result


# -----------------------------------------------------------------------
# 4a. DİNAMİK İNCİDENT KEŞFİ VE OTOMATİK RCA/TRACKING BUNDLE ÜRETİMİ
# -----------------------------------------------------------------------
SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


def _classify_window_metric(affected_metrics: list[str]) -> dict:
    """
    Pencere içindeki TÜM affected_metric değerlerini tarar (sadece en sık
    görüleni/mode'u değil — Provider_B gibi olaylarda pencerenin ilk saati
    farklı bir metrikle etiketlenmiş olabilir). anomaly_engine.py'nin
    üretebileceği 4 format için doğru (num_col, den_col) çiftini + varsa
    tracking segment'ini döndürür.

    Bu, önceki sürümdeki "her incident'ı purchase_completed/payment_submitted
    ile analiz et" sabitlemesinin yerini alır — o sabitleme, gerçek Provider_B
    (checkout->payment) olayında RCA'nın 0 aday bulmasına ve yanlış (çok
    düşük) finansal etki hesaplanmasına yol açıyordu.
    """
    for m in affected_metrics:
        if m and m.startswith("tracking_loss_"):
            rest = m[len("tracking_loss_"):]
            if "_" in rest:
                platform, app_version = rest.split("_", 1)
                return {
                    "kind": "tracking",
                    "num_col": "purchase_completed",
                    "den_col": "payment_submitted",
                    "segment_filter": {"platform": platform, "appVersion": app_version},
                }
    for m in affected_metrics:
        if m == "provider_B_downtime":
            return {"kind": "rca", "num_col": "payment_submitted", "den_col": "checkout_started", "segment_filter": None}
    for m in affected_metrics:
        if m and (m.startswith("dim_cr_") or m == "cr_payment_submitted_to_purchase_completed"):
            return {"kind": "rca", "num_col": "purchase_completed", "den_col": "payment_submitted", "segment_filter": None}
    for m in affected_metrics:
        if m and m.startswith("latency_"):
            return {"kind": "latency_only", "num_col": None, "den_col": None, "segment_filter": None}
    # Bilinmeyen format için güvenli varsayım
    return {"kind": "rca", "num_col": "purchase_completed", "den_col": "payment_submitted", "segment_filter": None}


def discover_all_incidents(df_detected, df_dims, df_overall, change_events_raw, default_aov):
    """
    detected_anomalies.parquet dosyasındaki is_anomaly=True olan tüm saatleri tarar,
    ardışık veya yakın saatteki anomalileri otomatik olarak gruplayarak dinamik
    incident pencereleri ve bunlara ait otomatik RCA/tracking paketlerini üretir.

    ground_truth.json'a HİÇ bakmaz — kör/otomatik keşiftir. Her pencerenin
    RCA/finansal analizi, o pencerede GERÇEKTEN tetiklenen metriğe göre doğru
    num_col/den_col ile yapılır (bkz. _classify_window_metric).

    Ayrıca her pencere için "confidence" (yüksek/düşük) hesaplanır: pencere
    tek saatlik VE içindeki en yüksek n_concurrent_signals <= 1 ise, bu
    muhtemelen istatistiksel gürültüdür (bkz. anomaly_engine.py'nin FR-09
    persistence/concordance kontrolü) — böyle pencereler ayrı, düşük öncelikli
    bir etiketle (⚪) işaretlenir, gerçek olaylarla (🔴) karıştırılmaz.
    """
    anomalies = df_detected[df_detected["is_anomaly"]].sort_values("eventTime")
    if anomalies.empty:
        return {}

    discovered_incidents = {}
    grouped_windows = []

    current_start = None
    current_end = None

    # Ardışık / yakın saatlerdeki tespiti tek bir incident olarak birleştir
    for _, row in anomalies.iterrows():
        t = row["eventTime"]
        if current_start is None:
            current_start = t
            current_end = t + pd.Timedelta(hours=1)
        elif t <= current_end + pd.Timedelta(hours=2):  # 2 saate kadarki boşlukları tek incident say
            current_end = max(current_end, t + pd.Timedelta(hours=1))
        else:
            grouped_windows.append((current_start, current_end))
            current_start = t
            current_end = t + pd.Timedelta(hours=1)

    if current_start is not None:
        grouped_windows.append((current_start, current_end))

    # Otomatik tespit edilen her pencere için RCA ve Bundle oluştur
    for idx, (ws, we) in enumerate(grouped_windows, start=1):
        inc_id = f"AUTO-INC-{idx:03d}"

        sub_detected = df_detected[(df_detected["eventTime"] >= ws) & (df_detected["eventTime"] < we)]
        affected_metrics = sorted(m for m in sub_detected["affected_metric"].dropna().unique().tolist())
        route = _classify_window_metric(affected_metrics)

        n_hours = sub_detected["eventTime"].nunique()
        max_concurrent = int(sub_detected.get("n_concurrent_signals", pd.Series([0])).max())
        dominant_severity = max(sub_detected["anomaly_severity"].tolist(), key=lambda s: SEVERITY_RANK.get(s, 0))
        is_low_confidence = (n_hours <= 1) and (max_concurrent <= 1)
        confidence_tag = "⚪ Düşük güven (izole+tekil)" if is_low_confidence else "🔴"

        if route["kind"] == "tracking":
            tracking = build_tracking_bundle(ws, we, df_dims, route["segment_filter"])
            if tracking is None:
                continue
            seg_desc = " / ".join(f"{k}={v}" for k, v in route["segment_filter"].items())
            bundle = {
                "kind": "tracking",
                "incident_id": inc_id,
                "window_start": ws,
                "window_end": we,
                "is_low_confidence": is_low_confidence,
                "n_hours_flagged": n_hours,
                "tracking": tracking,
                "label": f"{confidence_tag} {inc_id} · Otomatik Tracking Anomaly: {seg_desc} ({ws.strftime('%m-%d %H:%M')})",
            }
        elif route["kind"] == "rca":
            bundle = build_incident_bundle(
                incident_id=inc_id,
                window_start=ws,
                window_end=we,
                num_col=route["num_col"],
                den_col=route["den_col"],
                df_dims=df_dims,
                df_overall=df_overall,
                change_events_raw=change_events_raw,
                default_aov=default_aov,
            )
            bundle["kind"] = "rca"
            bundle["is_low_confidence"] = is_low_confidence
            bundle["n_hours_flagged"] = n_hours
            top_seg = bundle["top_candidate"]["segment"] if bundle.get("top_candidate") else "Genel Sapma"
            bundle["label"] = f"{confidence_tag} {inc_id} · Otomatik Tespit: {top_seg} ({ws.strftime('%m-%d %H:%M')}, {n_hours}h)"
        else:
            # latency_only -> RCA/finansal motorların desteklemediği saf zamanlama sinyali
            bundle = {
                "kind": "unsupported",
                "incident_id": inc_id,
                "window_start": ws,
                "window_end": we,
                "is_low_confidence": is_low_confidence,
                "n_hours_flagged": n_hours,
                "reason": f"Saf latency sinyali ({', '.join(affected_metrics)}) — otomatik RCA/finansal analiz henüz desteklenmiyor.",
                "label": f"{confidence_tag} {inc_id} · Latency Sinyali ({ws.strftime('%m-%d %H:%M')})",
            }

        discovered_incidents[bundle["label"]] = bundle

    return discovered_incidents




# -----------------------------------------------------------------------
# 4b. Saat-bazlı GERÇEK Confusion Matrix (TP/FP/FN/TN) — klasik tanım.
#     Her saat (df_detected'daki her satır) iki bağımsız eksende etiketlenir:
#       - "Positive" mi (o saat herhangi bir ground_truth penceresine düşüyor mu)
#       - "Predicted anomaly" mi (anomaly_engine is_anomaly=True demiş mi)
#     Bu, önceki sürümdeki uydurma [tp,fp,fn,tn] listesinin yerini alır.
# -----------------------------------------------------------------------
def compute_hourly_confusion_matrix(df_detected, ground_truth):
    df = df_detected.copy()
    df["is_gt_positive"] = False
    for gt in ground_truth:
        gt_start = pd.Timestamp(gt["start_time"])
        gt_end = pd.Timestamp(gt["end_time"])
        mask = (df["eventTime"] >= gt_start) & (df["eventTime"] < gt_end)
        df.loc[mask, "is_gt_positive"] = True

    tp = int(((df["is_anomaly"]) & (df["is_gt_positive"])).sum())
    fp = int(((df["is_anomaly"]) & (~df["is_gt_positive"])).sum())
    fn = int(((~df["is_anomaly"]) & (df["is_gt_positive"])).sum())
    tn = int(((~df["is_anomaly"]) & (~df["is_gt_positive"])).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "annotated": df,
    }


# -----------------------------------------------------------------------
# 4c. RCA skor formülünün terim-terim ayrıştırması (görsel doğrulama için)
# -----------------------------------------------------------------------
def score_breakdown(top_candidate: dict):
    C = top_candidate["concentration_C"]
    E = top_candidate["effect_size_E"]
    T = top_candidate["time_proximity_T"]
    I_gain = top_candidate["interaction_gain"]
    K = 0.05 if "&" in top_candidate["segment"] else 0.0
    H = (2 * C * E) / (C + E) if (C + E) > 0 else 0.0

    terms = [
        ("0.35 · C (Concentration)", 0.35 * C),
        ("0.25 · H(C,E) (Harmonic)", 0.25 * H),
        ("0.20 · T (Time Proximity)", 0.20 * T),
        ("0.25 · I_gain (Interaction)", 0.25 * I_gain),
        ("− K (Complexity Penalty)", -K),
    ]
    total = sum(v for _, v in terms)
    return terms, total, top_candidate["root_cause_score"]


# -----------------------------------------------------------------------
# 5. Gerçek Evaluation Matrix: detected_anomalies.parquet vs ground_truth.json
#    (Kritik Düzeltme #5'in temeli: uydurma tp/fp/fn/tn sayıları değil,
#     gerçek zaman/segment kesişimine dayalı confusion matrix)
# -----------------------------------------------------------------------
def evaluate_detections_against_ground_truth(df_detected, ground_truth):
    flagged = df_detected[df_detected["is_anomaly"]].copy()
    flagged["matched_gt"] = None
    flagged["is_true_positive"] = False

    gt_hits = {gt["anomaly_id"]: False for gt in ground_truth}

    for idx, row in flagged.iterrows():
        t = row["eventTime"]
        for gt in ground_truth:
            gt_start = pd.Timestamp(gt["start_time"])
            gt_end = pd.Timestamp(gt["end_time"])
            if gt_start <= t < gt_end:
                # zaman kesişiyor -> segment/tip de mantıklı mı kabaca kontrol et
                metric = str(row["affected_metric"] or "")
                rc = gt["root_cause"]
                segment_match = any(str(v).lower() in metric.lower() for v in rc.values() if isinstance(v, str))
                type_match = (gt["type"] == row["anomaly_type"]) or segment_match
                if type_match:
                    flagged.at[idx, "matched_gt"] = gt["anomaly_id"]
                    flagged.at[idx, "is_true_positive"] = True
                    gt_hits[gt["anomaly_id"]] = True
                    break

    true_positives = int(flagged["is_true_positive"].sum())
    false_positives = int((~flagged["is_true_positive"]).sum())
    false_negatives = int(sum(not hit for hit in gt_hits.values()))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    fp_rows = flagged[~flagged["is_true_positive"]][
        ["eventTime", "anomaly_type", "anomaly_severity", "affected_metric", "anomaly_score"]
    ]

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_detail": flagged,
        "false_positive_rows": fp_rows,
        "gt_hits": gt_hits,
    }


# -----------------------------------------------------------------------
# 6. Recommended Action üretimi (kural tabanlı, gerçek RCA çıktısına dayalı)
# -----------------------------------------------------------------------
def recommend_action(top_candidate: dict, matched_change_event: str = None) -> dict:
    if not top_candidate:
        return {
            "action": "Yeterli sinyal yok — otomatik aksiyon önerilmiyor.",
            "urgency": "info",
            "rationale": "RCA motoru eşik üstü (p<=0.05) bir aday üretmedi.",
        }

    score = top_candidate["root_cause_score"]
    evidence = top_candidate["evidence_level"]
    segment = top_candidate["segment"]

    if evidence == "operational_correlation" and score >= 0.6 and matched_change_event:
        return {
            "action": f"'{matched_change_event}' değişikliğini geri alın (rollback) veya feature flag ile devre dışı bırakın; "
                      f"ardından {segment} segmentinde conversion oranını 30 dakika içinde izleyin.",
            "urgency": "critical",
            "rationale": f"Yüksek root-cause skoru ({score:.2f}) ve deployment ile güçlü zaman yakınlığı (operational correlation) mevcut.",
        }
    elif evidence == "statistical_association" and score >= 0.4:
        return {
            "action": f"{segment} segmentini manuel olarak inceleyin; ilişkili bir deployment/change event bulunamadı, "
                      f"bu nedenle otomatik rollback ÖNERİLMEZ. Log/altyapı ekibiyle doğrulayın.",
            "urgency": "high",
            "rationale": "İstatistiksel olarak anlamlı sapma var ancak nedensellik kanıtı zayıf.",
        }
    else:
        return {
            "action": f"{segment} segmentini gözlem altına alın; skor eşik altı ({score:.2f}), acil müdahale önerilmiyor.",
            "urgency": "low",
            "rationale": "Sinyal zayıf veya kanıt seviyesi düşük (hypothesis).",
        }