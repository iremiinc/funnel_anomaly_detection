import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from pipeline import (
    load_raw_artifacts,
    get_real_aov,
    build_incident_bundle,
    build_tracking_bundle,
    discover_all_incidents,
    evaluate_detections_against_ground_truth,
    recommend_action,
    compute_hourly_confusion_matrix,
    score_breakdown,
)

st.set_page_config(page_title="Enterprise Funnel Anomaly & RCA Command Center", layout="wide")

# =============================================================================
# 0. GERÇEK VERİ + GERÇEK ENGINE'LERİ ÇALIŞTIRMA (Kritik Düzeltme #1)
#    -> app.py artık RCA/finansal/tracking mantığını KENDİ İÇİNDE
#       yeniden yazmıyor; sadece pipeline.py üzerinden gerçek engine
#       fonksiyonlarını (rca_engine, financial_impact_engine,
#       tracking_anomaly_detector, anomaly_engine) çağırıyor.
# =============================================================================

STATIC_INCIDENT_DEFS = {
    "ANO-001 · Android 5.4.2 Payment Crash": dict(
        incident_id="ANO-001",
        window_start="2026-08-06 14:00:00",
        window_end="2026-08-06 22:00:00",
        num_col="purchase_completed",
        den_col="payment_submitted",
        kind="rca",
    ),
    "ANO-002 · Provider_B Infra Outage": dict(
        incident_id="ANO-002",
        window_start="2026-08-10 10:00:00",
        window_end="2026-08-10 14:00:00",
        num_col="payment_submitted",
        den_col="checkout_started",
        kind="rca",
    ),
    "ANO-003 · iOS 3.2.0 Tracking Loss": dict(
        incident_id="ANO-003",
        window_start="2026-08-13 18:00:00",
        window_end="2026-08-14 00:00:00",
        kind="tracking",
        segment_filter={"platform": "ios", "appVersion": "3.2.0"},
    ),
}


@st.cache_data(show_spinner="Gerçek veri seti ve engine'ler yükleniyor...")
def get_base_data():
    df_overall, df_dims, df_detected, change_events_raw, ground_truth = load_raw_artifacts()
    aov = get_real_aov(df_overall)
    return df_overall, df_dims, df_detected, change_events_raw, ground_truth, aov


@st.cache_data(show_spinner="Otomatik Incident Keşif Motoru Çalıştırılıyor...")
def get_all_incident_bundles():
    df_overall, df_dims, df_detected, change_events_raw, ground_truth, aov = get_base_data()
    
    # 1. Statik incident paketlerini oluştur
    bundles = {}
    for label, cfg in STATIC_INCIDENT_DEFS.items():
        if cfg["kind"] == "rca":
            b = build_incident_bundle(
                cfg["incident_id"], cfg["window_start"], cfg["window_end"],
                cfg["num_col"], cfg["den_col"],
                df_dims, df_overall, change_events_raw, aov,
            )
            b["kind"] = "rca"
        else:
            tracking = build_tracking_bundle(
                cfg["window_start"], cfg["window_end"], df_dims, cfg["segment_filter"]
            )
            b = {
                "kind": "tracking",
                "incident_id": cfg["incident_id"],
                "window_start": pd.Timestamp(cfg["window_start"]),
                "window_end": pd.Timestamp(cfg["window_end"]),
                "tracking": tracking,
            }
        bundles[label] = b

    # 2. Dinamik olarak tespit edilen incident'ları pipeline.py üzerinden keşfet ve ekle
    discovered = discover_all_incidents(df_detected, df_dims, df_overall, change_events_raw, aov)
    bundles.update(discovered)
    
    return bundles


df_overall, df_dims, df_detected, change_events_raw, ground_truth, REAL_AOV = get_base_data()
all_bundles = get_all_incident_bundles()
eval_result = evaluate_detections_against_ground_truth(df_detected, ground_truth)

st.title("🚨 Enterprise Funnel Anomaly & RCA Command Center")

incident_label = st.selectbox("İncelenecek Incident (Statik & Otomatik Keşfedilenler)", list(all_bundles.keys()), index=0)
bundle = all_bundles[incident_label]

st.markdown("---")

if bundle["kind"] == "unsupported":
    st.warning(f"**{bundle['incident_id']}** için otomatik analiz desteklenmiyor.\n\n{bundle['reason']}")
    st.stop()

if bundle.get("is_low_confidence"):
    st.warning(
        f"⚪ **Düşük güven uyarısı** — bu tespit tek bir saatte, tek bir segmentte tetiklendi; "
        f"önceki/sonraki saatte teyit yok. FR-09 (false-positive control) sinyaline göre bu, "
        f"gerçek bir olaydan çok istatistiksel gürültü olabilir. Aşağıdaki finansal/RCA rakamları "
        f"buna göre temkinli değerlendirilmelidir."
    )

# =============================================================================
# 1️⃣ EXECUTIVE SUMMARY
# =============================================================================
st.header("1️⃣ Executive Summary")

if bundle["kind"] == "rca":
    top = bundle["top_candidate"]
    fin = bundle["financial"]
    ex1, ex2, ex3, ex4, ex5 = st.columns(5)
    ex1.metric("Anomaly Status", "🔴 CRITICAL" if top else "🟢 No Signal")
    ex2.metric("Kayıp Hacim (bu pencere)", f"{fin['total_lost_volume']:,}",
               f"{bundle['duration_hours']:.0f} saatlik pencere")
    ex3.metric("Revenue Impact (bu pencere)", f"₺{fin['total_revenue_impact']:,.2f}",
               f"AOV: ₺{REAL_AOV} (gerçek veriden)")
    ex4.metric("Etkilenen Kullanıcı", f"{fin['impacted_unique_users']:,}")
    if top:
        ex5.metric("Top Root Cause", top["segment"], f"Score: {top['root_cause_score']:.3f}")
    else:
        ex5.metric("Top Root Cause", "—")
else:
    tr = bundle["tracking"]
    ex1, ex2, ex3 = st.columns(3)
    ex1.metric("Anomaly Status", "🟡 TRACKING ERROR" if tr["is_tracking_error"] else "🔴 BUSINESS ISSUE")
    ex2.metric("Confidence", f"%{tr['confidence_score'] * 100:.1f}")
    ex3.metric("Current-step Drop", f"%{tr['metrics_summary']['current_step_drop_pct']:.1f}")
    st.info("Bu incident FR-19 tracking-vs-business ayrımına girdiği için finansal kaybı "
            "'kayıp sipariş' olarak değil 'kayıp telemetri' olarak değerlendirin — "
            "arka planda gerçek satış muhtemelen gerçekleşmiştir.")

st.markdown("---")

# =============================================================================
# 2️⃣ FUNNEL & 3️⃣ TIME SERIES (genel bağlam, tüm dönem)
# =============================================================================
col_funnel_sec, col_ts_sec = st.columns([1, 1])

with col_funnel_sec:
    st.header("2️⃣ Funnel Conversion Flow (incident penceresi)")
    ws, we = bundle["window_start"], bundle["window_end"]
    win_events = df_dims[(df_dims["eventTime"] >= ws - pd.Timedelta(days=3)) & (df_dims["eventTime"] < ws)]
    win_actual = df_dims[(df_dims["eventTime"] >= ws) & (df_dims["eventTime"] < we)]

    funnel_cols = ["product_viewed", "add_to_cart", "checkout_started", "payment_submitted", "purchase_completed"]
    # control ortalama saatlik hacmi pencere uzunluğuna ölçekle
    n_hours = max(1, win_actual["eventTime"].nunique())
    n_ctrl_hours = max(1, win_events["eventTime"].nunique())
    baseline_vals = [win_events[c].sum() / n_ctrl_hours * n_hours for c in funnel_cols]
    actual_vals = [win_actual[c].sum() for c in funnel_cols]

    fig_f = go.Figure()
    fig_f.add_trace(go.Funnel(name="Baseline (kontrol dönemi)", y=funnel_cols, x=baseline_vals,
                               textinfo="value+percent initial", marker={"color": "#2ECC71"}))
    fig_f.add_trace(go.Funnel(name="Actual (incident penceresi)", y=funnel_cols, x=actual_vals,
                               textinfo="value+percent initial", marker={"color": "#E74C3C"}))
    fig_f.update_layout(height=340, template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10),
                         legend=dict(orientation="h"))
    st.plotly_chart(fig_f, use_container_width=True)

with col_ts_sec:
    st.header("3️⃣ Time Series Anomaly Detection")
    ts_window_start = ws - pd.Timedelta(hours=24)
    ts_window_end = we + pd.Timedelta(hours=12)
    df_ts = df_overall[(df_overall["eventTime"] >= ts_window_start) & (df_overall["eventTime"] <= ts_window_end)]

    df_ts_detected = df_detected[
        (df_detected["eventTime"] >= ts_window_start) & (df_detected["eventTime"] <= ts_window_end)
        & (df_detected["is_anomaly"])
    ]

    severity_color = {"critical": "#E74C3C", "high": "#F39C12", "medium": "#F1C40F", "low": "#95A5A6"}

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df_ts["eventTime"], y=df_ts["baseline_expected"],
                                 name="Baseline (baseline_engine)", line=dict(color="gray", dash="dash")))
    fig_ts.add_trace(go.Scatter(x=df_ts["eventTime"], y=df_ts["cr_payment_submitted_to_purchase_completed"],
                                 name="Actual CR", line=dict(color="#E74C3C", width=3)))
    if not df_ts_detected.empty:
        fig_ts.add_trace(go.Scatter(
            x=df_ts_detected["eventTime"], y=df_ts_detected["cr_payment_submitted_to_purchase_completed"],
            mode="markers", name="anomaly_engine tespiti",
            marker=dict(size=12, symbol="x", color=[severity_color.get(s, "#E74C3C") for s in df_ts_detected["anomaly_severity"]],
                        line=dict(width=1, color="white")),
            text=df_ts_detected["affected_metric"],
            hovertemplate="%{x}<br>%{text}<br>CR=%{y:.3f}<extra></extra>",
        ))
    fig_ts.add_vrect(x0=ws, x1=we, fillcolor="red", opacity=0.10, line_width=0,
                      annotation_text="ground-truth pencere", annotation_position="top left")
    fig_ts.update_layout(height=340, template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h"))
    st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# =============================================================================
# 4️⃣ INCIDENT DISCOVERY CONSOLE (gerçek detected_anomalies.parquet)
# =============================================================================
st.header("4️⃣ Incident Discovery Console ")
flagged = eval_result["flagged_detail"]
inc_cols = st.columns(3)
for i, (label, cfg) in enumerate(STATIC_INCIDENT_DEFS.items()):
    ws_i, we_i = pd.Timestamp(cfg["window_start"]), pd.Timestamp(cfg["window_end"])
    hits = flagged[(flagged["eventTime"] >= ws_i) & (flagged["eventTime"] < we_i) & (flagged["is_true_positive"])]
    box = inc_cols[i % 3]
    if not hits.empty:
        top_hit = hits.iloc[0]
        box.error(f"**🚨 {cfg['incident_id']}**\n\n"
                  f"- **Type:** {top_hit['anomaly_type']}\n"
                  f"- **Severity:** {top_hit['anomaly_severity']}\n"
                  f"- **Detections in window:** {len(hits)}\n"
                  f"- **Status:** Detected by anomaly_engine ✅")
    else:
        box.warning(f"**⚠️ {cfg['incident_id']}**\n\nBu pencerede eşleşen gerçek tespit bulunamadı "
                     f"(recall açığı olabilir — bkz. Evaluation Matrix).")

st.markdown("---")

# =============================================================================
# 5️⃣ / 6️⃣ / 7️⃣ / 8️⃣  RCA DETAY BÖLÜMÜ (sadece kind == 'rca' incident'larda)
# =============================================================================
if bundle["kind"] == "rca":
    top = bundle["top_candidate"]
    col_left_engine, col_right_engine = st.columns([1, 1])

    with col_left_engine:
        st.header("6️⃣ FR-15 Hierarchical Drill-Down (gerçek segment verisi)")
        seg_df = bundle["segment_df"].copy()
        seg_df["drop"] = seg_df[bundle["baseline_col"]] - seg_df[bundle["actual_col"]]
        seg_df["drop"] = seg_df["drop"].clip(lower=0)
        seg_df_nonzero = seg_df[seg_df["drop"] > 0]
        if not seg_df_nonzero.empty:
            fig_sun = px.sunburst(
                seg_df_nonzero,
                path=["platform", "appVersion", "paymentProvider"],
                values="drop",
                color="drop",
                color_continuous_scale="Reds",
                title="Kayıp Hacmin Segment Hiyerarşisine Dağılımı (platform → appVersion → provider)",
            )
            fig_sun.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("Bu pencerede segment bazlı kayıp bulunamadı.")

        if top:
            with st.expander("Ham RCA çıktısı (rca_engine.run_production_rca)"):
                st.json({k: v for k, v in top.items() if k != "financial_impact"})

    with col_right_engine:
        st.header("7️⃣ FR-16 Dimension Interactions (Top adaylar)")
        cand_df = pd.DataFrame(bundle["rca_candidates"])[
            ["segment", "root_cause_score", "evidence_level", "concentration_C",
             "effect_size_E", "interaction_gain", "time_proximity_T", "p_value", "lost_volume"]
        ].head(8) if bundle["rca_candidates"] else pd.DataFrame()
        st.dataframe(cand_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # 9️⃣ Recommended Action (Kritik Düzeltme #5)
    # =========================================================================
    st.header("9️⃣ Recommended Action")
    rec = recommend_action(top, matched_change_event=top.get("matched_event") if top else None)
    urgency_box = {"critical": st.error, "high": st.warning, "low": st.info, "info": st.info}[rec["urgency"]]
    urgency_box(f"**Aksiyon:** {rec['action']}\n\n**Gerekçe:** {rec['rationale']}")

else:
    # =========================================================================
    # 9️⃣ FR-19 Tracking vs Business Detayı
    # =========================================================================
    st.header("9️⃣ FR-19 Tracking vs Business Anomaly (gerçek tracking_anomaly_detector çıktısı)")
    tr = bundle["tracking"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tracking Error?", "YES" if tr["is_tracking_error"] else "NO")
    c2.metric("Confidence", f"%{tr['confidence_score']*100:.1f}")
    c3.metric("Backend Mismatch", "YES" if tr["metrics_summary"]["backend_mismatch_detected"] else "NO")
    st.info(tr["reasoning"])
    st.json(tr["metrics_summary"])

    st.header("9️⃣.1 Recommended Action")
    st.warning("**Aksiyon:** SDK/telemetri ekibiyle iOS 3.2.0 client tarafındaki `purchase_completed` "
               "event tetikleyicisini inceleyin. Bu bir finansal kayıp değil, bir **veri kalitesi** "
               "sorunudur — finansal raporlama ve RCA motorlarını bu segment için geçici olarak "
               "backend/DB kaynaklı gerçek satış verisiyle besleyin.\n\n"
               "**Gerekçe:** Önceki adım (`payment_submitted`) sağlıklı seyrederken mevcut adımda "
               f"%{tr['metrics_summary']['current_step_drop_pct']:.0f} ani düşüş — klasik tracking-error imzası.")

st.markdown("---")

# =============================================================================
# 🔟 FINANCIAL IMPACT (sadece rca tipi incident'larda anlamlı)
# =============================================================================
if bundle["kind"] == "rca":
    st.header("🔟 Financial Impact")
    fin = bundle["financial"]
    f1, f2, f3 = st.columns(3)
    f1.metric("Total Loss (bu pencere)", f"₺{fin['total_revenue_impact']:,.2f}")
    f2.metric("Kayıp Hacim", f"{fin['total_lost_volume']:,}")
    f3.metric("Etkilenen Kullanıcı", f"{fin['impacted_unique_users']:,}")

st.markdown("---")

# =============================================================================
# 1️⃣1️⃣ EVALUATION MATRIX + FALSE POSITIVE INVESTIGATION (Kritik Düzeltme #5)
# =============================================================================
st.header("1️⃣1️⃣ Evaluation Matrix (Ground-Truth)")

cm = compute_hourly_confusion_matrix(df_detected, ground_truth)

cm_col, metric_col = st.columns([1, 1])

with cm_col:
    st.subheader("Confusion Matrix (saat-bazlı, tüm 14 günlük seri)")
    z = [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]]
    fig_cm = go.Figure(data=go.Heatmap(
        z=z,
        x=["Gerçek: Anomali (Positive)", "Gerçek: Normal (Negative)"],
        y=["Tahmin: Anomali", "Tahmin: Normal"],
        text=[[f"TP={cm['tp']}", f"FN={cm['fn']}"], [f"FP={cm['fp']}", f"TN={cm['tn']}"]],
        texttemplate="%{text}",
        textfont={"size": 18},
        colorscale="Blues",
        showscale=False,
    ))
    fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), template="plotly_dark")
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption(f"Toplam {cm['tp']+cm['fp']+cm['fn']+cm['tn']} saatlik gözlem "
               f"({len(ground_truth)} ground-truth incident × pencere uzunlukları) üzerinden hesaplandı.")

with metric_col:
    st.subheader("Metrikler")
    m1, m2 = st.columns(2)
    m1.metric("Precision", f"%{cm['precision']*100:.1f}")
    m2.metric("Recall", f"%{cm['recall']*100:.1f}")
    m3, m4 = st.columns(2)
    m3.metric("F1 Score", f"{cm['f1']:.3f}")
    m4.metric("FPR (False Positive Rate)", f"%{cm['fpr']*100:.2f}")
    st.caption(
        "F1 = 2·Precision·Recall / (Precision+Recall) — precision ve recall'ın dengesini tek "
        "sayıda özetler. FPR = FP / (FP+TN) — gerçek anomali olmayan saatlerin yüzde kaçının "
        "yanlışlıkla alarm ürettiğini gösterir; düşük olması istenir."
    )
    st.dataframe(
        pd.DataFrame([{
            "TP": cm["tp"], "FP": cm["fp"], "FN": cm["fn"], "TN": cm["tn"],
            "Precision": round(cm["precision"], 3), "Recall": round(cm["recall"], 3),
            "F1": round(cm["f1"], 3), "FPR": round(cm["fpr"], 4),
        }]),
        use_container_width=True, hide_index=True,
    )

st.subheader("🔍 False Positive Investigation")
fp_rows = eval_result["false_positive_rows"].copy()
if fp_rows.empty:
    st.success("Bu veri setinde eşleşmeyen (false positive) tespit bulunmadı.")
else:
    extra_cols = [c for c in ["anomaly_confidence", "n_concurrent_signals"] if c in df_detected.columns]
    if extra_cols:
        fp_rows = fp_rows.merge(df_detected[["eventTime"] + extra_cols], on="eventTime", how="left")
    st.dataframe(fp_rows, use_container_width=True, hide_index=True)
    if "anomaly_confidence" in fp_rows.columns:
        n_noise = fp_rows["anomaly_confidence"].str.startswith("düşük").sum()
        st.caption(
            f"Bunların **{n_noise}/{len(fp_rows)}** tanesi anomaly_engine.py'nin yeni persistence/concordance "
            f"kontrolüne göre zaten 'düşük güven' olarak işaretlenmiş durumda (severity otomatik 'low'a çekildi)."
        )