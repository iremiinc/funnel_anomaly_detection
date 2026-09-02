import pandas as pd
import numpy as np
from pathlib import Path

FUNNEL_STEPS = [
    "product_viewed",
    "add_to_cart",
    "checkout_started",
    "payment_submitted",
    "purchase_completed"
]

# Aynı saatte birden fazla koşul (örn. hem conversion hem data_quality) aynı
# severity ile tetiklenirse, hangisinin "primary" seçileceğini artık EKLENİŞ
# SIRASI değil, bu öncelik belirliyor. data_quality_anomaly her zaman kazanır
# çünkü d_comp==0 zaten d_cr==0<0.25 anlamına gelir (ikisi de tetiklenir) ve
# FR-19 gereği daha spesifik/aksiyon-alınabilir tanı olan tracking kaybı
# genel conversion anomalisi etiketinin arkasında kaybolmamalı.
TYPE_PRIORITY = {
    "data_quality_anomaly": 3,
    "conversion_latency_anomaly": 2,
    "conversion_anomaly": 1,
}
SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
SEVERITY_DOWNGRADE = {"critical": "low", "high": "low", "medium": "low", "low": "low"}


def detect_and_classify_anomalies(
    df_overall: pd.DataFrame, 
    df_dims: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Hem genel (overall) hem de boyut (dimension) bazlı metrikler üzerinde 
    anomali tespiti gerçekleştirir.

    FR-09 (False-positive control) — iki hafif kontrol eklendi:
      1) Concordance: aynı saatte kaç bağımsız koşul (segment/metrik) birden
         tetiklendi (`n_concurrent_signals`).
      2) Persistence: bu saat, önceki/sonraki saatte de anomali var mı yoksa
         tek başına mı duruyor (`is_isolated_hour`).
    Bir tespit hem izole HEM de tek bir sinyale dayanıyorsa (n_concurrent<=1),
    severity otomatik olarak "low"a düşürülüyor ve `anomaly_confidence`
    "muhtemel gürültü" olarak işaretleniyor — is_anomaly=True kalmaya devam
    ediyor (sistem hâlâ görüyor/raporluyor, hiçbir şeyi gizlemiyoruz), sadece
    gerçek, süregelen olaylarla aynı görsel ağırlığı almıyor.
    """
    df = df_overall.copy()
    
    # Anomali Etiket Kolonları
    df['is_anomaly'] = False
    df['anomaly_type'] = "normal"
    df['anomaly_severity'] = "none"
    df['affected_metric'] = None
    df['anomaly_score'] = 0.0
    df['anomaly_confidence'] = "n/a"
    df['n_concurrent_signals'] = 0

    # Saatlik Zaman İndeksleri
    time_points = df['eventTime'].unique()

    for t in time_points:
        idx_list = df[df['eventTime'] == t].index
        if len(idx_list) == 0:
            continue
        idx = idx_list[0]
        row = df.loc[idx]
        anomalies_found = []

        # -----------------------------------------------------------------
        # 1. Conversion Anomaly Detection (Genel & Segment Bazlı)
        # -----------------------------------------------------------------
        target_cr = row.get('cr_payment_submitted_to_purchase_completed', 0.0)
        expected_cr = row.get('baseline_expected', 0.0)
        lower_cr = row.get('baseline_lower_bound', 0.0)
        
        # Genel Seviye Kontrol
        if lower_cr > 0 and target_cr < lower_cr:
            drop_ratio = (expected_cr - target_cr) / max(expected_cr, 1e-5)
            if drop_ratio >= 0.20:
                anomalies_found.append({
                    "type": "conversion_anomaly",
                    "severity": "critical" if drop_ratio >= 0.40 else "high",
                    "metric": "cr_payment_submitted_to_purchase_completed",
                    "score": round(drop_ratio * 100, 2)
                })

        # Dimension Seviyesi Kontrol (Örn: Android 5.4.2 çöküşü)
        if df_dims is not None and not df_dims.empty:
            dim_window = df_dims[df_dims['eventTime'] == t]
            for _, d_row in dim_window.iterrows():
                d_submitted = d_row.get('payment_submitted', 0)
                d_completed = d_row.get('purchase_completed', 0)
                if d_submitted >= 10:
                    d_cr = d_completed / d_submitted
                    # Beklenen normal conversion (~%65) altı çöküş kontrolü
                    if d_cr < 0.25:
                        anomalies_found.append({
                            "type": "conversion_anomaly",
                            "severity": "critical",
                            "metric": f"dim_cr_{d_row.get('platform')}_{d_row.get('appVersion')}",
                            "score": round((1 - d_cr) * 100, 2)
                        })

        # -----------------------------------------------------------------
        # 2. Volume & Latency Anomaly Detection (Provider_B vb. Kesintiler)
        # -----------------------------------------------------------------
        # Latency kontrolü (checkout -> payment veya payment -> purchase)
        latency = row.get('latency_checkout_started_to_payment_submitted', 0.0)
        if pd.notna(latency) and latency > 120: # 120 sn üzeri anormal yavaşlama
            anomalies_found.append({
                "type": "conversion_latency_anomaly",
                "severity": "high",
                "metric": "latency_checkout_started_to_payment_submitted",
                "score": float(latency)
            })

        # Dimension Bazlı Provider / Volume Kesintisi (ANO-002)
        if df_dims is not None and not df_dims.empty:
            dim_window = df_dims[df_dims['eventTime'] == t]
            for _, d_row in dim_window.iterrows():
                # Provider_B dönüşüm çöküşü / Latency yükselmesi
                if d_row.get('paymentProvider') == 'Provider_B':
                    p_sub = d_row.get('payment_submitted', 0)
                    p_view = d_row.get('checkout_started', 0)
                    if p_view > 10 and (p_sub / max(p_view, 1)) < 0.30:
                        anomalies_found.append({
                            "type": "conversion_latency_anomaly",
                            "severity": "critical",
                            "metric": "provider_B_downtime",
                            "score": 90.0
                        })

        # -----------------------------------------------------------------
        # 3. Data Quality Anomaly Detection (iOS 3.2.0 Tracking Loss)
        # -----------------------------------------------------------------
        if df_dims is not None and not df_dims.empty:
            dim_window = df_dims[df_dims['eventTime'] == t]
            for _, d_row in dim_window.iterrows():
                # Belirli bir versiyonda ödeme var ama purchase event'i %100 kayıp
                d_sub = d_row.get('payment_submitted', 0)
                d_comp = d_row.get('purchase_completed', 0)
                
                if d_sub >= 15 and d_comp == 0:
                    anomalies_found.append({
                        "type": "data_quality_anomaly",
                        "severity": "critical",
                        "metric": f"tracking_loss_{d_row.get('platform')}_{d_row.get('appVersion')}",
                        "score": 100.0
                    })

        # -----------------------------------------------------------------
        # Önceliklendirme ve Karar
        # -----------------------------------------------------------------
        if anomalies_found:
            primary = sorted(
                anomalies_found,
                key=lambda x: (
                    SEVERITY_RANK.get(x['severity'], 0),
                    TYPE_PRIORITY.get(x['type'], 0),
                ),
                reverse=True,
            )[0]

            df.at[idx, 'is_anomaly'] = True
            df.at[idx, 'anomaly_type'] = primary['type']
            df.at[idx, 'anomaly_severity'] = primary['severity']
            df.at[idx, 'affected_metric'] = primary['metric']
            df.at[idx, 'anomaly_score'] = primary['score']
            df.at[idx, 'n_concurrent_signals'] = len(anomalies_found)

    # -----------------------------------------------------------------------
    # FR-09 İKİNCİ GEÇİŞ: Persistence kontrolü (izole mi, sürekli mi?)
    # Tek geçişte komşu saatlere bakılamıyor (henüz işlenmemiş olabilirler),
    # bu yüzden tüm saatler etiketlendikten SONRA ayrı bir geçişte kontrol
    # ediyoruz.
    # -----------------------------------------------------------------------
    df = df.sort_values('eventTime').reset_index(drop=True)
    flagged_times = set(df.loc[df['is_anomaly'], 'eventTime'])
    freq = pd.Timedelta(hours=1)

    for idx in df.index[df['is_anomaly']]:
        t = df.at[idx, 'eventTime']
        is_isolated = (t - freq) not in flagged_times and (t + freq) not in flagged_times
        n_signals = df.at[idx, 'n_concurrent_signals']

        if is_isolated and n_signals <= 1:
            # Tek saat + tek sinyal = ardışık-saat teyidi de yok, çoklu-segment
            # doğrulaması da yok -> muhtemel istatistiksel gürültü.
            df.at[idx, 'anomaly_severity'] = SEVERITY_DOWNGRADE.get(df.at[idx, 'anomaly_severity'], "low")
            df.at[idx, 'anomaly_confidence'] = "düşük (izole saat + tek segment — muhtemel gürültü)"
        else:
            reason_bits = []
            if not is_isolated:
                reason_bits.append("ardışık/komşu saatte de tespit var")
            if n_signals > 1:
                reason_bits.append(f"{n_signals} segment eşzamanlı tetiklendi")
            df.at[idx, 'anomaly_confidence'] = "yüksek (" + ", ".join(reason_bits) + ")"

    return df

if __name__ == "__main__":
    df_baseline = pd.read_parquet("funnel_metrics_with_baseline.parquet")
    df_dims = pd.read_parquet("funnel_metrics_by_dimensions.parquet") if Path("funnel_metrics_by_dimensions.parquet").exists() else None
    
    df_detected = detect_and_classify_anomalies(df_baseline, df_dims)
    df_detected.to_parquet("detected_anomalies.parquet", index=False)
    print("Anomali tespiti başarıyla tamamlandı!")