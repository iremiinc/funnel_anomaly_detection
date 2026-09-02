import pandas as pd
import numpy as np

def detect_tracking_vs_business_anomaly(
    df_anomaly: pd.DataFrame,
    current_step_col: str,
    previous_step_col: str,
    backend_success_col: str = None,
    zero_drop_threshold: float = 0.95
) -> dict:
    """
    FR-19: Tracking (Instrumentation) Hatası ile Gerçek İş Anomalisi Ayrımı Motoru
    
    Parametreler:
    - df_anomaly: Anomali penceresindeki veri
    - current_step_col: İncelenen adımla ilgili event sayısı (örn: purchase_completed_events)
    - previous_step_col: Bir önceki adımın event sayısı (örn: checkout_started_events)
    - backend_success_col: Varsa DB/Backend tarafındaki gerçek işlem sayısı
    - zero_drop_threshold: Event'in % kaç oranında bıçak gibi kesildiğini belirten eşik (%95+)
    """
    if df_anomaly.empty:
        return {"anomaly_type": "unknown", "confidence": 0.0}

    curr_actual = df_anomaly[current_step_col].sum()
    prev_actual = df_anomaly[previous_step_col].sum()
    
    # 1. Kontrol: Önceki adım normal akarken mevcut adım %95+ oranında sıfırlanmış mı?
    curr_drop_ratio = 1.0 - (curr_actual / max(1, df_anomaly[f"baseline_{current_step_col}"].sum()))
    prev_drop_ratio = 1.0 - (prev_actual / max(1, df_anomaly[f"baseline_{previous_step_col}"].sum()))
    
    is_sudden_zero = curr_drop_ratio >= zero_drop_threshold
    is_previous_step_healthy = prev_drop_ratio < 0.20  # Önceki adımda belirgin düşüş yok
    
    # 2. Kontrol: Backend / DB Verisi İle İstemci (Client) Event Uyuşmazlığı
    backend_mismatch = False
    if backend_success_col and backend_success_col in df_anomaly.columns:
        backend_actual = df_anomaly[backend_success_col].sum()
        # Backend'de işlemler var ama istemci event'i gelmiyorsa -> %100 Tracking Hatası
        if backend_actual > 0 and curr_actual == 0:
            backend_mismatch = True
        elif (backend_actual / max(1, curr_actual)) > 1.5:
            backend_mismatch = True

    # 3. Karar Mekanizması (Classification)
    if backend_mismatch or (is_sudden_zero and is_previous_step_healthy):
        anomaly_type = "tracking_instrumentation_error"
        confidence = 0.95 if backend_mismatch else 0.85
        reason = (
            "Önceki funnel adımları veya backend/DB kayıtları normal seyrederken, "
            "istemci tarafındaki telemetry event'lerinde aniden bıçak keskinliğinde düşüş tespit edildi. "
            "Bu durum bir iş krizinden ziyade SDK/Tracking hatasına işaret etmektedir."
        )
    else:
        anomaly_type = "business_operational_anomaly"
        confidence = 0.90
        reason = (
            "Düşüş sadece tek bir istemci event'i ile sınırlı kalmayıp "
            "funnel adımlarına ve sistem geneline oransal olarak yansımıştır. "
            "Gerçekleşen bir iş/sistem aksaklığı değerlendirilmektedir."
        )

    return {
        "anomaly_type": anomaly_type,
        "confidence_score": confidence,
        "is_tracking_error": (anomaly_type == "tracking_instrumentation_error"),
        "reasoning": reason,
        "metrics_summary": {
            "current_step_drop_pct": round(curr_drop_ratio * 100, 2),
            "previous_step_drop_pct": round(prev_drop_ratio * 100, 2),
            "backend_mismatch_detected": backend_mismatch
        }
    }