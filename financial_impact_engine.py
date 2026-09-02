import pandas as pd
import numpy as np

def calculate_financial_impact(
    df_anomaly: pd.DataFrame,
    baseline_col: str,
    actual_col: str,
    default_aov: float = 250.0,
    aov_col: str = None,
    user_id_col: str = None,
    is_active_anomaly: bool = False,
    anomaly_duration_hours: float = 1.0
) -> dict:
    """
    FR-20: Ciro ve Finansal Etki Hesabı Motoru
    
    Parametreler:
    - df_anomaly: Anomali penceresindeki veri (DataFrame)
    - baseline_col: Beklenen hacim/dönüşüm sayısı kolonu
    - actual_col: Gerçekleşen hacim/dönüşüm sayısı kolonu
    - default_aov: Ortalama Sepet Değeri (Average Order Value - TL/USD)
    - aov_col: Eğer veri setinde segment bazlı dinamik AOV varsa bu kolon kullanılır
    - user_id_col: Etkilenen benzersiz kullanıcı hesabı için kullanıcı ID kolonu
    - is_active_anomaly: Anomali hala devam ediyor mu? (Projeksiyon için)
    - anomaly_duration_hours: Anomalinin şu ana kadar geçen süresi (saat)
    """
    if df_anomaly.empty:
        return {
            "total_lost_volume": 0,
            "total_revenue_impact": 0.0,
            "impacted_unique_users": 0,
            "projected_24h_revenue_loss": 0.0
        }

    # 1. Hacimsel Kayıp Hesabı
    df_temp = df_anomaly.copy()
    df_temp['volume_drop'] = df_temp[baseline_col] - df_temp[actual_col]
    
    # Sadece negatif sapmaları (düşüşleri) kayıp olarak alıyoruz
    df_temp['volume_drop'] = df_temp['volume_drop'].apply(lambda x: max(0.0, x))
    total_lost_volume = int(df_temp['volume_drop'].sum())

    # 2. Finansal Ciro Kaybı Hesabı
    if aov_col and aov_col in df_temp.columns:
        # Dinamik AOV kullanımı
        df_temp['revenue_loss'] = df_temp['volume_drop'] * df_temp[aov_col]
        total_revenue_impact = float(df_temp['revenue_loss'].sum())
    else:
        # Sabit AOV kullanımı
        total_revenue_impact = float(total_lost_volume * default_aov)

    # 3. Etkilenen Benzersiz Kullanıcı Hesabı
    impacted_users = 0
    if user_id_col and user_id_col in df_temp.columns:
        # Başarısızlık yaşayan benzersiz kullanıcılar
        impacted_users = int(df_temp[df_temp['volume_drop'] > 0][user_id_col].nunique())
    else:
        # Kullanıcı ID yoksa kayıp hacim üzerinden kestirim
        impacted_users = int(total_lost_volume * 0.85) # Yaklaşık 1.15 işlem/kullanıcı varsayımı

    # 4. Projeksiyon Hesabı (24 Saatlik Risk Tahmini)
    projected_24h_loss = total_revenue_impact
    if is_active_anomaly and anomaly_duration_hours > 0:
        hourly_loss_rate = total_revenue_impact / anomaly_duration_hours
        projected_24h_loss = round(hourly_loss_rate * 24.0, 2)

    return {
        "total_lost_volume": total_lost_volume,
        "total_revenue_impact": round(total_revenue_impact, 2),
        "impacted_unique_users": impacted_users,
        "currency": "TRY",
        "is_active": is_active_anomaly,
        "projected_24h_revenue_loss": projected_24h_loss
    }


def calculate_segment_financial_breakdown(
    rca_candidates: list[dict],
    default_aov: float = 250.0
) -> list[dict]:
    """
    RCA'den çıkan kök neden adaylarının finansal etkilerini hesaplar ve ekler.
    """
    for candidate in rca_candidates:
        lost_vol = candidate.get("lost_volume", 0)
        candidate["financial_impact"] = {
            "estimated_revenue_loss": round(lost_vol * default_aov, 2),
            "lost_volume": lost_vol,
            "currency": "TRY"
        }
    return rca_candidates