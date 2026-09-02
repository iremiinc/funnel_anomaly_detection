import itertools
import pandas as pd
import numpy as np
from scipy import stats

def check_event_relevance(segment_dict: dict, event: dict) -> bool:
    """
    FR-14: Deployment / Change Event'inin segmente ait olup olmadığını doğrular.
    """
    if not event or 'scope' not in event:
        return True  # Genel altyapı event'i (örn. Global DB upgrade) tüm segmentleri kapsar
    
    event_scope = event['scope']  # Örn: {"platform": "android", "app_version": "5.4.2"}
    for key, val in event_scope.items():
        if key in segment_dict and str(segment_dict[key]).lower() != str(val).lower():
            return False
    return True


def calculate_two_proportion_z_test(seg_exp: float, seg_act: float, total_exp: float, total_act: float) -> float:
    """
    Segment vs Segment Dışındakiler (Others) arasında İki Yönlü (Two-Tailed) Z-Testi uygular.
    seg_exp / seg_act: Başarı ve Deneme hacimleri
    """
    other_exp = total_exp - seg_exp
    other_act = total_act - seg_act
    
    # Dönüşmeyen / Başarısız Kayıp Hacimleri
    x1 = seg_exp - seg_act
    x2 = other_exp - other_act
    
    n1 = seg_exp
    n2 = other_exp
    
    if n1 <= 0 or n2 <= 0 or (n1 + n2) <= 0:
        return 1.0
    
    p1 = x1 / n1
    p2 = x2 / n2
    
    p_pooled = (x1 + x2) / (n1 + n2)
    
    se = np.sqrt(p_pooled * (1.0 - p_pooled) * ((1.0 / n1) + (1.0 / n2)))
    if se == 0:
        return 1.0
        
    z_stat = (p1 - p2) / se
    
    # Two-tailed p-value
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return float(p_value)


def classify_evidence_level(
    has_matched_event: bool, 
    p_val: float, 
    has_causal_experiment: bool = False
) -> str:
    """
    FR-17: Hipotezleri Şartnameye Tam Uyumlu Kanıt Seviyelerine Göre Etiketler.
    'observed_fact' RCA çıktısı değildir, Anomaly Detector doğrudan gözlemidir.
    """
    # 1. Kontrollü deney / Rollback doğrulaması varsa
    if has_causal_experiment:
        return "causal_evidence"
    
    # 2. Zaman olarak yakın deployment / release çakışması varsa
    if has_matched_event and (p_val <= 0.05):
        return "operational_correlation"
    
    # 3. İstatistiksel olarak anlamlı sapma varsa ama event eşleşmiyorsa
    if p_val <= 0.05:
        return "statistical_association"
    
    # 4. Zayıf / Henüz kanıtlanmamış hipotez
    return "hypothesis"


def run_production_rca(
    df_anomaly: pd.DataFrame, 
    baseline_col: str, 
    actual_col: str, 
    dimensions: list[str], 
    change_events: list[dict] = None, 
    anomaly_time=None, 
    min_cohort: int = 50,
    strong_signal_threshold: float = 0.15
) -> list[dict]:
    """
    FR-11, FR-13, FR-14, FR-15, FR-16 (Signal Pruned), FR-17 Tam Uyumlu RCA Motoru
    """
    total_expected = df_anomaly[baseline_col].sum()
    total_actual = df_anomaly[actual_col].sum()
    total_drop = total_expected - total_actual
    
    if total_drop <= 0:
        return []

    # -------------------------------------------------------------------------
    # 1. AŞAMA: 1D Tekil Boyutların İncelemesi ve Konsantrasyon (Signal) Haritası
    # -------------------------------------------------------------------------
    single_dim_metrics = {}
    
    for dim in dimensions:
        grouped = df_anomaly.groupby(dim)[[baseline_col, actual_col]].sum().reset_index()
        for _, row in grouped.iterrows():
            exp, act = row[baseline_col], row[actual_col]
            if exp >= min_cohort and (exp - act) > 0:
                seg_drop = exp - act
                c_val = seg_drop / total_drop
                single_dim_metrics[f"{dim}={row[dim]}"] = c_val

    # -------------------------------------------------------------------------
    # 2. AŞAMA: Arama Uzayını Oluşturma & FR-16 Strong Signal Filtrelemesi
    # -------------------------------------------------------------------------
    candidates = []
    
    # Tekil boyutlar
    search_dimensions = [[d] for d in dimensions]
    
    # FR-16: İkili kombinasyonlar sadece en az bir bileşeni güçlü sinyal veriyorsa eklenir
    for d1, d2 in itertools.combinations(dimensions, 2):
        # Sadece bu boyut çiftinde güçlü sinyal üreten değerler var mı kontrolü
        search_dimensions.append([d1, d2])

    for dims in search_dimensions:
        dims = list(dims)
        grouped = df_anomaly.groupby(dims)[[baseline_col, actual_col]].sum().reset_index()
        
        for _, row in grouped.iterrows():
            exp = row[baseline_col]
            act = row[actual_col]
            seg_drop = exp - act
            
            # --- FR-15: Minimum Cohort Kontrolü ---
            if exp < min_cohort or seg_drop <= 0:
                continue
            
            # --- FR-16: Combination Signal Pruning (Zayıf Sinyal Budama) ---
            if len(dims) == 2:
                parent_1 = f"{dims[0]}={row[dims[0]]}"
                parent_2 = f"{dims[1]}={row[dims[1]]}"
                sig1 = single_dim_metrics.get(parent_1, 0.0)
                sig2 = single_dim_metrics.get(parent_2, 0.0)
                
                # İki ebeveyn de güçlü sinyal eşiğinin altındaysa kombinasyonu pas geç
                if max(sig1, sig2) < strong_signal_threshold:
                    continue
            
            # --- İki Yönlü Z-Testi (Segment vs. Others) ---
            p_val = calculate_two_proportion_z_test(exp, act, total_expected, total_actual)
            
            if p_val > 0.05: # Güven aralığı dışındakileri ele
                continue
                
            segment_dict = {d: row[d] for d in dims}
            segment_key = " & ".join([f"{d}={row[d]}" for d in dims])
            
            # Metrik Hesapları
            C = seg_drop / total_drop
            E = seg_drop / exp
            
            # --- FR-16: Interaction Gain Hesabı ---
            interaction_gain = 0.0
            if len(dims) == 2:
                parent_1 = f"{dims[0]}={row[dims[0]]}"
                parent_2 = f"{dims[1]}={row[dims[1]]}"
                max_parent_c = max(single_dim_metrics.get(parent_1, 0.0), single_dim_metrics.get(parent_2, 0.0))
                interaction_gain = max(0.0, C - max_parent_c)

            # --- FR-14: Context-Aware Deployment/Change Event Proximity ---
            t_score = 0.0
            matched_event = None
            has_causal_exp = False
            
            if change_events and anomaly_time:
                for ev in change_events:
                    t_diff = (anomaly_time - ev['timestamp']).total_seconds() / 3600.0
                    if 0 <= t_diff <= 6:
                        if check_event_relevance(segment_dict, ev):
                            event_t_score = 1.0 / (1.0 + t_diff)
                            if event_t_score > t_score:
                                t_score = event_t_score
                                matched_event = ev['event_name']
                                # Eğer event doğrudan bir rollback/A-B test verisi içeriyorsa
                                if ev.get('is_causal_experiment', False):
                                    has_causal_exp = True

            # Complexity Penalty
            K = 0.05 if len(dims) == 2 else 0.0
            
            # Harmonik C x E Dengesi
            harmonic_ce = (2 * C * E) / (C + E) if (C + E) > 0 else 0.0
            
            # --- FR-13: Root Cause Score Formülü ---
            root_cause_score = (0.35 * C) + (0.25 * harmonic_ce) + (0.20 * t_score) + (0.25 * interaction_gain) - K

            # --- FR-17: Evidence Classification ---
            evidence_level = classify_evidence_level(
                has_matched_event=bool(matched_event),
                p_val=p_val,
                has_causal_experiment=has_causal_exp
            )

            # Metinsel Açıklama Üretimi (FR-17 İnsani Anlatım Standardı)
            description = f"{segment_key} segmentinde istatistiksel sapma (C={C:.2f}, p={p_val:.4f}) tespit edildi."
            if matched_event:
                description += f" Bu durum '{matched_event}' olayı ile operasyonel korelasyon (T={t_score:.2f}) göstermektedir."
            else:
                description += " Doğrudan ilişkili operasyonel bir deployment bulunamadı (Neden-sonuç kanıtı henüz mevcut değil)."

            candidates.append({
                "segment": segment_key,
                "root_cause_score": round(root_cause_score, 4),
                "evidence_level": evidence_level,
                "description": description,
                "concentration_C": round(C, 4),
                "effect_size_E": round(E, 4),
                "interaction_gain": round(interaction_gain, 4),
                "time_proximity_T": round(t_score, 4),
                "p_value": round(p_val, 4),
                "sample_size_n": int(exp),
                "lost_volume": int(seg_drop),
                "matched_event": matched_event
            })

    return sorted(candidates, key=lambda x: x['root_cause_score'], reverse=True)