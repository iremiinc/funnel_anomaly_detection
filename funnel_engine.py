import pandas as pd
import numpy as np

FUNNEL_STEPS = [
    "product_viewed",
    "add_to_cart",
    "checkout_started",
    "payment_submitted",
    "purchase_completed"
]

DIMENSIONS = [
    "platform",
    "appVersion",
    "osVersion",
    "country",
    "campaign",
    "paymentProvider"
]

def load_events(file_path: str = "events.parquet") -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df['eventTime'] = pd.to_datetime(df['eventTime'])
    return df

def aggregate_funnel_metrics(df: pd.DataFrame, freq: str = "1h", groupByDims: list = None) -> pd.DataFrame:
    """
    Belirli bir zaman aralığında (freq) ve boyutlarda (groupByDims) 
    funnel metriklerini hesaplar.
    """
    if groupByDims is None:
        groupByDims = []
    
    # Time window grouper
    time_grouper = pd.Grouper(key='eventTime', freq=freq)
    group_cols = [time_grouper] + groupByDims

    # 1. Adım bazlı tekil kullanıcı sayıları (Unique User Count per Step)
    step_counts = (
        df.groupby(group_cols + ['eventName'])['userId']
        .nunique()
        .unstack(fill_value=0)
    )

    # Eksik adım varsa 0 ile tamamla
    for step in FUNNEL_STEPS:
        if step not in step_counts.columns:
            step_counts[step] = 0
            
    # Adımları doğru sıraya diz
    step_counts = step_counts[FUNNEL_STEPS].reset_index()

    # 2. Gelir Hesabı (Purchase Completed)
    revenue_df = (
        df[df['eventName'] == 'purchase_completed']
        .groupby(group_cols)['revenue']
        .sum()
        .reset_index()
    )

    # Metrikleri birleştir
    metrics = pd.merge(step_counts, revenue_df, on=[col for col in group_cols if col != time_grouper] + ['eventTime'], how='left')
    metrics['revenue'] = metrics['revenue'].fillna(0.0)

    # 3. Conversion Rate ve Drop-off Hesaplamaları
    metrics['users_entering'] = metrics[FUNNEL_STEPS[0]]
    
    # Step-by-Step Conversions
    for i in range(1, len(FUNNEL_STEPS)):
        prev_step = FUNNEL_STEPS[i-1]
        curr_step = FUNNEL_STEPS[i]
        
        cr_col = f"cr_{prev_step}_to_{curr_step}"
        drop_col = f"drop_{prev_step}_to_{curr_step}"
        lost_col = f"lost_{prev_step}_to_{curr_step}"

        # 0'a bölünme koruması
        metrics[cr_col] = np.where(
            metrics[prev_step] > 0,
            metrics[curr_step] / metrics[prev_step],
            0.0
        )
        metrics[drop_col] = 1.0 - metrics[cr_col]
        metrics[lost_col] = np.maximum(0, metrics[prev_step] - metrics[curr_step])

    # Overall Funnel Conversion Rate
    metrics['overall_cr'] = np.where(
        metrics[FUNNEL_STEPS[0]] > 0,
        metrics[FUNNEL_STEPS[-1]] / metrics[FUNNEL_STEPS[0]],
        0.0
    )

    return metrics

def calculate_step_latencies(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Kullanıcı bazlı adımlar arası geçiş sürelerinin median değerini (Latency) hesaplar.
    """
    # Event'leri kullanıcı ve zamana göre sırala
    df_sorted = df.sort_values(['userId', 'eventTime'])
    
    # Kullanıcının bir sonraki adımındaki zamanı bul
    df_sorted['next_eventTime'] = df_sorted.groupby('userId')['eventTime'].shift(-1)
    df_sorted['next_eventName'] = df_sorted.groupby('userId')['eventName'].shift(-1)
    
    df_sorted['duration_sec'] = (df_sorted['next_eventTime'] - df_sorted['eventTime']).dt.total_seconds()
    
    # Sadece ardışık geçerli funnel adımları arasındaki geçişleri filtrele
    latencies = []
    for i in range(len(FUNNEL_STEPS) - 1):
        step_from = FUNNEL_STEPS[i]
        step_to = FUNNEL_STEPS[i+1]
        
        valid_transitions = df_sorted[
            (df_sorted['eventName'] == step_from) & 
            (df_sorted['next_eventName'] == step_to) &
            (df_sorted['duration_sec'] > 0) &
            (df_sorted['duration_sec'] < 3600) # 1 saat üzeri oturum kopmalarını ele
        ]
        
        latency_grouped = (
            valid_transitions.groupby(pd.Grouper(key='eventTime', freq=freq))['duration_sec']
            .median()
            .reset_index()
            .rename(columns={'duration_sec': f'latency_{step_from}_to_{step_to}'})
        )
        latencies.append(latency_grouped)

    # Tüm latency metriklerini zaman ekseninde birleştir
    result = latencies[0]
    for l_df in latencies[1:]:
        result = pd.merge(result, l_df, on='eventTime', how='outer')
        
    return result

if __name__ == "__main__":
    print("Events yükleniyor...")
    df_events = load_events("events.parquet")
    
    print("Genel (Overall) Funnel Metrikleri Hesaplanıyor (1 Saatlik Pencereler)...")
    overall_metrics = aggregate_funnel_metrics(df_events, freq="1h")
    overall_latencies = calculate_step_latencies(df_events, freq="1h")
    
    # Metrikleri birleştir ve kaydet
    final_overall = pd.merge(overall_metrics, overall_latencies, on='eventTime', how='left')
    final_overall.to_parquet("funnel_metrics_overall.parquet", index=False)
    
    print("Boyut Bazlı (Dimension-based) Metrikler Hesaplanıyor...")
    # RCA (Root Cause Analysis) modülünün hızlı sorgulaması için boyut bazlı hesaplama
    dim_metrics = aggregate_funnel_metrics(df_events, freq="1h", groupByDims=["platform", "appVersion", "paymentProvider"])
    dim_metrics.to_parquet("funnel_metrics_by_dimensions.parquet", index=False)
    
    print("Metrik üretimi tamamlandı!")
    print(f"Overall Metrics Boyutu: {final_overall.shape}")
    print(f"Dimension Metrics Boyutu: {dim_metrics.shape}")