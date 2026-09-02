import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Konfigürasyon ve Sabitler
SEED = 42
np.random.seed(SEED)

START_DATE = datetime(2026, 8, 1, 0, 0, 0)
DAYS = 14
TOTAL_HOURS = DAYS * 24

FUNNEL_STEPS = [
    "product_viewed",
    "add_to_cart",
    "checkout_started",
    "payment_submitted",
    "purchase_completed",
]

BASE_CONVERSIONS = {
    "add_to_cart": 0.45,  # product_viewed -> add_to_cart
    "checkout_started": 0.70,  # add_to_cart -> checkout_started
    "payment_submitted": 0.80,  # checkout_started -> payment_submitted
    "purchase_completed": 0.65,  # payment_submitted -> purchase_completed
}

PLATFORMS = ["android", "ios", "web"]
PLATFORM_PROBS = [0.50, 0.40, 0.10]

APP_VERSIONS_ANDROID = ["5.4.0", "5.4.1", "5.4.2"]
APP_VERSIONS_IOS = ["3.1.0", "3.2.0"]

PAYMENT_PROVIDERS = ["Provider_A", "Provider_B", "Provider_C"]
PAYMENT_PROBS = [0.60, 0.30, 0.10]

CAMPAIGNS = ["none", "summer_sale", "flash_deals"]
COUNTRIES = ["TR", "DE", "US"]


def get_hourly_base_traffic(hour_idx):
    """Günlük ve haftalık sezonsallık içeren baz trafik hacmi (User/Saat)"""
    dt = START_DATE + timedelta(hours=hour_idx)
    hour = dt.hour
    day_of_week = dt.weekday()

    # Gün içi dalgalanma (Gece düşük, akşam yüksek)
    daily_pattern = 0.3 + 0.7 * np.sin((hour - 6) * np.pi / 12) ** 2

    # Haftasonu artışı
    weekend_factor = 1.2 if day_of_week in [5, 6] else 1.0

    base_users = int(1200 * daily_pattern * weekend_factor)
    return max(base_users, 100)


def generate_synthetic_dataset():
    events_list = []

    # Ground Truth ve Change Events Yapıları
    ground_truth = []
    change_events = []

    # ---------------------------------------------------------
    # Anomali Senaryoları Tanımı (Inject Anomalies)
    # ---------------------------------------------------------

    # 1. Senaryo: Android 5.4.2 Release Sonrası Payment Failure (Conversion Anomaly)
    anomaly_1_start = START_DATE + timedelta(days=5, hours=14)
    anomaly_1_end = anomaly_1_start + timedelta(hours=8)
    release_1_time = anomaly_1_start - timedelta(minutes=18)

    change_events.append(
        {
            "change_id": "CHG-001",
            "timestamp": release_1_time.isoformat(),
            "change_type": "app_release",
            "platform": "android",
            "version": "5.4.2",
            "description": "Android 5.4.2 minor release rolled out to production.",
        }
    )

    ground_truth.append(
        {
            "anomaly_id": "ANO-001",
            "type": "conversion_anomaly",
            "start_time": anomaly_1_start.isoformat(),
            "end_time": anomaly_1_end.isoformat(),
            "affected_step": "purchase_completed",
            "root_cause": {
                "platform": "android",
                "appVersion": "5.4.2",
                "trigger_change_event": "CHG-001",
            },
            "description": "Android 5.4.2 update broken payment SDK integration, dropping completion rate.",
        }
    )

    # 2. Senaryo: Provider_B Altyapı Kesintisi (Volume & Conversion Anomaly)
    anomaly_2_start = START_DATE + timedelta(days=9, hours=10)
    anomaly_2_end = anomaly_2_start + timedelta(hours=4)

    change_events.append(
        {
            "change_id": "CHG-002",
            "timestamp": anomaly_2_start.isoformat(),
            "change_type": "third_party_incident",
            "provider": "Provider_B",
            "description": "Provider_B API gateway latency elevation and timeouts.",
        }
    )

    ground_truth.append(
        {
            "anomaly_id": "ANO-002",
            "type": "conversion_latency_anomaly",
            "start_time": anomaly_2_start.isoformat(),
            "end_time": anomaly_2_end.isoformat(),
            "affected_step": "payment_submitted",
            "root_cause": {
                "paymentProvider": "Provider_B",
                "trigger_change_event": "CHG-002",
            },
            "description": "Provider_B downtime causing payment submissions to fail.",
        }
    )

    # 3. Senaryo: Data Quality / Tracking Loss (Purchase Completed Event Missing)
    anomaly_3_start = START_DATE + timedelta(days=12, hours=18)
    anomaly_3_end = anomaly_3_start + timedelta(hours=6)

    ground_truth.append(
        {
            "anomaly_id": "ANO-003",
            "type": "data_quality_anomaly",
            "start_time": anomaly_3_start.isoformat(),
            "end_time": anomaly_3_end.isoformat(),
            "affected_step": "purchase_completed",
            "root_cause": {
                "platform": "ios",
                "appVersion": "3.2.0",
                "issue": "tracking_event_dropped",
            },
            "description": "iOS 3.2.0 failed to fire purchase_completed SDK telemetry event despite successful purchase.",
        }
    )

    # ---------------------------------------------------------
    # Event Üretim Döngüsü
    # ---------------------------------------------------------
    print("Event üretimi başlatılıyor...")

    for hour_idx in range(TOTAL_HOURS):
        current_hour_time = START_DATE + timedelta(hours=hour_idx)
        user_count = get_hourly_base_traffic(hour_idx)

        for _ in range(user_count):
            user_id = str(uuid.uuid4())[:8]

            # Dimension Seçimleri
            platform = np.random.choice(PLATFORMS, p=PLATFORM_PROBS)

            if platform == "android":
                # Eğer Anomaly 1 zamanından sonraysa 5.4.2 yayılımı yüksek
                if current_hour_time >= release_1_time:
                    app_version = np.random.choice(
                        APP_VERSIONS_ANDROID, p=[0.1, 0.2, 0.7]
                    )
                else:
                    app_version = np.random.choice(
                        APP_VERSIONS_ANDROID, p=[0.6, 0.4, 0.0]
                    )
                os_version = np.random.choice(["Android 12", "Android 13", "Android 14"])
            elif platform == "ios":
                app_version = np.random.choice(APP_VERSIONS_IOS, p=[0.7, 0.3])
                os_version = np.random.choice(["iOS 17.1", "iOS 17.2"])
            else:
                app_version = "web_v1.0"
                os_version = "Windows/macOS"

            payment_provider = np.random.choice(
                PAYMENT_PROVIDERS, p=PAYMENT_PROBS
            )
            campaign = np.random.choice(CAMPAIGNS, p=[0.7, 0.2, 0.1])
            country = np.random.choice(COUNTRIES, p=[0.7, 0.2, 0.1])

            # Time progression inside the hour
            event_timestamp = current_hour_time + timedelta(
                seconds=np.random.randint(0, 3600)
            )

            # Step 1: product_viewed
            events_list.append(
                {
                    "eventName": "product_viewed",
                    "userId": user_id,
                    "eventTime": event_timestamp.isoformat(),
                    "platform": platform,
                    "appVersion": app_version,
                    "osVersion": os_version,
                    "country": country,
                    "campaign": campaign,
                    "paymentProvider": payment_provider,
                    "revenue": 0.0,
                }
            )

            # Funnel İlerleme Kontrolü: Add To Cart
            p_cart = BASE_CONVERSIONS["add_to_cart"]
            if np.random.rand() > p_cart:
                continue

            event_timestamp += timedelta(seconds=np.random.randint(5, 60))
            events_list.append(
                {
                    "eventName": "add_to_cart",
                    "userId": user_id,
                    "eventTime": event_timestamp.isoformat(),
                    "platform": platform,
                    "appVersion": app_version,
                    "osVersion": os_version,
                    "country": country,
                    "campaign": campaign,
                    "paymentProvider": payment_provider,
                    "revenue": 0.0,
                }
            )

            # Funnel İlerleme Kontrolü: Checkout Started
            p_checkout = BASE_CONVERSIONS["checkout_started"]
            if np.random.rand() > p_checkout:
                continue

            event_timestamp += timedelta(seconds=np.random.randint(5, 30))
            events_list.append(
                {
                    "eventName": "checkout_started",
                    "userId": user_id,
                    "eventTime": event_timestamp.isoformat(),
                    "platform": platform,
                    "appVersion": app_version,
                    "osVersion": os_version,
                    "country": country,
                    "campaign": campaign,
                    "paymentProvider": payment_provider,
                    "revenue": 0.0,
                }
            )

            # Funnel İlerleme Kontrolü: Payment Submitted
            p_payment = BASE_CONVERSIONS["payment_submitted"]

            # Anomali 2 Etkisi: Provider_B çökerse payment_submitted başarısızlığı/düşüşü
            if (
                anomaly_2_start <= event_timestamp <= anomaly_2_end
                and payment_provider == "Provider_B"
            ):
                p_payment = 0.15  # Şiddetli düşüş

            if np.random.rand() > p_payment:
                continue

            event_timestamp += timedelta(seconds=np.random.randint(10, 45))
            events_list.append(
                {
                    "eventName": "payment_submitted",
                    "userId": user_id,
                    "eventTime": event_timestamp.isoformat(),
                    "platform": platform,
                    "appVersion": app_version,
                    "osVersion": os_version,
                    "country": country,
                    "campaign": campaign,
                    "paymentProvider": payment_provider,
                    "revenue": 0.0,
                }
            )

            # Funnel İlerleme Kontrolü: Purchase Completed
            p_purchase = BASE_CONVERSIONS["purchase_completed"]

            # Anomali 1 Etkisi: Android 5.4.2 versiyonunda ödeme tamamlama çöküşü
            if (
                anomaly_1_start <= event_timestamp <= anomaly_1_end
                and platform == "android"
                and app_version == "5.4.2"
            ):
                p_purchase = 0.10  # %65 -> %10 çöküş

            if np.random.rand() > p_purchase:
                continue

            event_timestamp += timedelta(seconds=np.random.randint(2, 15))
            order_revenue = round(float(np.random.gamma(shape=5.0, scale=30.0)), 2)

            # Anomali 3 Etkisi: iOS 3.2.0 versiyonunda event tracking kaybı (Veri var ama event dropped)
            if (
                anomaly_3_start <= event_timestamp <= anomaly_3_end
                and platform == "ios"
                and app_version == "3.2.0"
            ):
                # Satış gerçekleşti ama event telemetriye düşmedi (Data Quality Anomaly)
                continue

            events_list.append(
                {
                    "eventName": "purchase_completed",
                    "userId": user_id,
                    "eventTime": event_timestamp.isoformat(),
                    "platform": platform,
                    "appVersion": app_version,
                    "osVersion": os_version,
                    "country": country,
                    "campaign": campaign,
                    "paymentProvider": payment_provider,
                    "revenue": order_revenue,
                }
            )

    # Dataframe Dönüşümü
    df_events = pd.DataFrame(events_list)

    # Parquet & JSON Çıktıları
    df_events.to_parquet("events.parquet", index=False)

    import json

    with open("change_events.json", "w", encoding="utf-8") as f:
        json.dump(change_events, f, indent=2, ensure_ascii=False)

    with open("ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"Başarıyla tamamlandı!")
    print(f"Toplam Üretilen Event Sayısı: {len(df_events):,}")
    print(
        f"Dosyalar oluşturuldu: 'events.parquet', 'change_events.json', 'ground_truth.json'"
    )


if __name__ == "__main__":
    generate_synthetic_dataset()