from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn

from pipeline import (
    load_raw_artifacts,
    get_real_aov,
    build_incident_bundle,
    build_tracking_bundle,
    discover_all_incidents,
)

app = FastAPI(
    title="Enterprise Funnel Anomaly & RCA Engine API",
    description="FR-01 Dinamik Funnel Yönetimi, Anomali Tespiti ve Kök Neden Analiz Servisi",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================================================================
# FR-01: DYNAMIC FUNNEL SCHEMA & MANAGEMENT (Pydantic Models)
# =============================================================================
class FunnelStep(BaseModel):
    step_order: int = Field(..., example=1, description="Adım sırası")
    step_name: str = Field(..., example="Checkout Start", description="Adım adı")
    event_name: str = Field(..., example="checkout_started", description="İlişkili event ismi")
    description: Optional[str] = Field(None, example="Kullanıcı ödeme adımını başlattı", description="Adım açıklaması")

class FunnelConfig(BaseModel):
    funnel_id: str = Field(..., example="checkout-funnel-v1", description="Benzersiz Funnel Kimliği")
    name: str = Field(..., example="E-Commerce Primary Checkout Funnel", description="Funnel Adı")
    timezone: str = Field("Europe/Istanbul", example="Europe/Istanbul", description="İş Zaman Dilimi")
    calculation_type: str = Field("user_level", example="user_level", description="’user_level’ veya ’session_level’")
    count_unique_events: bool = Field(True, description="Deduplication (Tekil sayım) aktif mi?")
    max_step_interval_hours: int = Field(24, description="Adımlar arası maksimum izin verilen süre (saat)")
    ordered: bool = Field(True, description="Adımlar kesin sıralı mı işlenmeli?")
    steps: List[FunnelStep]

# Bellek İçi Funnel Deposu (Varsayılan e-ticaret funneli ile başlatılıyor)
funnel_store: Dict[str, FunnelConfig] = {
    "default-checkout": FunnelConfig(
        funnel_id="default-checkout",
        name="Default E-Commerce Checkout",
        timezone="Europe/Istanbul",
        calculation_type="user_level",
        count_unique_events=True,
        max_step_interval_hours=24,
        ordered=True,
        steps=[
            FunnelStep(step_order=1, step_name="Product View", event_name="product_viewed"),
            FunnelStep(step_order=2, step_name="Add to Cart", event_name="add_to_cart"),
            FunnelStep(step_order=3, step_name="Checkout Start", event_name="checkout_started"),
            FunnelStep(step_order=4, step_name="Payment Submit", event_name="payment_submitted"),
            FunnelStep(step_order=5, step_name="Purchase Complete", event_name="purchase_completed"),
        ]
    )
}

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

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
def read_root():
    """Kök dizin yönlendirmesi - 404 hatasını önler ve dokümantasyon bağlantılarını verir."""
    return {
        "status": "online",
        "service": "Enterprise Funnel Anomaly & RCA Engine API",
        "swagger_docs": "http://127.0.0.1:8000/docs",
        "redoc_docs": "http://127.0.0.1:8000/redoc",
        "health_check": "http://127.0.0.1:8000/health"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "RCA Core API"}

# --- FR-01 Funnel Management Endpoints ---
@app.post("/api/v1/funnels", tags=["FR-01 Funnel Management"])
def create_funnel(config: FunnelConfig):
    """Yeni bir Funnel tanımı oluşturur veya var olanı günceller (Create Funnel)."""
    funnel_store[config.funnel_id] = config
    return {
        "success": True,
        "message": f"'{config.name}' başlıklı funnel başarıyla kaydedildi.",
        "funnel_id": config.funnel_id,
        "data": config
    }

@app.post("/api/v1/funnels/import", tags=["FR-01 Funnel Management"])
def import_funnel(config_json: Dict[str, Any]):
    """Dışarıdan JSON tanımı alarak hazır bir funnel şemasını içe aktarır (Import Funnel)."""
    try:
        config = FunnelConfig(**config_json)
        funnel_store[config.funnel_id] = config
        return {"success": True, "message": "Funnel JSON tanımı başarıyla içe aktarıldı.", "data": config}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geçersiz Funnel JSON formatı: {str(e)}")

@app.get("/api/v1/funnels", tags=["FR-01 Funnel Management"])
def list_funnels():
    """Kayıtlı tüm funnel tanımlarını (FR-01 şemasıyla) listeler."""
    return {"count": len(funnel_store), "funnels": list(funnel_store.values())}

# --- Analytics & Incident Discovery Endpoints ---
@app.get("/api/v1/incidents", tags=["Analytics & RCA"])
def get_incidents():
    """Tüm analiz paketlerini (Statik ve Otomatik Keşfedilenler) çalıştırır ve döndürür."""
    try:
        df_overall, df_dims, df_detected, change_events_raw, ground_truth = load_raw_artifacts()
        aov = get_real_aov(df_overall)

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
                    "window_start": str(cfg["window_start"]),
                    "window_end": str(cfg["window_end"]),
                    "tracking": tracking,
                }
            bundles[label] = b

        discovered = discover_all_incidents(df_detected, df_dims, df_overall, change_events_raw, aov)
        bundles.update(discovered)

        return {"success": True, "real_aov": aov, "bundles": bundles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # host parametresini "0.0.0.0" yerine "127.0.0.1" yapıyoruz
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)