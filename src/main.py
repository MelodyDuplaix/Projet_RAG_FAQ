from fastapi import FastAPI
from src.config.logging_config import setup_logging
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.routes import metrics, api_router
from datetime import datetime
from src.services.data_loader import load_faq_data

setup_logging()

app = FastAPI(
    title="FAQ IA API",
    description="API de réponse automatique aux questions FAQ",
    version="1.0.0"
)

app.add_middleware(LoggingMiddleware)

@app.get("/health", summary="Check API Health")
def health_route():
    """Provides a detailed health check for the API."""
    faq_df = load_faq_data()
    faq_count = len(faq_df) if not faq_df.empty else 0
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
        "faq_count": faq_count
    }

app.include_router(api_router.router, prefix="/api/v1")
app.include_router(metrics.router)