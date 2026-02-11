from fastapi import FastAPI
from src import routes

app = FastAPI(
    title="FAQ IA API",
    description="API de réponse automatique aux questions FAQ",
    version="1.0.0"
)

@app.get("/health", summary="Check API Health")
def health_route():
    """A simple health check endpoint."""
    return {"status": "ok"}

app.include_router(routes.router, prefix="/api/v1")