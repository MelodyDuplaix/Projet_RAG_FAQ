from fastapi import FastAPI
from src import routes

app = FastAPI(
    title="FAQ IA API",
    description="API de réponse automatique aux questions FAQ",
    version="1.0.0"
)

app.include_router(routes.router, prefix="/api/v1")