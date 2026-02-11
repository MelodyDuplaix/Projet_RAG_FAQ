from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, generate_latest

router = APIRouter()

REQUEST_COUNT = Counter(
    'faq_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

RESPONSE_TIME = Histogram(
    'faq_response_time_seconds',
    'Response time in seconds',
    ['strategy']
)

CONFIDENCE_SCORE = Histogram(
    'faq_confidence_score',
    'Distribution of confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

@router.get("/metrics")
async def get_metrics():
    """Endpoint for Prometheus."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
