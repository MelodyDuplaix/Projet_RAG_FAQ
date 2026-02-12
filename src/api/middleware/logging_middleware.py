import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("faq_api")

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        """Middleware to log all requests."""
        start_time = time.perf_counter()
        
        logger.info(f"{request.method} {request.url.path}")
        
        response = Response("Internal server error", status_code=500)
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"❌ Unhandled exception for {request.method} {request.url.path}: {e}", exc_info=True)
            raise e
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"⬅{response.status_code} in {duration_ms:.0f}ms")
            
            if duration_ms > 3000:
                logger.warning(f" High latency for {request.method} {request.url.path}: {duration_ms:.0f}ms")
            
        return response
