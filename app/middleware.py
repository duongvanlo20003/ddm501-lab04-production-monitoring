"""
HTTP metrics middleware.

"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, REQUESTS_IN_PROGRESS


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records count, latency and in-flight requests for every call."""

    # Scraping /metrics would otherwise count itself, and Prometheus polls it
    # every ten seconds — enough to dominate the request count on a quiet
    # service and make the traffic panel meaningless.
    EXCLUDED = {"/metrics"}

    async def dispatch(self, request: Request, call_next) -> Response:
        """TASK record count and latency for every request.

        Four things this must get right, each of which is a real outage
        someone has had:

          1. Skip the paths in EXCLUDED. Prometheus scrapes /metrics every
             ten seconds; counting those dominates the traffic panel on a
             quiet service.
          2. Label with the ROUTE TEMPLATE, not the resolved path — use the
             _route_template helper below. Without it, an endpoint like
             /applications/{id} mints one time series per id, and the
             cardinality explosion takes Prometheus down with it.
          3. Record in a `finally` block, so a request that raises on the way
             through is still counted. Middleware that only records on
             success reports a perfectly healthy service during an outage.
          4. Keep REQUESTS_IN_PROGRESS balanced: inc before, dec in the
             `finally`. An unbalanced gauge drifts upward forever.
        """
        path = request.url.path
        if path in self.EXCLUDED:
            return await call_next(request)

        started = time.perf_counter()
        status = "500"
        REQUESTS_IN_PROGRESS.inc()
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            endpoint = _route_template(request, path)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=status,
            ).inc()
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(time.perf_counter() - started)
            REQUESTS_IN_PROGRESS.dec()


def _route_template(request: Request, fallback: str) -> str:
    """The matched route's path template, or the raw path if none matched."""
    route = request.scope.get("route")
    return getattr(route, "path", fallback) or fallback
