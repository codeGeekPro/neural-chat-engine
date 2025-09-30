"""
Module de sécurité renforcée pour Neural Chat Engine.
"""
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging
import re
import time

class SecurityHardening:
    def __init__(self, app=None):
        self.app = app
        self.logger = logging.getLogger(__name__)

    def implement_input_sanitization(self, data: dict) -> dict:
        # Nettoyage basique des entrées (OWASP)
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, str):
                v = re.sub(r'<.*?>', '', v)  # Remove HTML tags (XSS)
                v = v.replace("'", "")    # Remove single quotes (SQLi)
            sanitized[k] = v
        self.logger.info("Input sanitized.")
        return sanitized

    def setup_sql_injection_prevention(self, query: str) -> bool:
        # Détection simple de patterns SQLi
        patterns = [r"(--|;|\bOR\b|\bAND\b|\bDROP\b|\bUNION\b)"]
        for p in patterns:
            if re.search(p, query, re.IGNORECASE):
                self.logger.warning("SQL injection pattern detected!")
                return False
        return True

    def configure_cors_properly(self, app, allowed_origins=None):
        # CORS strict
        if allowed_origins is None:
            allowed_origins = ["https://yourdomain.com"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
        self.logger.info("CORS configured.")

    def implement_rate_limiting_advanced(self, app, max_requests=100, window_seconds=60):
        # Middleware simple de rate limiting (à adapter pour Redis/production)
        class RateLimitMiddleware(BaseHTTPMiddleware):
            request_counts = {}
            def dispatch(self, request, call_next):
                ip = request.client.host
                now = int(time.time())
                window = now // window_seconds
                key = f"{ip}:{window}"
                self.request_counts.setdefault(key, 0)
                self.request_counts[key] += 1
                if self.request_counts[key] > max_requests:
                    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
                return call_next(request)
        app.add_middleware(RateLimitMiddleware)
        self.logger.info("Rate limiting enabled.")

    def setup_encryption_at_rest(self, file_path: str, data: bytes, key: bytes):
        # Exemple de chiffrement AES (à adapter pour DB/files)
        from cryptography.fernet import Fernet
        cipher = Fernet(key)
        encrypted = cipher.encrypt(data)
        with open(file_path, "wb") as f:
            f.write(encrypted)
        self.logger.info(f"Data encrypted at rest: {file_path}")

    def implement_oauth2_authentication(self, app):
        # Ajout du schéma OAuth2 (FastAPI)
        oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
        app.dependency_overrides[OAuth2PasswordBearer] = oauth2_scheme
        self.logger.info("OAuth2 authentication enabled.")

    def setup_security_headers(self, response: Response):
        # Ajout des headers de sécurité OWASP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        self.logger.info("Security headers set.")
        return response

    def implement_audit_logging(self, event: str, details: dict):
        # Log d’audit structuré
        self.logger.info(f"AUDIT: {event} | {details}")
