# app/core/context.py
import contextvars

# Membuat ContextVar untuk menyimpan Request ID.
# Default "system" digunakan untuk log yang terjadi di luar request HTTP 
# (misalnya saat server baru menyala atau saat shutdown).
request_id_context_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="system")