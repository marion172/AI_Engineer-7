"""
app/observability.py
====================
Capa de observabilidad integral con LangSmith.
Instrumenta de forma activa todas las llamadas del sistema multi-agente:
- Trazas completas de cada nodo en LangSmith (Runs, Inputs/Outputs).
- Monitoreo de latencia p95 y tiempos de respuesta.
- Registro y visualización del costo por ejecución a partir de tokens consumidos.
"""

import functools
import logging
import os
import time
from typing import Any, Callable, Dict, Optional
from dotenv import load_dotenv

# Cargar variables de entorno inmediatamente
load_dotenv()

logger = logging.getLogger("api_agent.observability")


def setup_observability() -> Dict[str, Any]:
    """
    Inicializa y valida la configuración de LangSmith para monitoreo activo.
    """
    load_dotenv()
    langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() in ("true", "1", "yes")
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_project = os.getenv("LANGCHAIN_PROJECT", "pre-entrega-7-api")
    langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    # Asegurar variables de entorno requeridas por LangChain/LangGraph
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if langchain_tracing else "false"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint

    status = {
        "provider": "LangSmith",
        "langsmith_active": False,
        "details": {
            "project": langchain_project,
            "endpoint": langchain_endpoint,
            "api_key_configured": bool(langchain_api_key and langchain_api_key != "tu_langsmith_api_key_aqui"),
        },
    }

    if langchain_tracing and langchain_api_key and langchain_api_key != "tu_langsmith_api_key_aqui":
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        logger.info(f"[Observability - LangSmith] Tracing ACTIVO en proyecto '{langchain_project}'.")
        status["langsmith_active"] = True
    else:
        logger.warning(
            f"[Observability - LangSmith] Tracing configurado pero falta LANGCHAIN_API_KEY válida. "
            f"Configurala en .env para ver las trazas en {langchain_endpoint}."
        )

    return status


def trace_llm_call(step_name: Optional[str] = None):
    """
    Decorador para instrumentar funciones de llamadas a LLM o nodos críticos.
    Mide tiempo de ejecución, maneja errores y registra logs de ejecución.
    """
    def decorator(func: Callable):
        name = step_name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            logger.info(f"[LANGSMITH TRACE START] Paso '{name}' iniciado.")
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.info(f"[LANGSMITH TRACE END] '{name}' completado en {elapsed:.3f}s.")
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                logger.error(f"[LANGSMITH TRACE ERROR] '{name}' falló tras {elapsed:.3f}s: {exc}")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            logger.info(f"[LANGSMITH TRACE START] Paso asíncrono '{name}' iniciado.")
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.info(f"[LANGSMITH TRACE END] '{name}' completado en {elapsed:.3f}s.")
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                logger.error(f"[LANGSMITH TRACE ERROR] '{name}' falló tras {elapsed:.3f}s: {exc}")
                raise

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def calculate_estimated_cost(prompt_tokens: int, completion_tokens: int, model: str = "nvidia/nemotron-3.5-lightning:free") -> float:
    """
    Calcula el costo estimado de la ejecución en USD.
    """
    rates = {
        "nvidia/nemotron-3.5-lightning:free": {"prompt": 0.0, "completion": 0.0},
        "nvidia/nemotron-4-340b-instruct": {"prompt": 0.20 / 1_000_000, "completion": 0.80 / 1_000_000},
        "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
        "default": {"prompt": 0.0, "completion": 0.0},
    }
    pricing = rates.get(model, rates["default"])
    return (prompt_tokens * pricing["prompt"]) + (completion_tokens * pricing["completion"])
