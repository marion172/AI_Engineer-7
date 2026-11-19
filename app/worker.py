"""
app/worker.py
=============
Worker asíncrono y gestor de persistencia en Redis.
Gestiona el ciclo de vida de las tareas en segundo plano sin bloquear el Event Loop de FastAPI:
- Estados: PENDING -> RUNNING -> (WAITING_APPROVAL) -> COMPLETED / FAILED
- Captura de excepciones con paso a FAILED para evitar loops infinitos de polling.
- Reanudación de checkpoints tras aprobación humana.
"""

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from app.graph import get_graph
from app.state import create_initial_state

logger = logging.getLogger("api_agent.worker")

# Almacén de respaldo en memoria si Redis no estuviese disponible
_MEMORY_TASK_STORE: Dict[str, Dict[str, Any]] = {}
_REDIS_CLIENT = None


def get_redis_client():
    """
    Instancia o reutiliza el cliente Redis.
    """
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis
        client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
        client.ping()
        _REDIS_CLIENT = client
        logger.info(f"[Redis] Conectado exitosamente a {redis_url}")
        return _REDIS_CLIENT
    except Exception as exc:
        logger.warning(
            f"[Redis] No se pudo conectar a Redis en {redis_url} ({exc}). "
            "Se usará almacenamiento seguro en memoria como fallback de respaldo."
        )
        _REDIS_CLIENT = None
        return None


def get_task_from_storage(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Lee el estado del trabajo desde Redis o desde el almacén de respaldo.
    """
    client = get_redis_client()
    if client:
        try:
            raw = client.get(f"task:{task_id}")
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.error(f"[Redis] Error leyendo tarea {task_id}: {exc}")

    return _MEMORY_TASK_STORE.get(task_id)


def save_task_to_storage(task_id: str, task_data: Dict[str, Any]) -> None:
    """
    Persiste el estado del trabajo en Redis con expiración de 7 días, y en memoria.
    """
    task_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    _MEMORY_TASK_STORE[task_id] = task_data

    client = get_redis_client()
    if client:
        try:
            client.set(f"task:{task_id}", json.dumps(task_data, default=str), ex=604800)
            # Agregar al conjunto de tareas recientes
            client.zadd("recent_tasks", {task_id: time.time()})
        except Exception as exc:
            logger.error(f"[Redis] Error guardando tarea {task_id}: {exc}")


def update_task_status(task_id: str, status: str, **kwargs) -> Dict[str, Any]:
    """
    Actualiza el estado y metadatos de un trabajo en Redis.
    """
    task_data = get_task_from_storage(task_id) or {
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    task_data["status"] = status
    for key, val in kwargs.items():
        task_data[key] = val

    save_task_to_storage(task_id, task_data)
    return task_data


def _get_execution_config(task_id: str) -> tuple[Dict[str, Any], Optional[Any]]:
    """
    Construye la configuración de ejecución con callbacks forzados y cliente autenticado para LangSmith.
    """
    config: Dict[str, Any] = {
        "configurable": {
            "thread_id": task_id,
        },
        "run_name": f"task_{task_id[:8]}",
        "tags": ["production-api", "multi-agent-workflow"],
    }

    tracer = None
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
    project_name = os.getenv("LANGCHAIN_PROJECT", "pre-entrega-7-api")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    if langchain_api_key and langchain_api_key != "tu_langsmith_api_key_aqui":
        try:
            from langsmith import Client
            from langchain_core.tracers import LangChainTracer
            ls_client = Client(api_key=langchain_api_key, api_url=endpoint)
            tracer = LangChainTracer(project_name=project_name, client=ls_client)
            config["callbacks"] = [tracer]
            logger.info(f"[WORKER] LangChainTracer con cliente autenticado adjuntado para '{project_name}'.")
        except Exception as exc:
            logger.warning(f"[WORKER] No se pudo instanciar LangChainTracer explícito: {exc}")

    return config, tracer


def list_recent_tasks(limit: int = 20) -> list[Dict[str, Any]]:
    """
    Lista los trabajos más recientes ordenados por fecha de creación.
    """
    client = get_redis_client()
    if client:
        try:
            task_ids = client.zrevrange("recent_tasks", 0, limit - 1)
            results = []
            for tid in task_ids:
                t = get_task_from_storage(tid)
                if t:
                    results.append(t)
            if results:
                return results
        except Exception as exc:
            logger.error(f"[Redis] Error listando tareas: {exc}")

    # Fallback memoria
    items = list(_MEMORY_TASK_STORE.values())
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items[:limit]


async def run_agent_task_background(
    task_id: str,
    query: str,
    require_approval: bool = False,
    max_iterations: int = 5,
) -> None:
    """
    Ejecuta el orquestador multi-agente en un hilo desacoplado del Event Loop de FastAPI.
    Controla interrupciones HITL y captura cualquier excepción para marcar el estado en FAILED.
    """
    logger.info(f"[WORKER] Iniciando ejecución en segundo plano para tarea: {task_id}")
    start_time = time.perf_counter()

    update_task_status(
        task_id,
        status="RUNNING",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    graph = get_graph()
    config, tracer = _get_execution_config(task_id)

    initial_state = create_initial_state(
        user_query=query,
        task_id=task_id,
        require_approval=require_approval,
        max_iterations=max_iterations,
    )

    try:
        # Ejecutar en threadpool para no bloquear el Event Loop de FastAPI
        result_state = await asyncio.to_thread(graph.invoke, initial_state, config)
        execution_time = round(time.perf_counter() - start_time, 3)

        if tracer and hasattr(tracer, "wait_for_futures"):
            try:
                tracer.wait_for_futures()
            except Exception:
                pass

        # Verificar si el estado final requiere aprobación pendiente
        approval_status = result_state.get("approval_status", "NOT_REQUIRED")

        if approval_status == "WAITING_APPROVAL":
            logger.info(f"[WORKER] Tarea {task_id} en pausa obligatoria por HITL.")
            update_task_status(
                task_id,
                status="WAITING_APPROVAL",
                requires_approval=True,
                approval_request=result_state.get("approval_request"),
                execution_time_seconds=execution_time,
            )
            return

        final_report = result_state.get("final_report", "Reporte generado.")
        logger.info(f"[WORKER] Tarea {task_id} finalizada exitosamente en {execution_time}s.")

        update_task_status(
            task_id,
            status="COMPLETED",
            result=final_report,
            execution_time_seconds=execution_time,
            completed_at=datetime.now(timezone.utc).isoformat(),
            sources=result_state.get("research_data", {}).get("sources", []),
            metrics=result_state.get("analysis_data", {}).get("calculations", []),
        )

    except GraphInterrupt as hitl_interrupt:
        #  Captura nativa de la interrupción de LangGraph
        execution_time = round(time.perf_counter() - start_time, 3)
        logger.info(f"[WORKER] Interrupción HITL detectada para tarea {task_id}: {hitl_interrupt}")

        # Extraer detalle de la interrupción
        interrupt_val = getattr(hitl_interrupt, "args", [{}])[0]
        update_task_status(
            task_id,
            status="WAITING_APPROVAL",
            requires_approval=True,
            approval_request=interrupt_val,
            execution_time_seconds=execution_time,
        )

    except Exception as exc:
        execution_time = round(time.perf_counter() - start_time, 3)
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        tb = traceback.format_exc()
        logger.error(f"[WORKER ERROR] Tarea {task_id} falló tras {execution_time}s:\n{tb}")

        #  Actualización obligatoria a FAILED en Redis ante excepciones
        update_task_status(
            task_id,
            status="FAILED",
            error=error_msg,
            traceback=tb,
            execution_time_seconds=execution_time,
            failed_at=datetime.now(timezone.utc).isoformat(),
        )


async def resume_agent_task_background(
    task_id: str,
    approved: bool,
    feedback: Optional[str] = None,
) -> None:
    """
    Reanuda una tarea pausada en LangGraph enviando la decisión humana al checkpoint.
    """
    logger.info(f"[WORKER] Reanudando tarea HITL {task_id} con approved={approved}")
    start_time = time.perf_counter()

    update_task_status(
        task_id,
        status="RUNNING",
        approval_response={"approved": approved, "feedback": feedback},
        resumed_at=datetime.now(timezone.utc).isoformat(),
    )

    graph = get_graph()
    config, tracer = _get_execution_config(task_id)

    try:
        resume_command = Command(resume={"approved": approved, "feedback": feedback})
        result_state = await asyncio.to_thread(graph.invoke, resume_command, config)
        execution_time = round(time.perf_counter() - start_time, 3)

        if tracer and hasattr(tracer, "wait_for_futures"):
            try:
                tracer.wait_for_futures()
            except Exception:
                pass

        final_report = result_state.get("final_report", "Reporte post-aprobación completado.")
        logger.info(f"[WORKER] Tarea reanudada {task_id} finalizada exitosamente en {execution_time}s.")

        update_task_status(
            task_id,
            status="COMPLETED",
            result=final_report,
            execution_time_seconds=execution_time,
            completed_at=datetime.now(timezone.utc).isoformat(),
            sources=result_state.get("research_data", {}).get("sources", []),
            metrics=result_state.get("analysis_data", {}).get("calculations", []),
        )

    except Exception as exc:
        execution_time = round(time.perf_counter() - start_time, 3)
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        tb = traceback.format_exc()
        logger.error(f"[WORKER RESUME ERROR] Tarea {task_id} falló al reanudar:\n{tb}")

        update_task_status(
            task_id,
            status="FAILED",
            error=error_msg,
            traceback=tb,
            execution_time_seconds=execution_time,
            failed_at=datetime.now(timezone.utc).isoformat(),
        )

