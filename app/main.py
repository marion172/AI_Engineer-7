"""
app/main.py
===========
API REST con FastAPI para el sistema multi-agente en producción.
Capacidades:
1. Endpoints Asíncronos: No bloqueantes, encolan y devuelven job_id de inmediato.
2. Gestión de Estado con Redis: Persistencia de jobs y checkpoints de LangGraph.
3. Observabilidad: Monitoreo activo con LangSmith.
4. Human-in-the-loop (HITL): Endpoint de aprobación para operaciones críticas.
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno antes de importar LangChain/LangGraph
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
if os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
if os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")

import asyncio
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.observability import setup_observability
from app.hitl import ApprovalPayload
from app.worker import (
    get_redis_client,
    get_task_from_storage,
    list_recent_tasks,
    resume_agent_task_background,
    run_agent_task_background,
    update_task_status,
)

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api_agent.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inicialización de recursos y observabilidad al iniciar FastAPI.
    """
    logger.info("[Startup] Inicializando capa de observabilidad...")
    obs_status = setup_observability()
    logger.info(f"[Startup] Estado de observabilidad: {obs_status}")

    logger.info("[Startup] Verificando conexión con Redis...")
    redis_conn = get_redis_client()
    if redis_conn:
        logger.info("[Startup] Redis conectado y listo.")
    else:
        logger.warning("[Startup] Redis no disponible en host local; fallback en memoria activo.")

    yield
    logger.info("[Shutdown] Finalizando API de producción.")


app = FastAPI(
    title="Multi-Agent Production API & Active Monitoring",
    description="API REST asíncrona para orquestación de agentes con persistencia en Redis, observabilidad y HITL.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Modelos Pydantic de Request y Response
# ==============================================================================

class TaskCreateRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="Consulta o directiva a resolver por el sistema multi-agente.",
        examples=["Analizar resultados de NVIDIA vs AMD y calcular variación de data center"],
    )
    require_approval: bool = Field(
        default=False,
        description="Fuerza una pausa HITL obligatoria para que un humano autorice antes de emitir el reporte.",
    )
    max_iterations: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Límite máximo de iteraciones del Supervisor para evitar bucles infinitos.",
    )


class TaskResponse(BaseModel):
    job_id: str
    status: str
    query: str
    created_at: str
    message: str


class TaskDetailResponse(BaseModel):
    task_id: str
    status: str
    query: str
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    requires_approval: Optional[bool] = False
    approval_request: Optional[Dict[str, Any]] = None
    approval_response: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    sources: Optional[List[str]] = None
    metrics: Optional[List[Any]] = None


# ==============================================================================
# Endpoints de la API
# ==============================================================================

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Multi-Agent Production API",
        "version": "1.0.0",
        "endpoints": {
            "create_task": "POST /tasks",
            "get_task_status": "GET /tasks/{id}",
            "approve_hitl": "POST /tasks/{id}/approve",
            "list_tasks": "GET /tasks",
            "health_check": "GET /health",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["General"])
async def health_check():
    """
    Verifica el estado de salud de la API, la conexión con Redis y los colectores de observabilidad.
    """
    redis_conn = get_redis_client()
    redis_healthy = False
    if redis_conn:
        try:
            redis_conn.ping()
            redis_healthy = True
        except Exception:
            redis_healthy = False

    obs_status = setup_observability()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        "redis": {
            "connected": redis_healthy,
            "mode": "redis_cluster/standalone" if redis_healthy else "in_memory_fallback",
        },
        "observability": {
            "provider": "LangSmith",
            "langsmith_active": bool(obs_status["langsmith_active"]),
            "project": obs_status["details"]["project"],
        },
    }


@app.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Tasks"],
)
async def create_task(payload: TaskCreateRequest):
    """
    Endpoint Asíncrono no bloqueante:
    Recibe la tarea, la persiste con estado PENDING en Redis, la despacha
    a un worker en segundo plano y devuelve inmediatamente el 'job_id'.
    """
    try:
        job_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Guardar estado inicial en Redis
        update_task_status(
            task_id=job_id,
            status="PENDING",
            query=payload.query,
            requires_approval=payload.require_approval,
            max_iterations=payload.max_iterations,
            created_at=now_iso,
            updated_at=now_iso,
            result=None,
            error=None,
        )

        # 2. Despachar al worker en segundo plano sin bloquear el Event Loop
        asyncio.create_task(
            run_agent_task_background(
                task_id=job_id,
                query=payload.query,
                require_approval=payload.require_approval,
                max_iterations=payload.max_iterations,
            )
        )

        logger.info(f"[POST /tasks] Tarea '{job_id}' encolada exitosamente para query: '{payload.query[:40]}...'")

        return TaskResponse(
            job_id=job_id,
            status="PENDING",
            query=payload.query,
            created_at=now_iso,
            message="Tarea encolada en segundo plano exitosamente. Consulte el estado en GET /tasks/{id}",
        )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[POST /tasks ERROR] Falló al encolar tarea:\n{tb}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al crear tarea: {type(exc).__name__}: {str(exc)}",
        )


@app.get(
    "/tasks/{task_id}",
    response_model=TaskDetailResponse,
    tags=["Tasks"],
)
async def get_task_status(task_id: str):
    """
    Consulta el estado del trabajo sin bloquear el Event Loop.
    Devuelve: PENDING, RUNNING, WAITING_APPROVAL, COMPLETED o FAILED.
    """
    task_data = get_task_from_storage(task_id)
    if not task_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"La tarea con ID '{task_id}' no existe en el sistema.",
        )

    return TaskDetailResponse(
        task_id=task_data.get("task_id", task_id),
        status=task_data.get("status", "UNKNOWN"),
        query=task_data.get("query", ""),
        created_at=task_data.get("created_at", ""),
        updated_at=task_data.get("updated_at"),
        completed_at=task_data.get("completed_at"),
        requires_approval=task_data.get("requires_approval", False),
        approval_request=task_data.get("approval_request"),
        approval_response=task_data.get("approval_response"),
        result=task_data.get("result"),
        error=task_data.get("error"),
        execution_time_seconds=task_data.get("execution_time_seconds"),
        sources=task_data.get("sources"),
        metrics=task_data.get("metrics"),
    )


@app.post(
    "/tasks/{task_id}/approve",
    tags=["Human-in-the-loop"],
)
async def approve_task(task_id: str, payload: ApprovalPayload):
    """
    Endpoint Human-in-the-loop (HITL):
    Recibe la autorización o rechazo de un operador humano para una tarea pausada en WAITING_APPROVAL.
    Reanuda el grafo en segundo plano inyectando la respuesta al checkpoint en Redis.
    """
    task_data = get_task_from_storage(task_id)
    if not task_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"La tarea con ID '{task_id}' no existe.",
        )

    current_status = task_data.get("status")
    if current_status != "WAITING_APPROVAL":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"La tarea '{task_id}' se encuentra en estado '{current_status}'. "
                "Solo se pueden autorizar tareas en estado 'WAITING_APPROVAL'."
            ),
        )

    # Despachar la reanudación del grafo en segundo plano
    asyncio.create_task(
        resume_agent_task_background(
            task_id=task_id,
            approved=payload.approved,
            feedback=payload.feedback,
        )
    )

    action_label = "AUTORIZADA" if payload.approved else "RECHAZADA"
    logger.info(f"[POST /tasks/{task_id}/approve] Tarea {action_label} por operador humano.")

    return {
        "job_id": task_id,
        "status": "RESUMING",
        "action": action_label,
        "message": f"Tarea {action_label}. La ejecución continuará en segundo plano.",
    }


@app.get(
    "/tasks",
    response_model=List[TaskDetailResponse],
    tags=["Tasks"],
)
async def list_tasks(limit: int = 20):
    """
    Lista las tareas recientes persistidas en el sistema.
    """
    tasks = list_recent_tasks(limit=limit)
    return [
        TaskDetailResponse(
            task_id=t.get("task_id", ""),
            status=t.get("status", "UNKNOWN"),
            query=t.get("query", ""),
            created_at=t.get("created_at", ""),
            updated_at=t.get("updated_at"),
            completed_at=t.get("completed_at"),
            requires_approval=t.get("requires_approval", False),
            approval_request=t.get("approval_request"),
            approval_response=t.get("approval_response"),
            result=t.get("result"),
            error=t.get("error"),
            execution_time_seconds=t.get("execution_time_seconds"),
            sources=t.get("sources"),
            metrics=t.get("metrics"),
        )
        for t in tasks
    ]

