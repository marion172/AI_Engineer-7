"""
app/hitl.py
===========
Implementación del flujo Human-in-the-loop (HITL) para tareas y herramientas críticas.
Pausa la ejecución del grafo mediante 'interrupt' de LangGraph y persiste el checkpoint en Redis
hasta recibir la aprobación externa vía POST /tasks/{id}/approve.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from langgraph.types import interrupt
from langchain_core.messages import AIMessage

from app.state import AgentState
from app.observability import trace_llm_call

logger = logging.getLogger("api_agent.hitl")


class ApprovalPayload(BaseModel):
    """Esquema de entrada para el endpoint POST /tasks/{id}/approve."""
    approved: bool = Field(..., description="True para autorizar la ejecución, False para rechazar.")
    feedback: Optional[str] = Field(None, description="Comentario o directiva adicional del operador humano.")


CRITICAL_PATTERNS = [
    "comprar", "compra", "vender", "venta",
    "transferir", "transferencia", "liquidar",
    "orden de ejecucion", "ejecutar orden",
    "rebalancear cartera", "modificar portfolio",
    "alto costo", "gasto critico", "transaccion",
]


def evaluate_task_criticality(
    query: str,
    require_approval_flag: bool = False,
    state: Optional[AgentState] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Detecta si una consulta u operación requiere obligatoriamente aprobación humana.
    Evalúa flags explícitos, palabras clave de alto impacto financiero y efectos secundarios.
    """
    if require_approval_flag:
        return True, "Aprobación solicitada explícitamente por el cliente en la creación de la tarea.", {
            "trigger": "explicit_flag",
            "risk_level": "HIGH",
        }

    query_lower = query.lower()
    matched_patterns = [p for p in CRITICAL_PATTERNS if p in query_lower]

    if matched_patterns:
        return True, f"Acción financiera crítica detectada en consulta: {', '.join(matched_patterns)}.", {
            "trigger": "critical_keyword_detection",
            "matched_patterns": matched_patterns,
            "risk_level": "CRITICAL",
        }

    return False, "Operación estándar de análisis y consulta.", {
        "trigger": "routine_query",
        "risk_level": "LOW",
    }


@trace_llm_call(step_name="hitl_human_approval_node")
def human_approval_node(state: AgentState) -> dict:
    """
    Nodo de interrupción humana en LangGraph.
    
    1. Si la tarea ya cuenta con aprobación previa (o no es crítica), prosigue.
    2. Si requiere aprobación y está pendiente, emite 'interrupt()' congelando el grafo
       en Redis hasta que el endpoint /approve envíe la respuesta.
    3. Si es aprobada, continúa a síntesis; si es rechazada, cancela de forma controlada.
    """
    task_id = state.get("task_id", "unknown_task")
    query = state.get("user_query", "")
    existing_status = state.get("approval_status", "NOT_REQUIRED")

    # Si ya venía aprobada desde una ejecución previa
    if existing_status == "APPROVED":
        logger.info(f"[HITL] Tarea {task_id} ya se encuentra APROBADA. Continuando flujo.")
        return {
            "approval_status": "APPROVED",
            "is_critical": True,
            "messages": [AIMessage(content="[HITL] Operación crítica previamente aprobada por el operador humano.")],
        }

    # Evaluar criticidad
    is_critical, reason, details = evaluate_task_criticality(
        query=query,
        require_approval_flag=state.get("requires_human_approval", False),
        state=state,
    )

    if not is_critical:
        logger.info(f"[HITL] Tarea {task_id} no requiere aprobación humana.")
        return {
            "is_critical": False,
            "approval_status": "NOT_REQUIRED",
        }

    logger.warning(
        f"[HITL] Pausando ejecución para tarea {task_id}. Razón: {reason}. "
        f"Esperando llamada a POST /tasks/{task_id}/approve"
    )

    # Objeto descriptivo que se almacena en el checkpoint
    approval_request_data = {
        "task_id": task_id,
        "reason": reason,
        "details": details,
        "status": "WAITING_APPROVAL",
        "query": query,
    }

    # Interrupción nativa de LangGraph: suspende y persiste el estado en Redis
    human_response: Any = interrupt({
        "type": "human_approval_required",
        "task_id": task_id,
        "request": approval_request_data,
    })

    # Al reanudar vía Command(resume=...), human_response contiene el payload enviado
    logger.info(f"[HITL] Tarea {task_id} reanudada con respuesta del operador: {human_response}")

    approved = False
    feedback = ""
    if isinstance(human_response, dict):
        approved = human_response.get("approved", False)
        feedback = human_response.get("feedback", "")
    elif isinstance(human_response, bool):
        approved = human_response

    if approved:
        logger.info(f"[HITL] Tarea {task_id} AUTORIZADA por operador humano.")
        return {
            "approval_status": "APPROVED",
            "approval_response": {"approved": True, "feedback": feedback},
            "is_critical": True,
            "messages": [
                AIMessage(
                    content=f"[HITL] Aprobación humana concedida: {feedback or 'Sin comentarios adicionales.'}. Procediendo con la ejecución."
                )
            ],
        }
    else:
        logger.warning(f"[HITL] Tarea {task_id} RECHAZADA por operador humano.")
        return {
            "approval_status": "REJECTED",
            "approval_response": {"approved": False, "feedback": feedback},
            "is_critical": True,
            "task_completed": True,
            "final_report": (
                f"### [Operación Cancelada]\n"
                f"La tarea '{query}' fue identificada como una acción de alta criticidad ({reason}) "
                f"y fue RECHAZADA por el operador humano.\n"
                f"**Motivo de rechazo / Feedback:** {feedback or 'Acción denegada por política de seguridad.'}"
            ),
            "messages": [
                AIMessage(
                    content=f"[HITL] Tarea rechazada por operador humano. Feedback: {feedback or 'Acción cancelada.'}"
                )
            ],
        }

