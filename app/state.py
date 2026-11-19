"""
app/state.py
============
Esquema de estado compartido del orquestador multi-agente con soporte
para persistencia en Redis y flujo Human-In-The-Loop (HITL).
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


AgentTarget = Literal["research_agent", "analyst_agent", "synthesizer", "human_approval", "FINISH"]
ApprovalStatus = Literal["PENDING", "APPROVED", "REJECTED", "NOT_REQUIRED"]


class RubricEvaluation(TypedDict, total=False):
    """Evaluación de suficiencia y rigor realizada por el Supervisor."""
    research_sufficiency: float  # De 0.0 a 1.0
    analysis_rigor: float        # De 0.0 a 1.0
    synthesis_ready: bool        # True si ya se puede sintetizar
    critique: str                # Justificación técnica


class AgentState(TypedDict):
    """
    Estado global de LangGraph persistido en Redis a través de checkpoints.
    Permite restaurar la conversación, manejar pausas HITL y evitar pérdida de estado.
    """
    # Historial de mensajes gestionado por LangGraph
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Identificador de la tarea en Redis
    task_id: str

    # Consulta original del usuario
    user_query: str

    # Próximo nodo al que se deriva el flujo
    next_agent: AgentTarget

    # Directiva específica enviada por el supervisor al especialista
    supervisor_feedback: str

    # Datos estructurados obtenidos por el agente de investigación
    research_data: Dict[str, Any]

    # Cálculos y análisis cuantitativos producidos por el analista
    analysis_data: Dict[str, Any]

    # Control de iteraciones para prevenir bucles infinitos
    iteration_count: int
    max_iterations: int

    # Criterios de calidad y suficiencia del supervisor
    rubric_scores: RubricEvaluation

    # Control de Human-in-the-loop (HITL)
    is_critical: bool
    requires_human_approval: bool
    approval_status: ApprovalStatus
    approval_request: Optional[Dict[str, Any]]
    approval_response: Optional[Dict[str, Any]]

    # Estado de finalización y reporte final
    task_completed: bool
    final_report: Optional[str]

    # Metadatos de observabilidad (tiempos, costo, modelo)
    metadata: Dict[str, Any]


def create_initial_state(
    user_query: str,
    task_id: str = "",
    require_approval: bool = False,
    max_iterations: int = 5,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentState:
    """
    Inicializa un estado seguro para una nueva tarea en el orquestador.
    """
    return {
        "messages": [],
        "task_id": task_id,
        "user_query": user_query,
        "next_agent": "research_agent",
        "supervisor_feedback": "Inicio de la tarea. Investigar datos clave de la consulta.",
        "research_data": {},
        "analysis_data": {},
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "rubric_scores": {
            "research_sufficiency": 0.0,
            "analysis_rigor": 0.0,
            "synthesis_ready": False,
            "critique": "Estado inicial creado.",
        },
        "is_critical": False,
        "requires_human_approval": require_approval,
        "approval_status": "PENDING" if require_approval else "NOT_REQUIRED",
        "approval_request": None,
        "approval_response": None,
        "task_completed": False,
        "final_report": None,
        "metadata": metadata or {},
    }

