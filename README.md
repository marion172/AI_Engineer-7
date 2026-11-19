# API REST de Producción y Monitoreo Activo - Sistema Multi-Agente

API REST asíncrona de alto rendimiento construida con **FastAPI** para exponer el sistema multi-agente jerárquico desarrollado en LangGraph, respaldada por **Redis** para la persistencia del estado de los trabajos y checkpoints de ejecución, instrumentación activa con **LangSmith** para observabilidad y un flujo **Human-in-the-loop (HITL)** para control y aprobación obligatoria en tareas críticas.

---

## ¿Qué problema resuelve este sistema?

En un entorno de producción, ejecutar sistemas multi-agente complejos presenta tres grandes desafíos:
1. **Bloqueo del servidor:** Las llamadas a múltiples LLMs y herramientas toman varios segundos. Si la API fuese síncrona, cada petición bloquearía el servidor dejando de responder a otros usuarios.
2. **Pérdida de estado:** Si la aplicación se reinicia o falla un nodo intermedio, el progreso se pierde y el cliente queda en espera infinita.
3. **Falta de visibilidad y control sobre acciones de riesgo:** Se necesita auditar qué nodos consumen más tokens, cuál es la latencia real (percentil 95), cuánto cuesta cada corrida en dólares y pausar obligatoriamente la ejecución ante acciones críticas (ej. órdenes financieras) para que un operador humano dé el visto bueno.

Esta solución desacopla la recepción del trabajo de su procesamiento, persiste todo el ciclo de vida en Redis, monitorea cada llamada en LangSmith y provee un punto de interrupción seguro para intervención humana.

---

## Arquitectura del Sistema

```
                      ┌────────────────────────────────────────┐
                      │             Cliente / HTTP             │
                      └──────────────────┬─────────────────────┘
                                         │ (POST /tasks < 15ms)
                                         ▼
                      ┌────────────────────────────────────────┐
                      │            FastAPI App                 │
                      │  - POST /tasks → Encola y devuelve ID  │
                      │  - GET /tasks/{id} → Consulta estado   │
                      │  - POST /tasks/{id}/approve → HITL     │
                      └──────────────┬──────────────────┬──────┘
                                     │                  │
                (Guarda estado       │                  │ (Despacha tarea
                 y checkpoints)      ▼                  ▼  en segundo plano)
        ┌───────────────────────────────┐      ┌─────────────────────────────┐
        │         Redis Storage         │◄─────┤     Asynchronous Worker     │
        │ - task:{id} (Estado del Job)  │      │  (Background Task Handler)  │
        │ - Checkpoints de LangGraph    │      └──────────────┬──────────────┘
        └───────────────────────────────┘                     │
                                                              ▼
                                               ┌─────────────────────────────┐
                                               │    LangGraph Orchestrator   │
                                               │  (Supervisor + Especialistas)│
                                               │   OpenRouter NVIDIA LLM     │
                                               └──────────────┬──────────────┘
                                                              │
                                                              ▼
                                               ┌─────────────────────────────┐
                                               │       LangSmith Cloud       │
                                               │  - Trazas de Nodos          │
                                               │  - Costo por Ejecución ($)  │
                                               │  - Latencia p95 & Tokens    │
                                               │  - Pausa y Reanudación HITL │
                                               └─────────────────────────────┘
```

---

## Componentes Principales

### 1. Endpoints Asíncronos (`app/main.py`)
- **`POST /tasks`**: Recibe la directiva, genera un `job_id`, guarda el estado inicial en Redis como `PENDING`, despacha el procesamiento en segundo plano (`asyncio.create_task`) y responde inmediatamente (**< 15 ms**) con código HTTP `202 Accepted`.
- **`GET /tasks/{id}`**: Endpoint no bloqueante para consultar el estado del trabajo (`PENDING`, `RUNNING`, `WAITING_APPROVAL`, `COMPLETED`, `FAILED`), tiempos de respuesta y resultados.
- **`POST /tasks/{id}/approve`**: Recibe la aprobación o rechazo humano para tareas en pausa y reanuda el grafo.
- **`GET /health`**: Verificación de conectividad con Redis, modelo LLM activo y observabilidad.

### 2. Persistencia y Resiliencia con Redis (`app/worker.py`)
- Almacenamiento centralizado del estado de cada trabajo (`task:{id}`).
- Persistencia de checkpoints del grafo mediante checkpointer de Redis, permitiendo pausar y reanudar flujos sin perder el contexto.
- **Gestión ante fallos:** Cualquier excepción ocurrida en segundo plano es capturada, actualizando el estado inmediatamente a `FAILED` con su stacktrace para evitar loops infinitos de polling en el cliente.

### 3. Flujo Human-in-the-loop (`app/hitl.py`)
- Detección de operaciones críticas (ej. órdenes de compra/venta de activos, rebalanceos de cartera o parámetro explícito `require_approval=True`).
- El nodo `human_approval_node` ejecuta una interrupción nativa (`interrupt`), congelando el grafo en Redis y cambiando el estado a `WAITING_APPROVAL`.
- Al recibir `POST /tasks/{id}/approve`, se inyecta `Command(resume=...)` reactivando la ejecución hasta el informe final.

### 4. Monitoreo Activo con LangSmith (`app/observability.py`)
- Registro de cada invocación de nodo (`supervisor`, `research_agent`, `analyst_agent`, `synthesizer`).
- Medición de la latencia acumulada y percentil 95 (p95).
- Cálculo automático del costo por ejecución en base a tokens de entrada y salida.

---

## Organización del Proyecto

```text
multi-agent-api/
├── app/
│   ├── main.py            # API FastAPI (endpoints asíncronos y Swagger docs)
│   ├── graph.py           # Orquestador LangGraph + checkpoints en Redis + HITL
│   ├── worker.py          # Worker en segundo plano y persistencia de estados
│   ├── observability.py   # Configuración de trazabilidad con LangSmith
│   ├── hitl.py            # Lógica de interrupción y reanudación de tareas críticas
│   ├── state.py           # Esquema de estado compartido (TypedDict)
│   ├── agents/            # Supervisor, Investigador, Analista y Síntesis
│   └── tools/             # Herramientas de cálculo analítico y búsqueda externa
├── screenshots/           # Capturas de métricas y observabilidad del sistema
├── run_load_test.py       # Script de benchmarking y prueba de concurrencia
├── docker-compose.yml     # Orquestación de contenedores (API + Redis)
├── Dockerfile             # Contenedor de la aplicación
├── requirements.txt       # Dependencias del proyecto
├── .env.example           # Plantilla de variables de entorno
├── .gitignore
└── README.md              # Documentación del sistema
```

---

## Instalación y Ejecución

### Opción 1: Con Docker Compose (Recomendado)

1. **Configurar el archivo `.env`:**
   ```bash
   cp .env.example .env
   ```
   Completar con las claves:
   ```env
   OPENAI_API_KEY=openrouter_api_key
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   OPENAI_MODEL=nvidia/nemotron-3.5-lightning:free
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=langsmith_api_key
   LANGCHAIN_PROJECT=pre-entrega-7-api
   ```

2. **Levantar los servicios:**
   ```bash
   docker-compose up -d
   ```

3. **Acceder a la documentación interactiva (Swagger UI):**
   👉 **http://localhost:8000/docs**

---

### Opción 2: Ejecución Local en Entorno Virtual

1. **Crear y activar el entorno virtual:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Iniciar Redis y la API:**
   ```bash
   # Iniciar Redis (por ejemplo vía Docker):
   docker run -d --name redis_local -p 6379:6379 redis:7-alpine

   # Iniciar la API con Uvicorn:
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

---

## Prueba de Carga y Concurrencia

El repositorio incluye un script para evaluar el comportamiento asíncrono y la respuesta del sistema bajo concurrencia:

```bash
python run_load_test.py
```

El script realiza lo siguiente:
1. Despacha 5 consultas simultáneas contra `POST /tasks` (demostrando encolado inmediato sin bloqueo).
2. Monitorea el progreso de cada una en segundo plano.
3. Detecta la tarea crítica que entra en pausa HITL y envía la aprobación correspondiente vía `POST /tasks/{id}/approve`.
4. Calcula y presenta en consola las métricas de latencia de la corrida (Mínima, Media, Mediana y Percentil 95).

---
