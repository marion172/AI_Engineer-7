"""
run_load_test.py
================
Script de prueba de carga y concurrencia.
Envía 5 peticiones concurrentes contra la API REST (FastAPI),
mide la latencia de respuesta inmediata (asincronía) y la latencia end-to-end,
y calcula los percentiles estadísticos de latencia (p50, p95).

Soporta ejecución asíncrona nativa con 'httpx' o fallback con 'urllib'.
"""

import asyncio
import json
import statistics
import time
from typing import Any, Dict, List

API_BASE_URL = "http://127.0.0.1:8000"

TEST_QUERIES = [
    {
        "query": "Analizar ingresos de NVIDIA en Q2 fiscal 2025 y comparar con AMD",
        "require_approval": False,
        "label": "Test 1: NVIDIA vs AMD",
    },
    {
        "query": "Evaluar crecimiento del segmento Data Center de AMD y su impacto en GPU",
        "require_approval": False,
        "label": "Test 2: AMD Data Center",
    },
    {
        "query": "Revisar costos y consumo energético en clusters H100 para entrenamiento AI",
        "require_approval": False,
        "label": "Test 3: AI Compute Economics",
    },
    {
        "query": "Analizar el CapEx combinado de los hyperscalers Big Tech en 2024",
        "require_approval": False,
        "label": "Test 4: Hyperscalers CapEx",
    },
    {
        "query": "Comprar acciones y rebalancear cartera tecnológica con alta criticidad",
        "require_approval": True,
        "label": "Test 5: Operacion Critica (HITL)",
    },
]


# ==============================================================================
# Capa de transporte HTTP (con fallback a urllib nativo)
# ==============================================================================

try:
    import httpx
    USE_HTTPX = True
except ImportError:
    USE_HTTPX = False
    import urllib.request
    import urllib.error


def _sync_request(method: str, url: str, data: Any = None) -> tuple[int, Dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status_code = resp.status
            content = json.loads(resp.read().decode("utf-8"))
            return status_code, content
    except urllib.error.HTTPError as e:
        try:
            content = json.loads(e.read().decode("utf-8"))
        except Exception:
            content = {"detail": str(e)}
        return e.code, content
    except Exception as e:
        return 500, {"error": str(e)}


async def http_get(url: str, client: Any = None) -> tuple[int, Dict[str, Any]]:
    if USE_HTTPX and client:
        res = await client.get(url, timeout=15.0)
        return res.status_code, res.json()
    else:
        return await asyncio.to_thread(_sync_request, "GET", url)


async def http_post(url: str, payload: Dict[str, Any], client: Any = None) -> tuple[int, Dict[str, Any]]:
    if USE_HTTPX and client:
        res = await client.post(url, json=payload, timeout=15.0)
        return res.status_code, res.json()
    else:
        return await asyncio.to_thread(_sync_request, "POST", url, payload)


# ==============================================================================
# Lógica de la prueba de carga
# ==============================================================================

async def dispatch_task(item: Dict[str, Any], client: Any = None) -> Dict[str, Any]:
    start = time.perf_counter()
    status_code, data = await http_post(
        f"{API_BASE_URL}/tasks",
        payload={
            "query": item["query"],
            "require_approval": item["require_approval"],
            "max_iterations": 4,
        },
        client=client,
    )
    dispatch_time = time.perf_counter() - start

    if status_code == 202:
        return {
            "label": item["label"],
            "job_id": data["job_id"],
            "status": data["status"],
            "dispatch_latency_ms": round(dispatch_time * 1000, 2),
            "require_approval": item["require_approval"],
        }
    else:
        return {
            "label": item["label"],
            "job_id": None,
            "status": f"ERROR_{status_code}",
            "dispatch_latency_ms": round(dispatch_time * 1000, 2),
            "require_approval": item["require_approval"],
        }


async def poll_task_until_done(task_info: Dict[str, Any], client: Any = None) -> Dict[str, Any]:
    job_id = task_info.get("job_id")
    if not job_id:
        return {"error": "Sin job_id valido"}

    start_polling = time.perf_counter()
    timeout_seconds = 180.0

    while (time.perf_counter() - start_polling) < timeout_seconds:
        status_code, data = await http_get(f"{API_BASE_URL}/tasks/{job_id}", client=client)
        if status_code == 200:
            current_status = data.get("status")

            if current_status == "WAITING_APPROVAL":
                print(f"   [HITL] Tarea {job_id} en WAITING_APPROVAL. Enviando aprobacion...")
                approve_status, approve_data = await http_post(
                    f"{API_BASE_URL}/tasks/{job_id}/approve",
                    payload={"approved": True, "feedback": "Aprobado por proceso automatizado de prueba."},
                    client=client,
                )
                print(f"   [HITL] Respuesta de aprobacion: HTTP {approve_status}")
                await asyncio.sleep(1.0)
                continue

            if current_status in ("COMPLETED", "FAILED"):
                total_duration = time.perf_counter() - start_polling
                return {
                    **task_info,
                    "final_status": current_status,
                    "execution_time_seconds": round(total_duration, 3),
                    "server_reported_time": data.get("execution_time_seconds"),
                    "result_preview": (data.get("result") or data.get("error") or "")[:120].replace("\n", " "),
                }

        await asyncio.sleep(1.0)

    return {
        **task_info,
        "final_status": "TIMEOUT",
        "execution_time_seconds": round(time.perf_counter() - start_polling, 3),
    }


def calculate_p95(latencies: List[float]) -> float:
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    idx = int(0.95 * len(sorted_latencies))
    if idx >= len(sorted_latencies):
        idx = len(sorted_latencies) - 1
    return sorted_latencies[idx]


async def run_load_test():
    engine_name = "httpx (asincrono)" if USE_HTTPX else "urllib (biblioteca estandar)"
    print("\n" + "=" * 70)
    print("INICIANDO PRUEBA DE CONCURRENCIA: 5 PETICIONES SIMULTANEAS")
    print(f"Destino: {API_BASE_URL}")    
    print("=" * 70)

    client = httpx.AsyncClient() if USE_HTTPX else None

    try:
        # 1. Comprobar salud de la API
        try:
            status_code, health = await http_get(f"{API_BASE_URL}/health", client=client)
            if status_code != 200:
                print(f"[ERROR] La API no respondio HTTP 200 en /health: {status_code}")
                return
            print("Conexion con API verificada:")
            print(f"   - Modelo: {health.get('model')}")
            print(f"   - Redis:  {health.get('redis', {}).get('mode')}")
            obs = health.get("observability", {})
            print(f"   - Observabilidad: {obs.get('provider')} (Activo: {obs.get('langsmith_active')})")
        except Exception as e:
            print(f"[ERROR] No se pudo conectar a la API en {API_BASE_URL}: {e}")
            return

        # 2. Encolar 5 peticiones concurrentes
        print("\nEncolando 5 peticiones en paralelo (POST /tasks)...")
        dispatch_start = time.perf_counter()
        dispatch_tasks = [dispatch_task(item, client=client) for item in TEST_QUERIES]
        dispatched_results = await asyncio.gather(*dispatch_tasks)
        total_dispatch_time = time.perf_counter() - dispatch_start

        print(f"5 peticiones encoladas en {total_dispatch_time * 1000:.2f} ms (promedio: {total_dispatch_time / 5 * 1000:.2f} ms/req)")
        for res in dispatched_results:
            print(f"   - [{res['label']}] Job ID: {res['job_id']} | Status: {res['status']} | Latencia Encolado: {res['dispatch_latency_ms']} ms")

        # 3. Monitorear finalización concurrente
        print("\nMonitoreando finalizacion de tareas en segundo plano...")
        poll_tasks = [poll_task_until_done(info, client=client) for info in dispatched_results]
        completed_results = await asyncio.gather(*poll_tasks)

    finally:
        if client:
            await client.aclose()

    # 4. Análisis de Métricas
    print("\n" + "=" * 70)
    print("RESULTADOS DE LA PRUEBA DE CONCURRENCIA")
    print("=" * 70)

    latencies = [r.get("execution_time_seconds", 0.0) for r in completed_results if r.get("final_status") in ("COMPLETED", "WAITING_APPROVAL")]
    
    for r in completed_results:
        print(f"[{r.get('label')}]")
        print(f"   - Job ID: {r.get('job_id')}")
        print(f"   - Estado: {r.get('final_status')}")
        print(f"   - Duracion: {r.get('execution_time_seconds')}s")
        print(f"   - Salida: {r.get('result_preview', 'N/A')[:90]}...\n")

    if latencies:
        mean_lat = statistics.mean(latencies)
        median_lat = statistics.median(latencies)
        p95_lat = calculate_p95(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)

        print("-" * 70)
        print("METRICAS DE LATENCIA:")
        print(f"   - Total peticiones:       {len(latencies)}")
        print(f"   - Latencia Minima:        {min_lat:.3f} s")
        print(f"   - Latencia Media:         {mean_lat:.3f} s")
        print(f"   - Latencia Mediana (p50): {median_lat:.3f} s")
        print(f"   - Latencia p95:           {p95_lat:.3f} s")
        print(f"   - Latencia Maxima:        {max_lat:.3f} s")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_load_test())
