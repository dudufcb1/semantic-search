# MCP Codebase Search Server (Python + FastMCP)

Servidor Model Context Protocol que reutiliza el índice semántico existente de Roo Code y expone
los tools `superior_codebase_search` y `superior_codebase_rerank` vía transporte `stdio`.

**Migrado de TypeScript a Python usando FastMCP.**

## Requisitos

- Python 3.10+
- `uv` (recomendado) o `pip`
- Acceso a la carpeta del workspace ya indexado por Roo Code
- Qdrant accesible con la colección creada por la extensión
- API key para el proveedor de embeddings compatible con OpenAI
- API key para el proveedor del Judge (LLM) compatible con OpenAI

## Instalación

### Opción 1: Usando `uv` (recomendado)

```bash
cd python-mcp
uv sync
```

### Opción 2: Usando `pip`

```bash
cd python-mcp
pip install -e .
```

## Configuración

Variables de entorno soportadas (crear archivo `.env` en la raíz del proyecto):

```bash
# Workspace por defecto (opcional)
MCP_CODEBASE_WORKSPACE=/ruta/absoluta/al/workspace
# o
WORKSPACE_PATH=/ruta/absoluta/al/workspace

# Qdrant (requerido)
MCP_CODEBASE_QDRANT_URL=http://localhost:6333
# o
QDRANT_URL=http://localhost:6333

MCP_CODEBASE_QDRANT_API_KEY=tu-api-key-opcional
# o
QDRANT_API_KEY=tu-api-key-opcional

# Embedder (requerido)
MCP_CODEBASE_EMBEDDER_PROVIDER=openai
# o
EMBEDDER_PROVIDER=openai

MCP_CODEBASE_EMBEDDER_API_KEY=sk-...
# o
EMBEDDER_API_KEY=sk-...

MCP_CODEBASE_EMBEDDER_MODEL_ID=text-embedding-3-small
# o
EMBEDDER_MODEL_ID=text-embedding-3-small

# Para proveedores compatibles con OpenAI
MCP_CODEBASE_EMBEDDER_BASE_URL=https://api.tu-proveedor.com/v1
# o
EMBEDDER_BASE_URL=https://api.tu-proveedor.com/v1

# Judge/LLM (requerido)
MCP_CODEBASE_JUDGE_PROVIDER=openai
# o
JUDGE_PROVIDER=openai

MCP_CODEBASE_JUDGE_API_KEY=sk-...
# o
JUDGE_API_KEY=sk-...

MCP_CODEBASE_JUDGE_MODEL_ID=gpt-4o-mini
# o
JUDGE_MODEL_ID=gpt-4o-mini

MCP_CODEBASE_JUDGE_MAX_TOKENS=1024
# o
JUDGE_MAX_TOKENS=1024

MCP_CODEBASE_JUDGE_TEMPERATURE=0
# o
JUDGE_TEMPERATURE=0

# Para proveedores compatibles con OpenAI
MCP_CODEBASE_JUDGE_BASE_URL=https://api.tu-proveedor.com/v1
# o
JUDGE_BASE_URL=https://api.tu-proveedor.com/v1

# Reranking (opcional)
MCP_CODEBASE_RERANKING_MAX_RESULTS=10
# o
RERANKING_MAX_RESULTS=10

MCP_CODEBASE_RERANKING_INCLUDE_REASON=true
# o
RERANKING_INCLUDE_REASON=true

MCP_CODEBASE_RERANKING_SUMMARIZE=false
# o
RERANKING_SUMMARIZE=false

# Search (opcional)
MCP_CODEBASE_SEARCH_MIN_SCORE=0.4
# o
SEARCH_MIN_SCORE=0.4

MCP_CODEBASE_SEARCH_MAX_RESULTS=20
# o
SEARCH_MAX_RESULTS=20
```

## Ejecución

### Usando FastMCP CLI

```bash
cd python-mcp
fastmcp run
```

O especificando el archivo de configuración:

```bash
fastmcp run fastmcp.json
```

### Usando Python directamente

```bash
cd python-mcp
python src/server.py
```

## Configuración en Claude Desktop / Cline

Edita tu archivo de configuración MCP (ej: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "codebase-search": {
      "command": "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/.venv/bin/fastmcp",
      "args": ["run"],
      "cwd": "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp",
      "alwaysAllow": ["superior_codebase_search", "superior_codebase_rerank"]
    }
  }
}
```

**Nota:** Cambia las rutas a la ubicación donde clonaste el repositorio.

O usando el script de inicio (más robusto):

```json
{
  "mcpServers": {
    "codebase-search": {
      "command": "/ruta/completa/a/python-mcp/start.sh",
      "args": [],
      "cwd": "/ruta/completa/a/python-mcp",
      "alwaysAllow": ["superior_codebase_search", "superior_codebase_rerank"]
    }
  }
}
```

**Nota:** Asegúrate de que `uv` esté instalado y en el PATH del sistema. Si tienes problemas, usa el script `start.sh`.

## Tools disponibles

### `superior_codebase_search`

Búsqueda semántica en código indexado.

**Parámetros:**
- `query` (string, obligatorio): Texto natural a buscar
- `workspacePath` (string, opcional): Ruta absoluta del workspace
- `path` (string, opcional): Prefijo de ruta para filtrar resultados

**Ejemplo:**
```json
{
  "query": "función de autenticación",
  "workspacePath": "/home/user/my-project",
  "path": "src/auth"
}
```

### `superior_codebase_rerank`

Búsqueda semántica con reordenamiento inteligente usando LLM.

**Parámetros:**
- `query` (string, obligatorio): Texto natural a buscar
- `workspacePath` (string, opcional): Ruta absoluta del workspace
- `path` (string, opcional): Prefijo de ruta para filtrar resultados
- `mode` (string, opcional): "rerank" (default) o "summary"

**Ejemplo:**
```json
{
  "query": "manejo de errores",
  "workspacePath": "/home/user/my-project",
  "mode": "summary"
}
```

## Estructura del Proyecto

```
python-mcp/
├── fastmcp.json          # Configuración FastMCP
├── pyproject.toml        # Dependencias Python
├── README.md             # Este archivo
├── .env                  # Variables de entorno (no versionado)
├── src/
│   ├── server.py         # Servidor FastMCP principal
│   ├── config.py         # Configuración con Pydantic
│   ├── embedder.py       # Cliente de embeddings
│   ├── judge.py          # Cliente LLM para reranking
│   └── qdrant_store.py   # Cliente Qdrant
```

## Diferencias con la versión TypeScript

- ✅ Mismo comportamiento funcional
- ✅ Mismas variables de entorno
- ✅ Mismos tools y parámetros
- ✅ Async/await nativo en Python
- ✅ Gestión de dependencias con `uv`
- ✅ Configuración declarativa con `fastmcp.json`
- ✅ Mejor manejo de errores con `ToolError`

## Desarrollo

### Ejecutar tests

```bash
cd python-mcp
pytest
```

### Verificar configuración

```bash
cd python-mcp
python -c "from src.config import settings; print(settings.model_dump())"
```

