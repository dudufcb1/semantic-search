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

1. Clonar el repositorio:
```bash
git clone https://github.com/dudufcb1/llm_codebase_search_python.git
cd llm_codebase_search_python/python-mcp
```

2. Crear entorno virtual e instalar dependencias:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install fastmcp qdrant-client httpx pydantic pydantic-settings python-dotenv
```

3. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tu configuración
```

## Configuración

Variables de entorno soportadas (crear archivo `.env` en la raíz del proyecto):

```bash
# Workspace por defecto (opcional)
MCP_CODEBASE_WORKSPACE=/ruta/absoluta/al/workspace
# o
WORKSPACE_PATH=/ruta/absoluta/al/workspace

# Modo de colección (opcional - default: "default")
# "default" = calcula colección con hash de workspace_path
# "codebase-indexer" = requiere qdrantCollection explícito del agente
MCP_CODEBASE_COLLECTION_SOURCE=default

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

## Integración con Claude Desktop

Este proyecto ofrece **DOS servidores MCP** para diferentes casos de uso:

### Servidor 1: `server.py` (DEFAULT MODE)

**Uso:** Para IDEs/clientes que NO usan `code-index-cli` y calculan el nombre de colección automáticamente usando hash SHA256 del workspace path.

**Tools disponibles:**
- `superior_codebase_search` - Búsqueda semántica simple
- `superior_codebase_rerank` - Búsqueda con reordenamiento LLM

**Configuración JSON (Claude Desktop):**

```json
{
  "mcpServers": {
    "codebase-search": {
      "command": "/ruta/absoluta/a/python-mcp/venv/bin/python",
      "args": [
        "/ruta/absoluta/a/python-mcp/src/server.py"
      ],
      "alwaysAllow": [
        "superior_codebase_search",
        "superior_codebase_rerank"
      ]
    }
  }
}
```

**Ejemplo real:**
```json
{
  "mcpServers": {
    "codebase-search": {
      "command": "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/venv/bin/python",
      "args": [
        "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/src/server.py"
      ],
      "alwaysAllow": [
        "superior_codebase_search",
        "superior_codebase_rerank"
      ]
    }
  }
}
```

### Servidor 2: `server_local.py` (LOCAL MODE - code-index-cli compatible)

**Uso:** Para clientes que usan `code-index-cli` para indexar y mantener el índice actualizado en tiempo real. El nombre de colección se lee desde `.codebase/state.json`.

**Tools disponibles:**
- `superior_codebase_rerank` - SOLO rerank (requiere `qdrantCollection` explícito)

**Configuración JSON (Claude Desktop):**

```json
{
  "mcpServers": {
    "codebase-search-local": {
      "command": "/ruta/absoluta/a/python-mcp/venv/bin/python",
      "args": [
        "/ruta/absoluta/a/python-mcp/src/server_local.py"
      ],
      "alwaysAllow": [
        "superior_codebase_rerank"
      ]
    }
  }
}
```

**Ejemplo real:**
```json
{
  "mcpServers": {
    "codebase-search-local": {
      "command": "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/venv/bin/python",
      "args": [
        "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/src/server_local.py"
      ],
      "alwaysAllow": [
        "superior_codebase_rerank"
      ]
    }
  }
}
```

### Configuración TOML (Cline, Roo Coder, etc.)

**Servidor DEFAULT (`server.py`):**

```toml
[mcp_servers.codebase-search]
type = "stdio"
command = "/ruta/absoluta/a/python-mcp/venv/bin/python"
args = ["/ruta/absoluta/a/python-mcp/src/server.py"]
timeout = 3600
```

**Ejemplo real:**
```toml
[mcp_servers.codebase-search]
type = "stdio"
command = "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/venv/bin/python"
args = ["/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/src/server.py"]
timeout = 3600
```

**Servidor LOCAL (`server_local.py`):**

```toml
[mcp_servers.codebase-search-local]
type = "stdio"
command = "/ruta/absoluta/a/python-mcp/venv/bin/python"
args = ["/ruta/absoluta/a/python-mcp/src/server_local.py"]
timeout = 3600
```

**Ejemplo real:**
```toml
[mcp_servers.codebase-search-local]
type = "stdio"
command = "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/venv/bin/python"
args = ["/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp/src/server_local.py"]
timeout = 3600
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

### Servidor DEFAULT (`server.py`)

#### `superior_codebase_search`

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

#### `superior_codebase_rerank`

Búsqueda semántica con reordenamiento inteligente usando LLM.

**Parámetros:**
- `query` (string, obligatorio): Texto natural a buscar
- `workspacePath` (string, obligatorio): Ruta absoluta del workspace
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

### Servidor LOCAL (`server_local.py`)

#### `superior_codebase_rerank`

Búsqueda semántica con reordenamiento inteligente usando LLM. **Requiere colección explícita desde `.codebase/state.json`.**

**Parámetros:**
- `query` (string, obligatorio): Texto natural a buscar
- `qdrantCollection` (string, obligatorio): Nombre de colección Qdrant desde `.codebase/state.json`
- `path` (string, opcional): Prefijo de ruta para filtrar resultados
- `mode` (string, opcional): "rerank" (default) o "summary"

**Ejemplo:**
```json
{
  "query": "authentication logic",
  "qdrantCollection": "codebase-f93e99958acc444e"
}
```

**Workflow del agente:**
1. Leer `.codebase/state.json` del workspace
2. Extraer el campo `qdrantCollection`
3. Pasar ese valor en el tool call

## Estructura del Proyecto

```
python-mcp/
├── fastmcp.json          # Configuración FastMCP
├── pyproject.toml        # Dependencias Python
├── README.md             # Este archivo
├── .env                  # Variables de entorno (no versionado)
├── src/
│   ├── server.py         # Servidor DEFAULT (hash automático)
│   ├── server_local.py   # Servidor LOCAL (code-index-cli compatible)
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

## Compatibilidad con code-index-cli

Este servidor MCP puede operar en dos modos según la variable de entorno `MCP_CODEBASE_COLLECTION_SOURCE`:

### Modo 1: `default` (por defecto)

El servidor calcula automáticamente el nombre de colección usando SHA256 de la ruta del workspace.

**Configuración (.env):**
```bash
MCP_CODEBASE_COLLECTION_SOURCE=default  # o simplemente omitir esta variable
```

**Uso de los tools:**
```json
{
  "query": "authentication logic",
  "workspacePath": "/home/user/my-project"
}
```

El servidor automáticamente:
1. Normaliza la ruta del workspace
2. Calcula SHA256 de la ruta
3. Genera nombre de colección: `ws-{hash[:16]}`

### Modo 2: `codebase-indexer`

Compatible con [code-index-cli](https://github.com/tu-repo/code-index-cli) que mantiene el índice actualizado en tiempo real.

**Configuración (.env):**
```bash
MCP_CODEBASE_COLLECTION_SOURCE=codebase-indexer
```

**Uso de los tools:**

El agente debe leer el archivo `.codebase/state.json` del workspace y pasar el `qdrantCollection`:

1. **Localizar:** `<workspace>/.codebase/state.json`
2. **Leer el contenido:**
```json
{
  "workspacePath": "/home/user/my-project",
  "qdrantCollection": "codebase-f93e99958acc444e",
  "createdAt": "2025-10-13T08:09:19.732Z",
  "updatedAt": "2025-10-13T20:08:42.023Z"
}
```
3. **Pasar el `qdrantCollection` en el tool call:**
```json
{
  "query": "authentication logic",
  "qdrantCollection": "codebase-f93e99958acc444e"
}
```

**¿Por qué este modo?**
- `code-index-cli` corre en una pestaña separada de Claude Code/Codex manteniendo el índice actualizado
- Tu IDE usa este MCP para buscar en el índice en tiempo real
- No necesitas mantener dos índices separados
- El nombre de colección es generado una vez por `code-index-cli` (UUID random) y guardado en `.codebase/state.json`

**Importante:**
- En modo `codebase-indexer`, el parámetro `workspacePath` se ignora
- El parámetro `qdrantCollection` es **obligatorio** en este modo
- Si no se proporciona `qdrantCollection`, el servidor retorna un error instructivo

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

