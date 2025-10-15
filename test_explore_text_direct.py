"""Demo de la nueva herramienta explore_other_workspace con TextDirectJudge."""

print("="*80)
print("NUEVA HERRAMIENTA: explore_other_workspace")
print("="*80)

print("""
🔥 FUNCIONALIDAD IMPLEMENTADA:

1. ✅ Nueva herramienta `explore_other_workspace` en ambos servidores
2. ✅ Prioridad de parámetros: qdrant_collection > workspace_path
3. ✅ Formato de texto directo (no JSON) - EXPERIMENTAL
4. ✅ Rerank por defecto (eliminando problemas de parsing)

📋 PARÁMETROS:
- query: string (requerido) - Búsqueda a realizar
- qdrant_collection: string (opcional) - Nombre de colección explícito (PRIORIDAD ALTA)
- workspace_path: string (opcional) - Ruta del workspace (se calcula colección)
- path: string (opcional) - Prefijo de ruta para filtrar resultados
- mode: "rerank" | "summary" (default: "rerank")

🎯 EJEMPLOS DE USO:

# Servidor principal (server.py)
superior_codebase_search(
    query="authentication implementation",
    qdrant_collection="codebase-abc123"  # Usa directamente esta colección
)

# Si tienes ambos parámetros, prioriza qdrant_collection
superior_codebase_search(
    query="login function",
    qdrant_collection="priority-collection",  # ✅ USA ESTE
    workspace_path="/ignored/path"            # ❌ IGNORADO
)

# Solo workspace_path (calcula colección automáticamente)
superior_codebase_search(
    query="error handling",
    workspace_path="/media/eduardo/56087475087455C9/Dev/independent-embeddings-indexer"
)

# Servidor local (server_local.py) - Mismo comportamiento
superior_codebase_rerank(
    query="database connection",
    qdrant_collection="codebase-local-xyz789"
)

🧪 FORMATO MEJORADO - Text Direct vs JSON:

ANTES (JSON Judge):
{
  "reranked": [
    {
      "filePath": "src/auth.py",
      "relevancia": 0.95,
      "razon": "Contains authentication logic"
    }
  ]
}

AHORA (Text Direct Judge - MEJORADO):
FILE: src/auth.py
RELEVANCE: 0.95
CODE_SNIPPET: def authenticate_user(username, password):
REASON: Contains the main authentication implementation

FILE: src/login.py
RELEVANCE: 0.87
CODE_SNIPPET: login_handler = (req, res) => {
REASON: Handles login requests and validation

FILE: src/session.py
RELEVANCE: 0.82
CODE_SNIPPET: class SessionManager:
REASON: Manages user session lifecycle

VENTAJAS del formato mejorado:
✅ Sin problemas de JSON malformado
✅ CODE_SNIPPET muestra líneas de código relevantes
✅ No vas "a ciegas" - ves exactamente qué código es relevante
✅ Parsing más simple y confiable
✅ Formato en inglés más consistente
✅ Información más rica para tomar decisiones

🔬 COMPARACIÓN DE RENDIMIENTO:

explore_other_workspace (NUEVO):
- ✅ TextDirectJudge (experimental)
- ✅ Sin fallbacks complejos de JSON
- ✅ Rerank siempre habilitado
- ✅ Formato de salida más limpio

superior_codebase_rerank (EXISTENTE):
- ❌ Judge con JSON parsing
- ❌ Múltiples fallbacks por JSON malformado
- ❌ Complejidad adicional de manejo de errores

📊 PRUEBAS RECOMENDADAS:

1. Test con colección explícita:
   explore_other_workspace(
       query="function definition",
       qdrant_collection="codebase-test123"
   )

2. Test con workspace path:
   explore_other_workspace(
       query="import statements",
       workspace_path="/media/eduardo/56087475087455C9/Dev/independent-embeddings-indexer"
   )

3. Test de prioridades:
   explore_other_workspace(
       query="class definition",
       qdrant_collection="priority-col",  # ✅ Se usa este
       workspace_path="/ignored/path"     # ❌ Se ignora
   )

4. Test con summary:
   explore_other_workspace(
       query="error handling patterns",
       qdrant_collection="codebase-xyz",
       mode="summary"
   )

🚀 PRÓXIMOS PASOS:

1. Probar consistencia del TextDirectJudge vs Judge normal
2. Evaluar calidad de resultados
3. Si funciona bien, migrar otras herramientas al formato de texto
4. Documentar las mejoras en robustez

💡 NOTAS TÉCNICAS:

- TextDirectJudge usa el mismo endpoint que Judge normal
- System prompt en INGLÉS para mayor consistencia
- NUEVO: Campo CODE_SNIPPET obligatorio con líneas de código relevantes
- Parsing con regex simple en lugar de JSON.loads()
- Compatibilidad con español para transición (RELEVANCIA/RAZON)
- Fallback a resultados originales si TextDirectJudge falla
- Compatible con ambos servidores (server.py y server_local.py)
- Formato: FILE: | RELEVANCE: | CODE_SNIPPET: | REASON:
""")

print("="*80)
print("IMPLEMENTACIÓN COMPLETADA ✅")
print("="*80)