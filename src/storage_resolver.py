"""Helper para resolver qué storage usar (SQLite o Qdrant) basado en parámetros y disponibilidad."""

import sys
from pathlib import Path
from typing import Literal, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StorageResolution:
    """Resultado de resolver qué storage usar."""
    storage_type: Literal["sqlite", "qdrant"]
    """Tipo de storage a usar: 'sqlite' o 'qdrant'"""
    
    identifier: str
    """Identificador legible del target (para logging/mensajes)"""
    
    sqlite_path: Optional[Path] = None
    """Ruta al archivo SQLite si storage_type='sqlite'"""
    
    qdrant_collection: Optional[str] = None
    """Nombre de colección Qdrant si storage_type='qdrant'"""
    
    fallback_used: bool = False
    """True si se usó fallback (ej: SQLite no existe, se usa Qdrant)"""
    
    fallback_reason: Optional[str] = None
    """Razón del fallback si se usó"""


class StorageResolver:
    """Resuelve qué storage usar basado en parámetros y disponibilidad.

    Lógica de resolución (en orden de prioridad):

    1. Si `qdrant_collection` está especificado explícitamente:
       → Usar Qdrant con esa colección (prioridad máxima)

    2. Si `workspace_path` está especificado:
       a. Buscar colección Qdrant en `.codebase/` (archivos codebase-* o ws-*)
          → Si existe: usar Qdrant con esa colección

       b. Buscar SQLite en `.codebase/vectors.db`
          → Si existe: usar SQLite

       c. Calcular colección Qdrant desde workspace_path (roo-code style)
          → Usar Qdrant con colección calculada

    3. Si nada está especificado:
       → Error: se requiere al menos workspace_path o qdrant_collection

    Fuentes de indexación soportadas:
    - codebase-index CLI (colección Qdrant en .codebase/)
    - sqlite-vec (SQLite en .codebase/vectors.db)
    - roo-code (colección Qdrant calculada desde path)
    """
    
    def __init__(self, qdrant_store=None):
        """Inicializa el resolver.
        
        Args:
            qdrant_store: Instancia de QdrantStore para calcular nombres de colección
        """
        self.qdrant_store = qdrant_store
    
    def _check_qdrant_collection_file(self, workspace: Path) -> Optional[str]:
        """Busca archivo con nombre de colección Qdrant en .codebase/

        Busca archivos como:
        - .codebase/codebase-abc123 (sin extensión)
        - .codebase/collection_name.txt

        Returns:
            Nombre de colección si se encuentra, None si no
        """
        codebase_dir = workspace / ".codebase"
        if not codebase_dir.exists():
            return None

        # Buscar archivos que empiecen con "codebase-" o "ws-"
        for file in codebase_dir.iterdir():
            if file.is_file():
                name = file.name
                # Archivo sin extensión que empieza con codebase- o ws-
                if name.startswith(("codebase-", "ws-")) and "." not in name:
                    return name
                # Archivo .txt con nombre de colección
                if name.endswith(".txt"):
                    with open(file, "r") as f:
                        content = f.read().strip()
                        if content.startswith(("codebase-", "ws-")):
                            return content

        return None

    def resolve(
        self,
        workspace_path: Optional[str] = None,
        qdrant_collection: Optional[str] = None,
        storage_type: Literal["sqlite", "qdrant"] = "qdrant"
    ) -> StorageResolution:
        """Resuelve qué storage usar basado en parámetros y disponibilidad.

        Lógica de resolución:

        1. Si `qdrant_collection` está especificado explícitamente:
           → Usar Qdrant con esa colección (prioridad máxima)

        2. Si `workspace_path` está especificado:
           a. Buscar colección Qdrant en `.codebase/` (archivos codebase-* o ws-*)
           b. Si no existe, buscar SQLite en `.codebase/vectors.db`
           c. Si no existe, calcular colección Qdrant desde workspace_path (roo-code style)

        Args:
            workspace_path: Ruta del workspace (opcional)
            qdrant_collection: Nombre de colección Qdrant explícito (opcional)
            storage_type: Tipo de storage preferido: "sqlite" o "qdrant" (default: "qdrant")
                         NOTA: Este parámetro es ignorado si se encuentra storage en .codebase/

        Returns:
            StorageResolution con información sobre qué storage usar

        Raises:
            ValueError: Si no se puede resolver ningún storage válido
        """
        # Prioridad 1: Colección Qdrant explícita
        if qdrant_collection and qdrant_collection.strip():
            collection_name = qdrant_collection.strip()
            return StorageResolution(
                storage_type="qdrant",
                identifier=f"colección Qdrant '{collection_name}'",
                qdrant_collection=collection_name
            )

        # Validar que al menos workspace_path esté presente
        if not workspace_path or not workspace_path.strip():
            raise ValueError(
                "Se requiere 'workspace_path' o 'qdrant_collection'.\n\n"
                "- workspace_path: Ruta del workspace para buscar storage disponible\n"
                "- qdrant_collection: Nombre explícito de colección Qdrant (prioridad máxima)"
            )

        workspace = Path(workspace_path.strip())

        # Prioridad 2: Buscar colección Qdrant en .codebase/
        qdrant_collection_name = self._check_qdrant_collection_file(workspace)
        if qdrant_collection_name:
            return StorageResolution(
                storage_type="qdrant",
                identifier=f"workspace '{workspace_path}' (colección Qdrant: {qdrant_collection_name})",
                qdrant_collection=qdrant_collection_name
            )

        # Prioridad 3: Buscar SQLite en .codebase/vectors.db
        sqlite_path = workspace / ".codebase" / "vectors.db"
        if sqlite_path.exists():
            return StorageResolution(
                storage_type="sqlite",
                identifier=f"SQLite en '{workspace_path}'",
                sqlite_path=sqlite_path
            )

        # Prioridad 4: Calcular colección Qdrant desde workspace_path (roo-code style)
        if not self.qdrant_store:
            raise ValueError(
                f"No se encontró storage en {workspace}/.codebase/ y no se puede calcular "
                "colección Qdrant (qdrant_store no configurado)"
            )

        normalized_workspace = self.qdrant_store._normalize_workspace_path(str(workspace))
        collection_name = self.qdrant_store._get_collection_name(normalized_workspace)

        return StorageResolution(
            storage_type="qdrant",
            identifier=f"workspace '{workspace_path}' (colección Qdrant calculada: {collection_name})",
            qdrant_collection=collection_name,
            fallback_used=True,
            fallback_reason="No se encontró storage en .codebase/, usando colección Qdrant calculada (roo-code style)"
        )
    
    def log_resolution(self, resolution: StorageResolution, ctx=None):
        """Log información sobre la resolución de storage.
        
        Args:
            resolution: Resultado de resolve()
            ctx: FastMCP context para logging (opcional)
        """
        msg = f"[Storage Resolver] Usando {resolution.storage_type}: {resolution.identifier}"
        
        if resolution.fallback_used:
            msg += f" (fallback: {resolution.fallback_reason})"
        
        if ctx:
            import asyncio
            asyncio.create_task(ctx.info(msg))
        else:
            print(msg, file=sys.stderr)

