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
    
    Lógica de resolución:
    
    1. Si `qdrant_collection` está especificado explícitamente:
       → Usar Qdrant con esa colección (prioridad máxima)
    
    2. Si `storage_type="sqlite"` y `workspace_path` está especificado:
       → Buscar SQLite en `{workspace_path}/.codebase/vectors.db`
       → Si existe: usar SQLite
       → Si NO existe: fallback a Qdrant (si se puede calcular colección)
    
    3. Si `storage_type="qdrant"` (default) y `workspace_path` está especificado:
       → Calcular colección Qdrant desde workspace_path
       → Usar Qdrant
    
    4. Si nada está especificado:
       → Error: se requiere al menos workspace_path o qdrant_collection
    """
    
    def __init__(self, qdrant_store=None):
        """Inicializa el resolver.
        
        Args:
            qdrant_store: Instancia de QdrantStore para calcular nombres de colección
        """
        self.qdrant_store = qdrant_store
    
    def resolve(
        self,
        workspace_path: Optional[str] = None,
        qdrant_collection: Optional[str] = None,
        storage_type: Literal["sqlite", "qdrant"] = "qdrant"
    ) -> StorageResolution:
        """Resuelve qué storage usar basado en parámetros.
        
        Args:
            workspace_path: Ruta del workspace (opcional)
            qdrant_collection: Nombre de colección Qdrant explícito (opcional)
            storage_type: Tipo de storage preferido: "sqlite" o "qdrant" (default: "qdrant")
        
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
                "- workspace_path: Ruta del workspace para buscar SQLite o calcular colección Qdrant\n"
                "- qdrant_collection: Nombre explícito de colección Qdrant (prioridad máxima)\n"
                "- storage_type: 'sqlite' (preferir SQLite) o 'qdrant' (default, preferir Qdrant)"
            )
        
        workspace = Path(workspace_path.strip())
        
        # Prioridad 2: SQLite si se solicita explícitamente
        if storage_type == "sqlite":
            sqlite_path = workspace / ".codebase" / "vectors.db"
            
            if sqlite_path.exists():
                # SQLite existe, usarlo
                return StorageResolution(
                    storage_type="sqlite",
                    identifier=f"SQLite en '{workspace_path}'",
                    sqlite_path=sqlite_path
                )
            else:
                # SQLite no existe, intentar fallback a Qdrant
                if self.qdrant_store:
                    normalized_workspace = self.qdrant_store._normalize_workspace_path(str(workspace))
                    collection_name = self.qdrant_store._get_collection_name(normalized_workspace)
                    
                    return StorageResolution(
                        storage_type="qdrant",
                        identifier=f"workspace '{workspace_path}' (colección Qdrant: {collection_name})",
                        qdrant_collection=collection_name,
                        fallback_used=True,
                        fallback_reason=f"SQLite no encontrado en {sqlite_path}, usando Qdrant como fallback"
                    )
                else:
                    raise ValueError(
                        f"SQLite no encontrado en {sqlite_path} y no se puede usar Qdrant como fallback "
                        "(qdrant_store no configurado)"
                    )
        
        # Prioridad 3: Qdrant (default)
        if storage_type == "qdrant":
            if not self.qdrant_store:
                raise ValueError(
                    "storage_type='qdrant' requiere que qdrant_store esté configurado"
                )
            
            normalized_workspace = self.qdrant_store._normalize_workspace_path(str(workspace))
            collection_name = self.qdrant_store._get_collection_name(normalized_workspace)
            
            return StorageResolution(
                storage_type="qdrant",
                identifier=f"workspace '{workspace_path}' (colección Qdrant: {collection_name})",
                qdrant_collection=collection_name
            )
        
        # No debería llegar aquí
        raise ValueError(f"storage_type inválido: {storage_type}")
    
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

