"""
Lifecycle Manager - Gère le cycle de vie des patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import time


class LifecycleEventType(Enum):
    EMERGE = "emerge"
    FORK = "fork"
    MERGE = "merge"
    FREEZE = "freeze"
    DISSOLVE = "dissolve"


@dataclass
class LifecycleEvent:
    nh_id: bytes
    event_type: LifecycleEventType
    timestamp: int
    parent_ids: List[bytes] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class LifecycleManager:
    """
    Gère les événements de cycle de vie des patterns.
    """
    
    def __init__(self):
        # Historique des événements
        self.events: List[LifecycleEvent] = []
        
        # Graphe de lignées
        self.lineage_graph: Dict[bytes, List[bytes]] = {}  # enfant -> parents
        self.descendants: Dict[bytes, List[bytes]] = {}    # parent -> enfants
        
        # État des patterns
        self.pattern_status: Dict[bytes, str] = {}
    
    def emit_emerge(self, nh_id: bytes, lineage: List[bytes]):
        """Émet un événement d'émergence."""
        event = LifecycleEvent(
            nh_id=nh_id,
            event_type=LifecycleEventType.EMERGE,
            timestamp=int(time.time() * 1000),
            parent_ids=lineage
        )
        self.events.append(event)
        
        # Mettre à jour le graphe
        self.lineage_graph[nh_id] = lineage
        for parent in lineage:
            if parent not in self.descendants:
                self.descendants[parent] = []
            self.descendants[parent].append(nh_id)
        
        self.pattern_status[nh_id] = 'active'
    
    def emit_fork(self, parent_id: bytes, children_ids: List[bytes]):
        """Émet un événement de fork."""
        for child_id in children_ids:
            event = LifecycleEvent(
                nh_id=child_id,
                event_type=LifecycleEventType.FORK,
                timestamp=int(time.time() * 1000),
                parent_ids=[parent_id]
            )
            self.events.append(event)
            
            # Mettre à jour le graphe
            if child_id not in self.lineage_graph:
                self.lineage_graph[child_id] = []
            self.lineage_graph[child_id].append(parent_id)
            
            if parent_id not in self.descendants:
                self.descendants[parent_id] = []
            self.descendants[parent_id].append(child_id)
    
    def emit_merge(self, parent_ids: List[bytes], child_id: bytes):
        """Émet un événement de merge."""
        event = LifecycleEvent(
            nh_id=child_id,
            event_type=LifecycleEventType.MERGE,
            timestamp=int(time.time() * 1000),
            parent_ids=parent_ids
        )
        self.events.append(event)
        
        # Mettre à jour le graphe
        self.lineage_graph[child_id] = parent_ids
        for parent in parent_ids:
            if parent not in self.descendants:
                self.descendants[parent] = []
            self.descendants[parent].append(child_id)
        
        self.pattern_status[child_id] = 'active'
    
    def emit_freeze(self, nh_id: bytes):
        """Émet un événement de gel."""
        event = LifecycleEvent(
            nh_id=nh_id,
            event_type=LifecycleEventType.FREEZE,
            timestamp=int(time.time() * 1000)
        )
        self.events.append(event)
        self.pattern_status[nh_id] = 'frozen'
    
    def emit_dissolve(self, nh_id: bytes):
        """Émet un événement de dissolution."""
        event = LifecycleEvent(
            nh_id=nh_id,
            event_type=LifecycleEventType.DISSOLVE,
            timestamp=int(time.time() * 1000)
        )
        self.events.append(event)
        self.pattern_status[nh_id] = 'dissolved'
    
    def get_ancestors(self, nh_id: bytes, depth: int = -1) -> List[bytes]:
        """Retourne les ancêtres d'un pattern."""
        if nh_id not in self.lineage_graph:
            return []
        
        ancestors = []
        current_level = self.lineage_graph[nh_id]
        current_depth = 0
        
        while current_level and (depth == -1 or current_depth < depth):
            ancestors.extend(current_level)
            next_level = []
            for ancestor in current_level:
                if ancestor in self.lineage_graph:
                    next_level.extend(self.lineage_graph[ancestor])
            current_level = next_level
            current_depth += 1
        
        return ancestors
    
    def get_descendants(self, nh_id: bytes, depth: int = -1) -> List[bytes]:
        """Retourne les descendants d'un pattern."""
        if nh_id not in self.descendants:
            return []
        
        result = []
        current_level = self.descendants[nh_id]
        current_depth = 0
        
        while current_level and (depth == -1 or current_depth < depth):
            result.extend(current_level)
            next_level = []
            for descendant in current_level:
                if descendant in self.descendants:
                    next_level.extend(self.descendants[descendant])
            current_level = next_level
            current_depth += 1
        
        return result
    
    def get_events(self, 
                   nh_id: Optional[bytes] = None,
                   event_type: Optional[LifecycleEventType] = None,
                   since: Optional[int] = None) -> List[LifecycleEvent]:
        """Filtre et retourne des événements."""
        result = self.events
        
        if nh_id is not None:
            result = [e for e in result if e.nh_id == nh_id]
        
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        
        if since is not None:
            result = [e for e in result if e.timestamp >= since]
        
        return result