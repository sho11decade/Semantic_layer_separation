"""Caching mechanism for LLM and detection results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class CacheManager:
    """Manages caching of pipeline results."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache manager.
        
        Args:
            cache_dir: Custom cache directory. Defaults to .semantic_layer_cache
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser()
        else:
            self.cache_dir = Path.home() / ".semantic_layer_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _make_key(image_path: str | Path, prompt: str) -> str:
        """Generate cache key from image path and prompt.
        
        Args:
            image_path: Path to image file
            prompt: Planning prompt used
            
        Returns:
            Cache key (hash string)
        """
        content = f"{Path(image_path).resolve()}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_planning_result(self, image_path: str | Path, prompt: str) -> Optional[dict]:
        """Get cached LLM planning result.
        
        Args:
            image_path: Path to image file
            prompt: Planning prompt
            
        Returns:
            Cached result dict if exists, None otherwise
        """
        key = self._make_key(image_path, prompt)
        cache_file = self.cache_dir / f"plan_{key}.json"
        
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                return None
        return None
    
    def set_planning_result(self, image_path: str | Path, prompt: str, result: dict) -> None:
        """Cache LLM planning result.
        
        Args:
            image_path: Path to image file
            prompt: Planning prompt
            result: Result to cache
        """
        key = self._make_key(image_path, prompt)
        cache_file = self.cache_dir / f"plan_{key}.json"
        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    
    def get_detection_result(self, image_path: str | Path, targets: list[str]) -> Optional[list[dict]]:
        """Get cached detection result.
        
        Args:
            image_path: Path to image file
            targets: Target labels for detection
            
        Returns:
            Cached result list if exists, None otherwise
        """
        key = self._make_key(image_path, "|".join(targets))
        cache_file = self.cache_dir / f"detect_{key}.json"
        
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                return None
        return None
    
    def set_detection_result(self, image_path: str | Path, targets: list[str], result: list[dict]) -> None:
        """Cache detection result.
        
        Args:
            image_path: Path to image file
            targets: Target labels for detection
            result: Result to cache
        """
        key = self._make_key(image_path, "|".join(targets))
        cache_file = self.cache_dir / f"detect_{key}.json"
        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    
    def clear(self) -> int:
        """Clear all cache files.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes.
        
        Returns:
            Total cache size
        """
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
