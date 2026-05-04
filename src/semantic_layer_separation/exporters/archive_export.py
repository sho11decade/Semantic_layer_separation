"""Archive and organize output files."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def create_archive(output_dir: str | Path, archive_name: Optional[str] = None) -> Path:
    """Create a compressed archive of output directory.
    
    Args:
        output_dir: Directory to archive
        archive_name: Optional custom archive name
        
    Returns:
        Path to created archive
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    if archive_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"semantic_layer_separation_{timestamp}"
    
    archive_base = output_dir.parent / archive_name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
    return Path(archive_path)


def organize_output(output_dir: str | Path) -> dict:
    """Organize output files into a structured summary.
    
    Args:
        output_dir: Directory containing output files
        
    Returns:
        Dictionary with organization summary
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    files = list(output_dir.glob("*.png"))
    metadata_file = output_dir / "layers.json"
    
    summary = {
        "output_dir": str(output_dir),
        "png_files": len(files),
        "metadata_exists": metadata_file.exists(),
        "files": [f.name for f in files],
    }
    
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
