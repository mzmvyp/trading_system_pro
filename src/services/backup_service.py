"""
Backup Service - Automated backup with rotation and verification.
Source: trader_monitor
Features:
- Scheduled backups (configurable interval)
- ZIP compression with SHA256 verification
- Rotation policy (keep N most recent)
- Restore with integrity check
"""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class BackupService:
    """
    Automated backup service for trading data and configuration.
    Supports compression, verification, rotation, and restore.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.backup_dir = Path(config.get("backup_dir", "data/backups"))
        self.max_backups = config.get("max_backups", 10)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Directories/files to backup
        self.backup_targets = config.get("backup_targets", [
            "data/portfolio",
            "data/models",
            "data/trade_history.json",
            "data/signals",
            ".env",
        ])

    def create_backup(self, label: Optional[str] = None) -> Dict:
        """Create a new backup."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        label_str = f"_{label}" if label else ""
        backup_name = f"backup_{timestamp}{label_str}"
        zip_path = self.backup_dir / f"{backup_name}.zip"

        try:
            files_backed_up = 0
            total_size = 0

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for target in self.backup_targets:
                    target_path = Path(target)
                    if target_path.is_file():
                        zf.write(target_path, target_path)
                        files_backed_up += 1
                        total_size += target_path.stat().st_size
                    elif target_path.is_dir():
                        for file_path in target_path.rglob("*"):
                            if file_path.is_file():
                                zf.write(file_path, file_path)
                                files_backed_up += 1
                                total_size += file_path.stat().st_size

            # Calculate checksum
            checksum = self._calculate_checksum(zip_path)

            # Save metadata
            metadata = {
                "backup_name": backup_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files_count": files_backed_up,
                "original_size": total_size,
                "compressed_size": zip_path.stat().st_size,
                "checksum_sha256": checksum,
                "targets": self.backup_targets,
            }

            meta_path = self.backup_dir / f"{backup_name}.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Rotate old backups
            self._rotate_backups()

            logger.info(f"Backup created: {backup_name} ({files_backed_up} files, {total_size} bytes)")

            return {
                "success": True,
                "backup_name": backup_name,
                "path": str(zip_path),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Backup creation error: {e}")
            return {"success": False, "error": str(e)}

    def restore_backup(self, backup_name: str, target_dir: Optional[str] = None) -> Dict:
        """Restore from a backup with integrity verification."""
        zip_path = self.backup_dir / f"{backup_name}.zip"
        meta_path = self.backup_dir / f"{backup_name}.json"

        if not zip_path.exists():
            return {"success": False, "error": f"Backup not found: {backup_name}"}

        try:
            # Verify checksum
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                expected_checksum = metadata.get("checksum_sha256")
                actual_checksum = self._calculate_checksum(zip_path)
                if expected_checksum and actual_checksum != expected_checksum:
                    return {"success": False, "error": "Checksum mismatch - backup corrupted"}

            # Restore
            restore_dir = Path(target_dir) if target_dir else Path(".")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(restore_dir)

            logger.info(f"Backup restored: {backup_name}")
            return {"success": True, "backup_name": backup_name, "restored_to": str(restore_dir)}

        except Exception as e:
            logger.error(f"Restore error: {e}")
            return {"success": False, "error": str(e)}

    def list_backups(self) -> List[Dict]:
        """List all available backups."""
        backups = []
        for meta_file in sorted(self.backup_dir.glob("*.json"), reverse=True):
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)
                zip_path = self.backup_dir / f"{metadata['backup_name']}.zip"
                metadata["exists"] = zip_path.exists()
                backups.append(metadata)
            except Exception:
                continue
        return backups

    def _rotate_backups(self):
        """Remove old backups beyond max_backups limit."""
        backups = sorted(self.backup_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)

        for old_backup in backups[self.max_backups:]:
            try:
                old_backup.unlink()
                meta = old_backup.with_suffix(".json")
                if meta.exists():
                    meta.unlink()
                logger.info(f"Rotated old backup: {old_backup.name}")
            except Exception as e:
                logger.warning(f"Failed to rotate backup {old_backup.name}: {e}")

    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
