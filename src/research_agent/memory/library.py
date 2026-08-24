"""Append-only persistence for experimental trajectories."""

from pathlib import Path

from research_agent.interfaces import ExperienceRecord, MemoryCondition


class ExperienceLibrary:
    """Store and query validated experience records in JSONL format.

    Attributes:
        file_path: Path to the JSONL experience file.
    """

    def __init__(self, file_path: str = "data/experiences.jsonl") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ExperienceRecord) -> None:
        """Append one trajectory after verifying result provenance.

        Args:
            record: Validated experience record to append.

        Raises:
            ValueError: If record config_hash does not match result config_hash.
        """

        if record.config.config_hash != record.result.config_hash:
            raise ValueError(
                "Configuration and result have different hashes.",
            )

        with self.file_path.open("a", encoding="utf-8") as file:
            file.write(record.model_dump_json() + "\n")

    def load_all(self) -> list[ExperienceRecord]:
        """Load and validate every non-empty JSONL record.

        Returns:
            List of parsed ExperienceRecord instances.
        """

        if not self.file_path.exists():
            return []

        records: list[ExperienceRecord] = []
        with self.file_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(
                        ExperienceRecord.model_validate_json(line),
                    )
        return records

    def contains_config_hash(
        self,
        config_hash: str,
        condition: MemoryCondition | None = None,
        seed: int | None = None,
    ) -> bool:
        """Check for a duplicate within an optional condition/seed scope.

        Args:
            config_hash: SHA-256 fingerprint to check.
            condition: Optional condition filter.
            seed: Optional seed filter.

        Returns:
            True if a matching experience record exists, False otherwise.
        """

        for record in self.load_all():
            if record.config.config_hash != config_hash:
                continue
            if condition is not None and record.config.condition != condition:
                continue
            if seed is not None and record.config.seed != seed:
                continue
            return True
        return False
