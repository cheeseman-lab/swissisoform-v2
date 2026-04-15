"""Module protocol and validation for SwissIsoform v2 pipeline.

Every annotation module must conform to ModuleProtocol and pass
validate_module_output after execution.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite


@runtime_checkable
class ModuleProtocol(Protocol):
    """Protocol that all annotation modules must implement.

    Attributes:
        MODULE_NAME: Unique identifier for this module (used as annotations key).
        OUTPUT_COLUMNS: Column names this module produces (prefixed with MODULE_NAME_).
        SCOPE: Either 'site' (per-TIS) or 'gene' (per-gene).
    """

    MODULE_NAME: str
    OUTPUT_COLUMNS: list[str]
    SCOPE: str

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the module with pipeline configuration."""
        ...

    def run(
        self, tis_sites: list[TranslationInitiationSite]
    ) -> list[TranslationInitiationSite]:
        """Run the module on a list of TIS sites.

        Args:
            tis_sites: Input sites to annotate.

        Returns:
            The same sites with annotations[MODULE_NAME] populated.
        """
        ...


def validate_module_output(
    input_sites: list[TranslationInitiationSite],
    output_sites: list[TranslationInitiationSite],
    module_name: str,
    output_columns: list[str],
) -> None:
    """Validate that a module's output conforms to the pipeline contract.

    Checks:
    1. No sites were dropped (len(output) == len(input)).
    2. Every output site has annotations[module_name].
    3. Every expected column key is present in the annotation dict.

    Args:
        input_sites: The sites passed to the module.
        output_sites: The sites returned by the module.
        module_name: The MODULE_NAME of the module being validated.
        output_columns: The OUTPUT_COLUMNS of the module (prefixed with module_name_).

    Raises:
        ValueError: If any validation check fails.
    """
    if len(output_sites) != len(input_sites):
        raise ValueError(
            f"Module '{module_name}' dropped sites: "
            f"input={len(input_sites)}, output={len(output_sites)}"
        )

    # Strip module_name_ prefix from output_columns to get annotation dict keys
    prefix = f"{module_name}_"
    expected_keys = [
        col[len(prefix):] if col.startswith(prefix) else col for col in output_columns
    ]

    for i, site in enumerate(output_sites):
        if module_name not in site.annotations:
            raise ValueError(
                f"Module '{module_name}' site {i} ({site.tis_id}) "
                f"missing annotations['{module_name}']"
            )

        annotation = site.annotations[module_name]
        for key in expected_keys:
            if key not in annotation:
                raise ValueError(
                    f"Module '{module_name}' site {i} ({site.tis_id}) "
                    f"missing column '{key}' in annotations"
                )
