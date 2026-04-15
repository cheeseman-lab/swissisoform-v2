"""Module protocols and validation for SwissIsoform v2 pipeline.

Two module types:
- ProteinModule: takes a protein sequence, returns annotation dict.
  Used for biophysics, motifs, localization, clinical, conservation.
  The wiring layer runs these on both canonical and isoform proteins.

- SiteModule: takes a TIS site, returns annotation dict.
  Used for core_identity, initiation_context — modules that need
  TIS-level metadata (orf_type, kozak_context, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite


@runtime_checkable
class ProteinModule(Protocol):
    """Protocol for modules that annotate a protein sequence.

    These modules are sequence-in, annotations-out. The wiring layer
    decides whether to pass canonical or isoform protein and where
    to store the result.

    Attributes:
        MODULE_NAME: Unique identifier (used as annotations key).
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: ``"C"`` for per-candidate or ``"G"`` for per-gene.
    """

    MODULE_NAME: str
    OUTPUT_COLUMNS: list[str]
    SCOPE: str

    def __init__(self, config: PipelineConfig) -> None: ...

    def annotate(self, protein: str) -> dict[str, Any]:
        """Annotate a single protein sequence.

        Args:
            protein: Amino acid sequence (may include trailing '*').

        Returns:
            Annotation dict with unprefixed keys. Values are either
            scalars (float, int, str, bool, None) or positional hit
            lists (list of dicts with 'pos' and optionally 'end' keys).
        """
        ...


@runtime_checkable
class SiteModule(Protocol):
    """Protocol for modules that annotate a TIS site using its metadata.

    These modules need TIS-level fields (orf_type, kozak_context, etc.)
    beyond just the protein sequence.

    Attributes:
        MODULE_NAME: Unique identifier (used as annotations key).
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: ``"C"`` for per-candidate.
    """

    MODULE_NAME: str
    OUTPUT_COLUMNS: list[str]
    SCOPE: str

    def __init__(self, config: PipelineConfig) -> None: ...

    def annotate_site(self, site: TranslationInitiationSite) -> dict[str, Any]:
        """Annotate a single TIS site.

        Args:
            site: A TranslationInitiationSite with identity and metadata populated.

        Returns:
            Annotation dict with unprefixed keys.
        """
        ...


def validate_protein_annotations(
    annotations: dict[str, Any],
    module_name: str,
    output_columns: list[str],
) -> None:
    """Validate that a ProteinModule's output has all expected keys.

    Args:
        annotations: The dict returned by annotate().
        module_name: The MODULE_NAME of the module.
        output_columns: The OUTPUT_COLUMNS (prefixed with module_name_).

    Raises:
        ValueError: If any expected key is missing.
    """
    prefix = f"{module_name}_"
    expected_keys = [
        col[len(prefix):] if col.startswith(prefix) else col for col in output_columns
    ]
    for key in expected_keys:
        if key not in annotations:
            raise ValueError(
                f"Module '{module_name}' missing key '{key}' in annotations"
            )


def validate_module_output(
    input_sites: list[TranslationInitiationSite],
    output_sites: list[TranslationInitiationSite],
    module_name: str,
    output_columns: list[str],
) -> None:
    """Validate that a module's output conforms to the pipeline contract.

    Checks:
    1. No sites were dropped (len(output) == len(input)).
    2. Every output site has isoform_annotations[module_name].
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

    prefix = f"{module_name}_"
    expected_keys = [
        col[len(prefix):] if col.startswith(prefix) else col for col in output_columns
    ]

    for i, site in enumerate(output_sites):
        if module_name not in site.isoform_annotations:
            raise ValueError(
                f"Module '{module_name}' site {i} ({site.tis_id}) "
                f"missing annotations['{module_name}']"
            )

        annotation = site.isoform_annotations[module_name]
        for key in expected_keys:
            if key not in annotation:
                raise ValueError(
                    f"Module '{module_name}' site {i} ({site.tis_id}) "
                    f"missing column '{key}' in annotations"
                )
