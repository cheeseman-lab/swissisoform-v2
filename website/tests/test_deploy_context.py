"""Guard the two package sets the deploy depends on staying in agreement.

``prepare_deploy.sh`` verifies that every module it stages *imports*, but it runs
that check under the developer's interpreter — which has the whole pipeline
environment installed. So a staged module that imports something the image does not
install passes staging green and then kills the container: gunicorn spawns no
workers and the healthcheck fails with no hint why. That is the same failure mode
the staging import check was written for, one layer up.

Checking it here rather than in ``prepare_deploy.sh`` keeps the deploy script doing
one job, and puts the assertion where assertions belong. It reads the repo tree, so
it needs no staged build context.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

WEBSITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEBSITE_ROOT.parent
DOCKERFILE = WEBSITE_ROOT / "Dockerfile"
PREPARE_DEPLOY = WEBSITE_ROOT / "prepare_deploy.sh"

#: Packages resolved from the repo itself, not from pip.
LOCAL = {"swissisoform", "swissisoform_site", "scripts"}

#: Import name -> distribution name, where the two differ.
ALIASES = {"Bio": "biopython", "yaml": "pyyaml", "PIL": "pillow", "sklearn": "scikit-learn"}

#: Imported lazily inside functions the container never calls. Named individually
#: rather than skipping lazy imports wholesale, so a lazy import that *is* reachable
#: still trips this test.
KNOWN_UNREACHABLE = {
    # clinical/validate.py's genome readers. The scan hands the classifier the
    # coding sequence out of orf_index.parquet, so no FASTA is ever opened.
    "pysam": "clinical.validate genome readers; the CDS ships in orf_index.parquet",
}

#: Imported directly but never pinned, because a pinned package guarantees them.
#: Satisfied by proving the *parent* is installed, so removing the parent from the
#: Dockerfile still fails this test.
GUARANTEED_BY = {
    "numpy": "pandas",
    "markupsafe": "flask",  # via jinja2
    "werkzeug": "flask",  # Flask's own WSGI layer — it cannot be installed without it
}


def _installed_distributions() -> set[str]:
    """Package names on the Dockerfile's ``pip install`` line."""
    text = DOCKERFILE.read_text()
    return {name.lower() for name in re.findall(r'"([A-Za-z0-9_.-]+)[><=]', text)}


def _staged_module_paths() -> list[Path]:
    """The repo files ``prepare_deploy.sh`` copies into the build context.

    Derived from the script rather than restated here, so a new ``cp`` line is
    covered automatically. The ``variantquery`` glob is expanded the way the script
    expands it, minus the ``__main__.py`` it skips.
    """
    script = PREPARE_DEPLOY.read_text()
    paths = [
        REPO_ROOT / rel
        for rel in re.findall(r"^cp\s+\.\./(src/\S+\.py)\s", script, flags=re.MULTILINE)
    ]
    if "../src/swissisoform/variantquery/*.py" in script:
        paths += [
            path
            for path in sorted((REPO_ROOT / "src/swissisoform/variantquery").glob("*.py"))
            if path.name != "__main__.py"
        ]
    # Plus the site package itself, which the Dockerfile copies wholesale.
    paths += sorted((WEBSITE_ROOT / "src/swissisoform_site").rglob("*.py"))
    return paths


def _third_party_imports(path: Path) -> set[str]:
    """Top-level third-party package names imported anywhere in *path*."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found = [(node.module or "").split(".")[0]]
        else:
            continue
        names.update(
            name
            for name in found
            if name and name not in LOCAL and name not in sys.stdlib_module_names
        )
    return names


def test_prepare_deploy_stages_files_that_exist() -> None:
    """A ``cp`` line naming a file that has moved would fail mid-deploy."""
    paths = _staged_module_paths()
    assert paths, "found no staged modules — has prepare_deploy.sh changed shape?"
    missing = [str(path) for path in paths if not path.is_file()]
    assert not missing, f"prepare_deploy.sh stages files that do not exist: {missing}"


def test_every_staged_import_is_installed_in_the_image() -> None:
    """Nothing the image runs may import a package the image does not install."""
    installed = _installed_distributions()
    assert installed, "no packages parsed out of the Dockerfile's pip install line"

    missing: dict[str, set[str]] = {}
    for path in _staged_module_paths():
        for name in _third_party_imports(path):
            if name in KNOWN_UNREACHABLE:
                continue
            dist = GUARANTEED_BY.get(name.lower()) or ALIASES.get(name, name).lower()
            if dist not in installed:
                missing.setdefault(dist, set()).add(str(path.relative_to(REPO_ROOT)))

    assert not missing, (
        "staged code imports packages the image does not install — add them to the "
        f"pip install line in website/Dockerfile: "
        f"{ {dist: sorted(where) for dist, where in sorted(missing.items())} }"
    )


@pytest.mark.parametrize("dist", ["biopython", "pandas", "pyarrow", "flask", "gunicorn"])
def test_dockerfile_pins_the_packages_the_site_needs(dist: str) -> None:
    """Spot-check the ones whose absence is not obvious from the site's own imports.

    ``biopython`` in particular arrives only transitively, through the vendored
    ``clinical.validate`` — nothing under ``swissisoform_site`` mentions it.
    """
    assert dist in _installed_distributions()
