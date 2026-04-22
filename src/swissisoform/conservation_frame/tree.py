"""Minimal Newick parser + MRCA-depth utilities.

We don't need a full phylogenetics library here — just enough to parse the
Newick string emitted by ``halStats --tree`` and compute, per species, the
number of edges between the reference (``hg38``) and the most recent common
ancestor of the reference and that species.

Branch lengths (``:0.025``), bootstrap labels, and quoted names are
tolerated but discarded. The output tree is a plain :class:`Node` graph
with parent links so path-to-root is O(depth).

Use :func:`depth_from_reference` to turn a Newick + reference name into
``{species: depth_int}``. Bigger numbers = older / more distant MRCA.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """A Newick tree node (internal or leaf).

    Attributes:
        name: Label of the node. Empty string for unnamed internals.
        children: Child nodes (empty for leaves).
        parent: Parent node, or ``None`` for the root.
    """

    name: str = ""
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None


# ---------------------------------------------------------------------------
# Newick parser
# ---------------------------------------------------------------------------


def parse_newick(text: str) -> Node:
    """Parse a Newick string into a :class:`Node` tree.

    Supports the subset emitted by ``halStats --tree`` and most phylogenetic
    software: nested parens, leaf names, internal node labels, branch
    lengths (``:0.5``), comments (``[...]``), and an optional terminating
    ``;``. Quoted names (``'A. thaliana'``) are unquoted. Unknown trailing
    characters after a successful parse are ignored.

    Args:
        text: Newick string (may contain whitespace and trailing ``;``).

    Returns:
        Root :class:`Node` of the parsed tree.

    Raises:
        ValueError: If the string is empty or starts with an unexpected
            token. Malformed subtrees are tolerated where possible.
    """
    # Strip comments and whitespace, drop the terminating ';' if present.
    stripped = _strip_comments(text).strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("Empty Newick string")

    parser = _Parser(stripped)
    root = parser.parse_node(parent=None)
    _link_parents(root)
    return root


def _strip_comments(text: str) -> str:
    """Drop ``[...]`` comment blocks from *text* (non-nested)."""
    out: list[str] = []
    in_comment = False
    for ch in text:
        if ch == "[":
            in_comment = True
            continue
        if ch == "]":
            in_comment = False
            continue
        if not in_comment:
            out.append(ch)
    return "".join(out)


def _link_parents(node: Node) -> None:
    """Attach ``parent`` pointers so path-to-root is O(depth)."""
    for child in node.children:
        child.parent = node
        _link_parents(child)


class _Parser:
    """Recursive-descent Newick parser — not exposed.

    Scope is deliberately tiny: enough to consume ``halStats --tree``
    output. Unusual dialects (NHX tags, translate tables) are not handled.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def parse_node(self, parent: Node | None) -> Node:
        """Parse a single subtree starting at ``self.pos``."""
        self._skip_ws()
        node = Node(parent=parent)
        if self._peek() == "(":
            self.pos += 1
            node.children.append(self.parse_node(parent=node))
            while self._peek() == ",":
                self.pos += 1
                node.children.append(self.parse_node(parent=node))
            if self._peek() == ")":
                self.pos += 1
        # Optional label (internal or leaf name).
        node.name = self._read_label()
        # Optional branch length ``:<number>``.
        if self._peek() == ":":
            self.pos += 1
            self._read_label()  # discard branch length
        return node

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _read_label(self) -> str:
        self._skip_ws()
        if self.pos >= len(self.text):
            return ""
        ch = self.text[self.pos]
        if ch in "(),:;":
            return ""
        if ch in "'\"":
            return self._read_quoted(ch)
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in "(),:;":
            self.pos += 1
        return self.text[start : self.pos].strip()

    def _read_quoted(self, quote: str) -> str:
        self.pos += 1  # opening quote
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            self.pos += 1
        label = self.text[start : self.pos]
        if self.pos < len(self.text):
            self.pos += 1  # closing quote
        return label


# ---------------------------------------------------------------------------
# Depth utilities
# ---------------------------------------------------------------------------


def find_leaf(root: Node, name: str) -> Node | None:
    """Return the first leaf with ``node.name == name``, or None."""
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if not node.children and node.name == name:
            return node
        stack.extend(node.children)
    return None


def _path_to_root(node: Node) -> list[Node]:
    """Return the list of ancestors from *node* up to the root (inclusive)."""
    path: list[Node] = []
    current: Node | None = node
    while current is not None:
        path.append(current)
        current = current.parent
    return path


def mrca_depth(root: Node, reference: str, target: str) -> int | None:
    """Edges between *reference* and the MRCA of (reference, target).

    ``depth(reference) == 0``. Species sharing ``reference``'s immediate
    parent return 1; species whose MRCA is one level further out return 2;
    and so on. Useful for ranking species by how deeply conserved a feature
    would need to be to appear in them.

    Args:
        root: Tree root.
        reference: Reference leaf name (e.g. ``"hg38"``).
        target: Other leaf name.

    Returns:
        Integer depth, or ``None`` if either name is missing.
    """
    ref_leaf = find_leaf(root, reference)
    tgt_leaf = find_leaf(root, target)
    if ref_leaf is None or tgt_leaf is None:
        return None
    if ref_leaf is tgt_leaf:
        return 0

    ref_ancestors = _path_to_root(ref_leaf)
    tgt_ancestor_set = {id(n) for n in _path_to_root(tgt_leaf)}
    for depth, ancestor in enumerate(ref_ancestors):
        if id(ancestor) in tgt_ancestor_set:
            return depth
    return None  # disconnected — shouldn't happen on a valid tree


def depth_from_reference(newick: str, reference: str) -> dict[str, int]:
    """Map every leaf name to its MRCA depth from *reference*.

    Species that cannot be connected to the reference (malformed or
    disconnected) are omitted rather than assigned ``None`` — callers can
    use a default when looking up.

    Args:
        newick: Newick tree string.
        reference: Reference leaf name.

    Returns:
        ``{species: depth_int}`` including ``reference`` itself (depth 0).
    """
    root = parse_newick(newick)
    ref_leaf = find_leaf(root, reference)
    if ref_leaf is None:
        return {}

    ref_ancestors = _path_to_root(ref_leaf)
    ancestor_depth = {id(node): depth for depth, node in enumerate(ref_ancestors)}

    out: dict[str, int] = {}
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if not node.children and node.name:
            depth = _depth_via_ancestors(node, ancestor_depth)
            if depth is not None:
                out[node.name] = depth
        stack.extend(node.children)
    return out


def _depth_via_ancestors(
    leaf: Node, ancestor_depth: dict[int, int]
) -> int | None:
    """Walk *leaf* toward root until the first ancestor in the reference path."""
    current: Node | None = leaf
    while current is not None:
        if id(current) in ancestor_depth:
            return ancestor_depth[id(current)]
        current = current.parent
    return None
