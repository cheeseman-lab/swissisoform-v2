"""Species lists for Path 1 (primate) and Path 2 (mammalian) conservation.

Names follow UCSC/Zoonomia HAL genome identifiers (e.g. ``hg38``,
``panTro6``). These lists are provisional; the exact genome names in a
given Zoonomia release should be confirmed with ``halStats --genomes
<file>.hal`` and refined as needed. A mismatch between a name here and a
genome present in the HAL results in that species being reported as
"absent" — there's no crash.
"""

from __future__ import annotations

# Reference genome. hal2maf uses this to orient every alignment block.
REFERENCE_SPECIES: str = "hg38"

# Path 1: primate radiation within the Zoonomia alignment. Extend with
# ``halStats --genomes`` output if a release adds new primate assemblies
# (e.g. Expanded Primates 2023).
PRIMATE_SPECIES: list[str] = [
    "panTro6",      # Chimpanzee
    "panPan3",      # Bonobo
    "gorGor6",      # Gorilla
    "ponAbe3",      # Orangutan
    "nomLeu3",      # Gibbon
    "rheMac10",     # Rhesus macaque
    "macFas5",      # Crab-eating macaque
    "papAnu4",      # Baboon
    "chlSab2",      # Vervet / African green monkey
    "nasLar1",      # Proboscis monkey
    "rhiRox1",      # Golden snub-nosed monkey
    "colAng1",      # Black-and-white colobus
    "calJac4",      # Common marmoset
    "saiBol1",      # Bolivian squirrel monkey
    "cebCap1",      # White-faced capuchin
    "aotNan1",      # Nancy Ma's night monkey
    "tarSyr2",      # Philippine tarsier
    "micMur3",      # Gray mouse lemur
    "proCoq1",      # Coquerel's sifaka
    "eulFla1",      # Blue-eyed black lemur
    "lemCat1",      # Ring-tailed lemur
    "otoGar3",      # Northern greater galago
]


# Path 2: 23-species mammalian radiation spanning all major clades.
# Covers Euarchontoglires, Laurasiatheria, Afrotheria, Xenarthra, and
# Marsupialia / Monotremata outgroups — enough breadth to call deep
# conservation signals above primate-specific innovation.
MAMMALIAN_SPECIES: list[str] = [
    "panTro6",      # Chimpanzee          (Hominidae)
    "rheMac10",     # Rhesus macaque      (Cercopithecidae)
    "calJac4",      # Common marmoset     (Platyrrhini)
    "micMur3",      # Mouse lemur         (Strepsirrhini)
    "mm10",         # House mouse         (Rodentia)
    "rn7",          # Brown rat           (Rodentia)
    "oryCun2",      # Rabbit              (Lagomorpha)
    "canFam6",      # Dog                 (Carnivora)
    "felCat9",      # Cat                 (Carnivora)
    "equCab3",      # Horse               (Perissodactyla)
    "bosTau9",      # Cow                 (Artiodactyla)
    "susScr11",     # Pig                 (Artiodactyla)
    "turTru2",      # Bottlenose dolphin  (Cetacea)
    "myoLuc2",      # Little brown bat    (Chiroptera)
    "rhiFer1",      # Greater horseshoe bat (Chiroptera)
    "eriEur2",      # Hedgehog            (Eulipotyphla)
    "loxAfr3",      # African elephant    (Afrotheria)
    "echTel2",      # Lesser hedgehog tenrec (Afrotheria)
    "dasNov3",      # Nine-banded armadillo (Xenarthra)
    "choHof1",      # Two-toed sloth      (Xenarthra)
    "monDom5",      # Gray short-tailed opossum (Marsupialia)
    "phaCin1",      # Koala               (Marsupialia)
    "ornAna2",      # Platypus            (Monotremata)
]
