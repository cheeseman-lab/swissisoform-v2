"""Species lists for Path 1 (primate) and Path 2 (mammalian) conservation.

Names follow Zoonomia 241-mammal HAL genome identifiers, which are Latin
binomials with an underscore (e.g. ``Homo_sapiens``, ``Pan_troglodytes``).
Confirm names against ``halStats --genomes <file>.hal`` — a mismatch here
means that species is silently reported as "absent" (no crash).

The 241-mammal Cactus alignment is placental-only (no marsupials /
monotremes), so Path 2 is a placental-radiation list rather than the
class-spanning set the earlier UCSC-assembly version implied.
"""

from __future__ import annotations

# Reference genome. hal2maf uses this to orient every alignment block;
# must match exactly one genome in the HAL.
REFERENCE_SPECIES: str = "Homo_sapiens"

# Path 1: primate radiation within the Zoonomia 241-mammal alignment.
# 23 species spanning great apes, gibbons, Old-World monkeys, New-World
# monkeys, and strepsirrhines.
PRIMATE_SPECIES: list[str] = [
    # Hominidae
    "Pan_troglodytes",
    "Pan_paniscus",
    "Gorilla_gorilla",
    "Pongo_abelii",
    # Hylobatidae
    "Nomascus_leucogenys",
    # Cercopithecidae — Papionini / Cercopithecini
    "Macaca_mulatta",
    "Macaca_fascicularis",
    "Macaca_nemestrina",
    "Papio_anubis",
    "Chlorocebus_sabaeus",
    "Cercopithecus_neglectus",
    # Cercopithecidae — Colobinae
    "Nasalis_larvatus",
    "Rhinopithecus_roxellana",
    "Rhinopithecus_bieti",
    "Colobus_angolensis",
    "Piliocolobus_tephrosceles",
    # Platyrrhini (New-World monkeys)
    "Callithrix_jacchus",
    "Saimiri_boliviensis",
    "Cebus_albifrons",
    "Aotus_nancymaae",
    # Strepsirrhini
    "Microcebus_murinus",
    "Propithecus_coquereli",
    "Lemur_catta",
    "Otolemur_garnettii",
    "Nycticebus_coucang",
]


# Path 2: placental-mammal radiation spanning Euarchontoglires,
# Laurasiatheria, Afrotheria, and Xenarthra. 20 species — enough breadth
# to distinguish deep mammalian conservation from primate-specific signal.
# The Zoonomia 241-mammal HAL is placental-only, so no opossum/platypus.
MAMMALIAN_SPECIES: list[str] = [
    # Primates (anchor points)
    "Pan_troglodytes",       # Hominidae
    "Macaca_mulatta",        # Cercopithecidae
    "Callithrix_jacchus",    # Platyrrhini
    "Microcebus_murinus",    # Strepsirrhini
    # Rodentia
    "Mus_musculus",
    "Rattus_norvegicus",
    # Lagomorpha
    "Oryctolagus_cuniculus",
    # Carnivora
    "Canis_lupus_familiaris",
    "Felis_catus",
    # Perissodactyla
    "Equus_caballus",
    # Artiodactyla + Cetacea
    "Bos_taurus",
    "Sus_scrofa",
    "Tursiops_truncatus",
    # Chiroptera
    "Myotis_lucifugus",
    "Pteropus_vampyrus",
    # Eulipotyphla
    "Erinaceus_europaeus",
    # Afrotheria
    "Loxodonta_africana",
    "Echinops_telfairi",
    # Xenarthra
    "Dasypus_novemcinctus",
    "Choloepus_didactylus",
]
