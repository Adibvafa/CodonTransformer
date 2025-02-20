"""
File: CodonRestrictionSites.py
------------------------------
Includes functions for handling forbidden sequences and restriction sites in DNA sequences.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import re
from Bio.Seq import Seq


@dataclass
class ForbiddenSequenceViolation:
    """Represents a violation of forbidden sequence rules."""
    sequence: str
    position: int
    type: str  # 'restriction_site' or 'user_defined'
    is_reverse_complement: bool = False


@dataclass
class ForbiddenSequenceConfig:
    """Configuration for forbidden sequence handling."""
    # Pre-defined restriction sites to check
    restriction_sites: Set[str] = field(default_factory=lambda: {
        'BsaI': 'GGTCTC',  # BsaI recognition site
        'BbsI': 'GAAGAC',  # BbsI recognition site
        'BsmBI': 'CGTCTC',  # BsmBI recognition site
        'EcoRI': 'GAATTC',  # EcoRI recognition site
        'BamHI': 'GGATCC',  # BamHI recognition site
        'XhoI': 'CTCGAG',  # XhoI recognition site
        'NotI': 'GCGGCCGC',  # NotI recognition site
        'XbaI': 'TCTAGA',  # XbaI recognition site
    })

    # Additional user-defined forbidden sequences
    additional_sequences: Set[str] = field(default_factory=set)

    # Strategy for handling forbidden sequences
    strategy: str = "hybrid"  # 'mutate', 'regenerate', or 'hybrid'

    # Maximum number of regeneration attempts
    max_attempts: int = 10

    # Maximum number of mutations allowed per sequence
    max_mutations: int = 3

    # Whether to check reverse complement sequences
    check_reverse_complement: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_strategies = {'mutate', 'regenerate', 'hybrid'}
        if self.strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")

        if self.max_mutations < 0:
            raise ValueError("max_mutations must be non-negative")


def get_reverse_complement(sequence: str) -> str:
    """
    Get the reverse complement of a DNA sequence.

    Args:
        sequence (str): DNA sequence

    Returns:
        str: Reverse complement sequence
    """
    return str(Seq(sequence).reverse_complement())


def find_forbidden_sequences(
        dna: str,
        config: ForbiddenSequenceConfig
) -> List[ForbiddenSequenceViolation]:
    """
    Find all forbidden sequences in a DNA sequence.

    Args:
        dna (str): DNA sequence to check
        config (ForbiddenSequenceConfig): Configuration for checking

    Returns:
        List[ForbiddenSequenceViolation]: List of violations found
    """
    violations = []
    dna = dna.upper()

    # Combine restriction sites and additional sequences
    all_sequences = {
        seq: 'restriction_site' if name in config.restriction_sites else 'user_defined'
        for name, seq in config.restriction_sites.items()
    }
    all_sequences.update({seq: 'user_defined' for seq in config.additional_sequences})

    # Check each sequence and its reverse complement
    for seq, seq_type in all_sequences.items():
        # Check forward sequence
        for match in re.finditer(seq, dna):
            violations.append(ForbiddenSequenceViolation(
                sequence=seq,
                position=match.start(),
                type=seq_type,
                is_reverse_complement=False
            ))

        # Check reverse complement if enabled
        if config.check_reverse_complement:
            rev_comp = get_reverse_complement(seq)
            for match in re.finditer(rev_comp, dna):
                violations.append(ForbiddenSequenceViolation(
                    sequence=rev_comp,
                    position=match.start(),
                    type=seq_type,
                    is_reverse_complement=True
                ))

    return violations


def suggest_mutations(
        dna: str,
        violation: ForbiddenSequenceViolation,
        codon_boundaries: bool = True
) -> List[Tuple[str, int]]:
    """
    Suggest possible mutations to fix a forbidden sequence violation.

    Args:
        dna (str): Original DNA sequence
        violation (ForbiddenSequenceViolation): Violation to fix
        codon_boundaries (bool): Whether to respect codon boundaries

    Returns:
        List[Tuple[str, int]]: List of (mutated sequence, position) pairs
    """
    mutations = []
    seq_len = len(violation.sequence)
    pos = violation.position

    # Define possible nucleotide substitutions
    substitutions = {
        'A': ['T', 'C', 'G'],
        'T': ['A', 'C', 'G'],
        'C': ['A', 'T', 'G'],
        'G': ['A', 'T', 'C']
    }

    # Try each position in the forbidden sequence
    for i in range(seq_len):
        current_pos = pos + i

        # Skip if not on codon boundary when required
        if codon_boundaries and current_pos % 3 != 0:
            continue

        original_base = dna[current_pos]

        # Try each possible substitution
        for new_base in substitutions[original_base]:
            mutated_seq = (
                    dna[:current_pos] +
                    new_base +
                    dna[current_pos + 1:]
            )

            # Verify the mutation removes the forbidden sequence
            if not any(
                    v.sequence == violation.sequence
                    for v in find_forbidden_sequences(
                        mutated_seq[pos:pos + seq_len],
                        ForbiddenSequenceConfig(
                            additional_sequences={violation.sequence}
                        )
                    )
            ):
                mutations.append((mutated_seq, current_pos))

    return mutations


def fix_forbidden_sequences(
        dna: str,
        config: ForbiddenSequenceConfig,
        protein: Optional[str] = None
) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Attempt to fix forbidden sequences in a DNA sequence.

    Args:
        dna (str): DNA sequence to fix
        config (ForbiddenSequenceConfig): Configuration for fixing
        protein (Optional[str]): Original protein sequence to maintain

    Returns:
        Tuple[str, List[Tuple[int, str, str]]]:
            - Fixed DNA sequence
            - List of (position, original, new) changes made
    """
    violations = find_forbidden_sequences(dna, config)
    if not violations:
        return dna, []

    changes_made = []
    fixed_dna = dna
    mutations_remaining = config.max_mutations

    for violation in violations:
        if mutations_remaining <= 0:
            break

        # Get possible mutations
        mutations = suggest_mutations(
            fixed_dna,
            violation,
            codon_boundaries=protein is not None
        )

        # Apply first valid mutation
        for mutated_seq, position in mutations:
            # Verify protein sequence is maintained if provided
            if protein and str(Seq(mutated_seq).translate()) != protein:
                continue

            # Record the change
            changes_made.append((
                position,
                fixed_dna[position],
                mutated_seq[position]
            ))

            fixed_dna = mutated_seq
            mutations_remaining -= 1
            break

    return fixed_dna, changes_made


def format_forbidden_sequence_report(
        violations: List[ForbiddenSequenceViolation],
        changes: List[Tuple[int, str, str]] = None
) -> str:
    """
    Format a report of forbidden sequence violations and fixes.

    Args:
        violations (List[ForbiddenSequenceViolation]): Found violations
        changes (List[Tuple[int, str, str]], optional): Changes made to fix violations

    Returns:
        str: Formatted report
    """
    report = []
    report.append("Forbidden Sequence Analysis")
    report.append("=" * 40)

    if not violations:
        report.append("No forbidden sequences found.")
        return "\n".join(report)

    report.append(f"Found {len(violations)} forbidden sequence(s):")
    for v in violations:
        report.append(f"\n- Sequence: {v.sequence}")
        report.append(f"  Position: {v.position}")
        report.append(f"  Type: {v.type}")
        if v.is_reverse_complement:
            report.append("  (Reverse Complement)")

    if changes:
        report.append("\nChanges made:")
        for pos, old, new in changes:
            report.append(f"Position {pos}: {old} → {new}")

    return "\n".join(report)
