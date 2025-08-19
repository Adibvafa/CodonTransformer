"""
File: CodonComplexity.py
------------------------
Includes functions for checking DNA sequence complexity and enforcing synthesis rules.
"""

from typing import Dict, List, Optional, Tuple, Union
import re
from dataclasses import dataclass

from CodonTransformer.CodonUtils import DNASequencePrediction

def calculate_gc_content(sequence: str) -> float:
    """Calculate the GC content of a DNA sequence.

    Args:
        sequence (str): The DNA sequence

    Returns:
        float: GC content as a percentage
    """
    if not sequence:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('g') + sequence.count('C') + sequence.count('c')
    return (gc_count / len(sequence)) * 100

@dataclass
class ComplexityViolation:
    """Represents a violation of sequence complexity rules."""
    rule_name: str
    description: str
    severity: float  # Score contribution to total complexity
    suggestion: str
    location: Optional[Tuple[int, int]] = None  # Start and end positions if applicable

@dataclass
class ComplexityConfig:
    """Configuration for sequence complexity checking."""
    max_repeat_length: int = 20  # Maximum length of repeats
    min_repeat_tm: float = 60.0  # Minimum melting temperature for repeat check
    min_gc_content: float = 25.0  # Minimum global GC content
    max_gc_content: float = 65.0  # Maximum global GC content
    max_gc_deviation: float = 52.0  # Maximum deviation in GC content within gene
    gc_window_size: int = 50  # Window size for GC content calculation
    max_homopolymer_length: int = 5  # Maximum length of homopolymer runs
    his_tag_pattern: str = "CACCAC"  # Required pattern for HIS tags
    enabled_rules: List[str] = None  # List of rules to check, None means all

    def __post_init__(self):
        if self.enabled_rules is None:
            self.enabled_rules = [
                "repeat_sequences",
                "gc_content",
                "gc_deviation",
                "homopolymers",
                "his_tag"
            ]

def calculate_tm(sequence: str) -> float:
    """
    Calculate the melting temperature of a DNA sequence.
    Uses a basic nearest-neighbor method.

    Args:
        sequence (str): DNA sequence

    Returns:
        float: Estimated melting temperature in Celsius
    """
    if len(sequence) < 2:
        return 0.0

    # Basic nearest-neighbor parameters (simplified)
    nn_params = {
        'AA': -7.9, 'TT': -7.9,
        'AT': -7.2, 'TA': -7.2,
        'CA': -8.5, 'TG': -8.5,
        'GT': -8.4, 'AC': -8.4,
        'CT': -7.8, 'AG': -7.8,
        'GA': -8.2, 'TC': -8.2,
        'CG': -10.6, 'GC': -9.8,
        'GG': -8.0, 'CC': -8.0
    }

    # Calculate entropy contribution
    dH = sum(nn_params.get(sequence[i:i+2], 0) for i in range(len(sequence)-1))

    # Add initiation parameters
    dH += 0.1 * len(sequence)

    # Approximate Tm calculation
    return round((dH * 1000) / (len(sequence) * -10 + 108), 1)

def find_repeats(sequence: str, min_length: int = 20, min_tm: float = 60.0) -> List[Tuple[str, List[int]]]:
    """
    Find repeated sequences in DNA that meet length and Tm criteria.

    Args:
        sequence (str): DNA sequence
        min_length (int): Minimum length of repeats to find
        min_tm (float): Minimum melting temperature threshold

    Returns:
        List[Tuple[str, List[int]]]: List of (repeat sequence, positions) tuples
    """
    repeats = []
    sequence = sequence.upper()

    # Look for repeats of various lengths
    for length in range(min_length, min(50, len(sequence))):
        for i in range(len(sequence) - length + 1):
            pattern = sequence[i:i+length]
            if calculate_tm(pattern) >= min_tm:
                # Find all occurrences
                positions = [m.start() for m in re.finditer(f'(?={pattern})', sequence)]
                if len(positions) > 1:
                    repeats.append((pattern, positions))

    return repeats

def calculate_gc_windows(sequence: str, window_size: int = 50) -> List[float]:
    """
    Calculate GC content in sliding windows.

    Args:
        sequence (str): DNA sequence
        window_size (int): Size of sliding window

    Returns:
        List[float]: List of GC percentages for each window
    """
    gc_contents = []
    for i in range(0, len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        gc_contents.append(calculate_gc_content(window))
    return gc_contents

def find_homopolymers(sequence: str, max_length: int = 5) -> List[Tuple[str, int]]:
    """
    Find homopolymer runs longer than specified length.

    Args:
        sequence (str): DNA sequence
        max_length (int): Maximum allowed homopolymer length

    Returns:
        List[Tuple[str, int]]: List of (homopolymer sequence, position) tuples
    """
    homopolymers = []
    for match in re.finditer(r'([ATCG])\1{' + str(max_length) + ',}', sequence):
        homopolymers.append((match.group(), match.start()))
    return homopolymers

def check_sequence_complexity(
    sequence: str,
    config: Optional[ComplexityConfig] = None
) -> List[ComplexityViolation]:
    """
    Check DNA sequence against complexity rules.

    Args:
        sequence (str): DNA sequence to check
        config (Optional[ComplexityConfig]): Configuration for complexity checking

    Returns:
        List[ComplexityViolation]: List of complexity rule violations
    """
    if config is None:
        config = ComplexityConfig()

    violations = []
    sequence = sequence.upper()

    # Check repeat sequences
    if "repeat_sequences" in config.enabled_rules:
        repeats = find_repeats(sequence, config.max_repeat_length, config.min_repeat_tm)
        if repeats:
            repeat_percentage = sum(len(r[0]) * len(r[1]) for r in repeats) / len(sequence) * 100
            if repeat_percentage > 40:
                violations.append(ComplexityViolation(
                    rule_name="repeat_sequences",
                    description=f"Repeated sequences comprise {repeat_percentage:.1f}% of sequence",
                    severity=8.8 if repeat_percentage > 60 else 4.4,
                    suggestion="Redesign to reduce repeats to less than 40% of sequence"
                ))

    # Check global GC content
    if "gc_content" in config.enabled_rules:
        gc_content = calculate_gc_content(sequence)
        if not config.min_gc_content <= gc_content <= config.max_gc_content:
            violations.append(ComplexityViolation(
                rule_name="gc_content",
                description=f"Global GC content ({gc_content:.1f}%) outside allowed range",
                severity=4.2,
                suggestion=f"Adjust sequence to have GC content between {config.min_gc_content}% and {config.max_gc_content}%"
            ))

    # Check GC content deviation
    if "gc_deviation" in config.enabled_rules:
        gc_windows = calculate_gc_windows(sequence, config.gc_window_size)
        if gc_windows:
            max_gc = max(gc_windows)
            min_gc = min(gc_windows)
            gc_deviation = max_gc - min_gc
            if gc_deviation > config.max_gc_deviation:
                max_gc_pos = gc_windows.index(max_gc) * config.gc_window_size
                violations.append(ComplexityViolation(
                    rule_name="gc_deviation",
                    description=f"GC content deviation ({gc_deviation:.1f}%) exceeds maximum",
                    severity=4.0,
                    suggestion="Reduce GC content variation between regions",
                    location=(max_gc_pos, max_gc_pos + config.gc_window_size)
                ))

    # Check homopolymers
    if "homopolymers" in config.enabled_rules:
        homopolymers = find_homopolymers(sequence, config.max_homopolymer_length)
        if homopolymers:
            violations.append(ComplexityViolation(
                rule_name="homopolymers",
                description=f"Contains {len(homopolymers)} long homopolymer runs",
                severity=1.0,
                suggestion="Break up long runs of identical nucleotides"
            ))

    # Check HIS tag pattern if present
    if "his_tag" in config.enabled_rules and "CAC" in sequence:
        his_pattern = re.compile(r'(CAC){2,}')
        if not his_pattern.search(sequence):
            violations.append(ComplexityViolation(
                rule_name="his_tag",
                description="Incorrect HIS tag pattern",
                severity=0.5,
                suggestion="Use alternating CAC/CAT codons for HIS tags"
            ))

    return violations

def get_total_complexity_score(violations: List[ComplexityViolation]) -> float:
    """
    Calculate total complexity score from violations.

    Args:
        violations (List[ComplexityViolation]): List of complexity violations

    Returns:
        float: Total complexity score
    """
    return sum(v.severity for v in violations)

def predict_with_complexity_check(
    predict_func,
    protein: str,
    organism: Union[int, str],
    complexity_config: Optional[ComplexityConfig] = None,
    max_attempts: int = 10,
    **kwargs
) -> Tuple[DNASequencePrediction, Optional[str]]:
    """
    Wrapper for DNA sequence prediction that checks sequence complexity.

    Args:
        predict_func: Function that predicts DNA sequences
        protein (str): Input protein sequence
        organism (Union[int, str]): Organism ID or name
        complexity_config (Optional[ComplexityConfig]): Configuration for complexity checking
        max_attempts (int): Maximum number of prediction attempts
        **kwargs: Additional arguments for predict_func

    Returns:
        Tuple[DNASequencePrediction, Optional[str]]:
            - The predicted sequence (best one if multiple attempts)
            - Complexity report for the sequence
    """
    best_prediction = None
    best_score = float('inf')
    best_report = None

    # Try generating sequences until we find one that passes complexity checks
    # or reach max attempts
    for attempt in range(max_attempts):
        # Get a new prediction
        if kwargs.get('deterministic', True):
            kwargs['deterministic'] = False
            kwargs['temperature'] = 0.2 + (attempt * 0.1)  # Gradually increase temperature

        prediction = predict_func(protein=protein, organism=organism, **kwargs)

        # Check sequence complexity
        violations = check_sequence_complexity(
            prediction.predicted_dna,
            config=complexity_config
        )
        score = get_total_complexity_score(violations)

        # Generate report
        report = format_complexity_report(prediction.predicted_dna, violations)

        # Keep track of best sequence seen
        if score < best_score:
            best_prediction = prediction
            best_score = score
            best_report = report

        # If this sequence passes complexity checks, return it
        if score < 10:
            return prediction, report

    # If we couldn't find a sequence that passes checks, return best one seen
    return best_prediction, best_report

def format_complexity_report(
    sequence: str,
    violations: List[ComplexityViolation]
) -> str:
    """
    Format complexity check results into a readable report.

    Args:
        sequence (str): The analyzed DNA sequence
        violations (List[ComplexityViolation]): List of found violations

    Returns:
        str: Formatted report string
    """
    total_score = get_total_complexity_score(violations)

    report = []
    report.append("DNA Sequence Complexity Analysis")
    report.append("=" * 40)
    report.append(f"Sequence Length: {len(sequence)} bp")
    report.append(f"Total Complexity Score: {total_score:.1f}")

    if total_score >= 10:
        report.append("\nWARNING: Sequence may be too complex for synthesis")

    if violations:
        report.append("\nComplexity Issues Found:")
        for v in violations:
            report.append(f"\n{v.rule_name}:")
            report.append(f"  Description: {v.description}")
            report.append(f"  Severity Score: {v.severity}")
            report.append(f"  Suggestion: {v.suggestion}")
            if v.location:
                report.append(f"  Location: {v.location[0]}-{v.location[1]}")
    else:
        report.append("\nNo complexity issues found.")

    return "\n".join(report)
"""
# Usage example
if __name__ == "__main__":
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    import torch

    # Example protein sequence
    protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"
    organism = "Escherichia coli general"

    # Set up custom complexity rules
    config = ComplexityConfig(
        max_repeat_length=18,      # More stringent repeat check
        max_gc_content=60.0,       # Lower max GC content
        max_gc_deviation=45.0,     # More stringent GC deviation check
        max_homopolymer_length=4,  # Stricter homopolymer limit
        enabled_rules=[            # Only check these specific rules
            "repeat_sequences",
            "gc_content",
            "gc_deviation",
            "homopolymers"
        ]
    )

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize prediction with complexity checking
    prediction, complexity_report = predict_with_complexity_check(
        predict_func=predict_dna_sequence,
        protein=protein,
        organism=organism,
        complexity_config=config,
        device=device,
        deterministic=False,  # Use non-deterministic mode for multiple attempts
        temperature=0.2,      # Start with conservative sampling
        top_p=0.95,          # Use nucleus sampling
        max_attempts=5       # Try up to 5 times to get a good sequence
    )

    # Print results
    print("===== Sequence Generation Results =====")
    print("\nPredicted DNA sequence:")
    print(prediction.predicted_dna)
    print("\nComplexity Analysis:")
    print(complexity_report)
"""
