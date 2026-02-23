import re
from typing import List, Set, Dict
import string

COMPLIANCE_TRIGGER_KEYWORDS = [
    "what happens",
    "what will happen",
    "if we do not",
    "if we don't",
    "fail to",
    "failure to",
    "not maintain",
    "non compliance",
    "non-compliance",
    "violate",
    "penalty",
    "penalties",
    "consequence",
    "consequences",
    "sanction",
    "sanctions",
]

# Acronym mappings for query expansion
ACRONYM_MAPPINGS: Dict[str, str] = {
    # "pso": "Payment System Operator",
    # "psd": "Payment Systems Department",
    # "npsb": "National Payment Switch Bangladesh",
    # "bb": "Bangladesh Bank",
    # "aml": "Anti-Money Laundering",
    # "kyc": "Know Your Customer",
    # "cft": "Combating the Financing of Terrorism",
    # "mfs": "Mobile Financial Services",
    # "nbfi": "Non-Bank Financial Institution",
}

# Definitional question patterns
DEFINITIONAL_PATTERNS = [
    "what is",
    "what are",
    "tell me about",
    "explain",
    "describe",
    "define",
    "information about",
    "details about",
    "can you tell me",
    "i want to know",
]


def is_consequence_question(question: str) -> bool:
    if not question:
        return False
    lowered = question.lower()
    return any(trigger in lowered for trigger in COMPLIANCE_TRIGGER_KEYWORDS)


def is_definitional_question(question: str) -> bool:
    """Check if the question is asking for a definition or explanation."""
    if not question:
        return False
    lowered = question.lower()
    return any(pattern in lowered for pattern in DEFINITIONAL_PATTERNS)


def expand_acronyms_in_query(question: str) -> List[str]:
    """Expand acronyms in a query to include both acronym and full form variants."""
    if not question:
        return [""]
    
    variants: List[str] = [question]
    lowered = question.lower()
    
    # Find all acronyms in the query (as whole words)
    found_acronyms: List[tuple[str, str]] = []
    words = f" {lowered} ".split()
    for acronym, full_form in ACRONYM_MAPPINGS.items():
        # Check if acronym appears as a whole word
        if acronym in words or acronym.upper() in [w.upper() for w in words]:
            found_acronyms.append((acronym, full_form))
    
    # If no acronyms found, return original query
    if not found_acronyms:
        return [question]
    
    # Generate variants for each found acronym
    for acronym, full_form in found_acronyms:
        # Match acronym as whole word (case insensitive)
        pattern = re.compile(r'\b' + re.escape(acronym) + r'\b', re.IGNORECASE)
        
        # Variant 1: Replace with full form
        variant_full = pattern.sub(full_form, question)
        if variant_full.lower() != lowered:
            variants.append(variant_full)
        
        # Variant 2: Add full form alongside acronym
        variant_both = pattern.sub(f"{acronym} {full_form}", question)
        if variant_both.lower() != lowered:
            variants.append(variant_both)
        
        # Variant 3: Uppercase acronym version
        variant_upper = pattern.sub(acronym.upper(), question)
        if variant_upper.lower() != lowered:
            variants.append(variant_upper)
    
    # Deduplicate while preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for variant in variants:
        key = variant.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(variant.strip())
    
    return deduped if deduped else [question]


def build_query_variants(question: str) -> List[str]:
    """Generate semantic variants for queries, including acronym expansion and consequence question handling."""
    if not question:
        return [""]
    
    normalized = " ".join(question.strip().split())
    lowered = normalized.lower()
    
    # Always start with acronym expansion for all queries
    variants: List[str] = expand_acronyms_in_query(normalized)
    
    # Check if this is a definitional question
    is_def = is_definitional_question(lowered)
    
    # Check if this is a consequence question
    is_consequence = is_consequence_question(lowered)
    
    # For definitional questions, add more semantic variants
    if is_def:
        # Extract the main topic (remove definitional phrases)
        topic = lowered
        for pattern in DEFINITIONAL_PATTERNS:
            if pattern in topic:
                topic = topic.replace(pattern, "").strip()
                break
        
        # Add variants with different phrasings
        definitional_variants = [
            topic,
            f"definition of {topic}",
            f"what is {topic}",
            f"explain {topic}",
            f"information about {topic}",
        ]
        variants.extend(definitional_variants)
    
    # For consequence questions, add compliance-focused variants
    if is_consequence:
        focus_terms: List[str] = []
        if "pso" in lowered or "payment system operator" in lowered:
            focus_terms.append("Payment System Operator regulations")
        if "psd" in lowered or "payment systems department" in lowered:
            focus_terms.append("PSD circular compliance")
        if "money laundering" in lowered:
            focus_terms.append("Money Laundering Prevention Act directives")
        if "npsb" in lowered or "national payment switch bangladesh" in lowered:
            focus_terms.append("NPSB operating rules")
        
        if not focus_terms:
            focus_terms.append("Bangladesh Bank payment regulations")
        
        templates = [
            "penalties for violating {term}",
            "consequences of non-compliance with {term}",
            "Bangladesh Bank enforcement actions for {term}",
            "what happens if institutions ignore {term}",
        ]
        
        for term in focus_terms:
            for template in templates:
                variants.append(template.format(term=term))
    
    # Deduplicate while preserving order
    deduped: List[str] = []
    seen: Set[str] = set()
    for variant in variants:
        clean_variant = variant.strip()
        if not clean_variant:
            continue
        key = clean_variant.lower()
        if key in seen:
            continue
        deduped.append(clean_variant)
        seen.add(key)
    
    return deduped if deduped else [normalized]


# Common stop words to filter out from keyword extraction
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "will", "with", "what", "when", "where", "who", "why",
    "how", "can", "could", "should", "would", "may", "might", "must",
    "this", "these", "those", "they", "them", "their", "there", "then",
    "if", "or", "but", "not", "no", "yes", "do", "does", "did", "done",
    "have", "has", "had", "been", "being", "get", "got", "go", "went",
    "say", "said", "see", "saw", "know", "knew", "think", "thought",
    "take", "took", "come", "came", "want", "wanted", "use", "used",
    "find", "found", "give", "gave", "tell", "told", "work", "worked",
}


def extract_keywords_from_query(query: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """
    Extract important keywords from a query for document filtering.
    
    Args:
        query: The user's query string
        min_length: Minimum length for a keyword to be included
        max_keywords: Maximum number of keywords to return
    
    Returns:
        List of normalized keywords (lowercase, stripped)
    """
    if not query:
        return []
    
    # Normalize: lowercase and remove punctuation
    normalized = query.lower()
    # Remove punctuation but keep spaces
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words
    words = normalized.split()
    
    # Filter out stop words and short words
    keywords = []
    for word in words:
        word = word.strip()
        if len(word) >= min_length and word not in STOP_WORDS:
            keywords.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    # Limit to max_keywords
    return unique_keywords[:max_keywords]

