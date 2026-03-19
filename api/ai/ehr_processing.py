from spacy.tokens import Span, Token
import spacy
from typing import List, Dict, Tuple
from ehr_labels import  legitimate_context_patterns, discourse_marker_patterns, specificity_patterns, bias_indicator_patterns, race_ethnicity_patterns, socioeconomic_patterns, age_patterns, generalizing_patterns, deficit_patterns

def create_bias_detection_ruler(model_name):
    """Create spaCy pipeline with all pattern matching rules."""
    nlp = spacy.load("en_core_web_sm")
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")

    ruler = nlp.add_pipe("entity_ruler", before="ner")

    all_patterns = (
        legitimate_context_patterns +
        discourse_marker_patterns +
        specificity_patterns +
        bias_indicator_patterns +
        race_ethnicity_patterns +
        socioeconomic_patterns +
        age_patterns +
        generalizing_patterns +
        deficit_patterns
    )

    ruler.add_patterns(all_patterns)

    return nlp

def check_dependency_relationship(demographic_ent: Span, problematic_ent: Span) -> Tuple[bool, float]:
    """
    Check if demographic is the grammatical subject of problematic language.
    Returns (is_subject, dependency_weight)
    """
    # Get the root tokens of each span
    demo_root = demographic_ent.root
    prob_root = problematic_ent.root

    # Check if demographic is subject of the problematic verb
    # Walk up the dependency tree from problematic to see if demographic is the subject
    current = prob_root
    path_length = 0
    max_path = 5  # Don't traverse too far


    #Step 1: Set the default likelihood that the deompgrahic entity is biased based on the gramattical structuring
    for child in prob_root.children:
        print(f"  - '{child.text}' (idx: {child.i}, DEP: {child.dep_}, POS: {child.pos_})")

        # Check if this child is in demographic span
        if child.i >= demographic_ent.start and child.i < demographic_ent.end:
            print(f"    ✓ This child IS within demographic span!")
            if child.dep_ in ["nsubj", "nsubjpass"]:
                print(f"    ✓✓ AND it's a subject! Dependency: {child.dep_}")

                # Check if this is a direct action or just a state description
                if prob_root.dep_ in ["ROOT", "conj", "ccomp", "xcomp"]:
                    print(f"    → Direct action (prob_root.dep_={prob_root.dep_}), strong relationship")
                    return True, 1.0
                elif prob_root.dep_ in ["acomp", "amod"]:
                    print(f"    → State description (prob_root.dep_={prob_root.dep_}), weaker relationship")
                    return True, 0.6
                else:
                    print(f"    → Other dependency (prob_root.dep_={prob_root.dep_}), medium relationship")
                    return True, 0.8

    #Step 2:
    while current and path_length < max_path:
        print(f"  Step {path_length}: '{current.text}' (idx: {current.i}, DEP: {current.dep_})")

        # Check children at this level
        print(f"    Checking children of '{current.text}':")
        found_subject = False
        for child in current.children:
            print(f"      - '{child.text}' (idx: {child.i}, DEP: {child.dep_})")

            if child.dep_ in ["nsubj", "nsubjpass"]:
                # Check if this subject is within demographic span
                if child.i >= demographic_ent.start and child.i < demographic_ent.end:
                    print(f"        ✓✓ FOUND SUBJECT RELATIONSHIP!")

                # Check if this is a direct action or just a state description
                if prob_root.dep_ in ["ROOT", "conj", "ccomp", "xcomp"]:
                    print(f"        → Direct action (prob_root.dep_={prob_root.dep_}), strong relationship")
                    return True, 1.0
                elif prob_root.dep_ in ["acomp", "amod"]:
                    print(f"        → State description (prob_root.dep_={prob_root.dep_}), weaker relationship")
                    return True, 0.6
                else:
                    print(f"        → Other dependency (prob_root.dep_={prob_root.dep_}), medium relationship")
                    return True, 0.8


        # Move up the tree
        if current.head.i == current.i:  # Root of sentence
            break
        current = current.head
        path_length += 1

    # Check proximity as fallback

    token_distance = abs(demographic_ent.end - problematic_ent.start)
    if token_distance <= 5:
        return False, 0.8  # Close proximity
    elif token_distance <= 10:
        return False, 0.5  # Medium proximity
    else:
        return False, 0.2  # Far apart

def count_negations_in_scope(sent: Span, problematic_ent: Span) -> int:
    """
    Count negations that could affect the problematic entity.
    Focus on negations within the same clause or before the entity.
    """
    negation_markers = [
        "not", "never", "no", "neither", "nor", "n't",
        "nothing", "nobody", "nowhere"
    ]

    # Known double-negative words (prefix un-, in-, non-, etc.)
    negative_affixes = ["un", "in", "im", "non", "dis"]

    neg_count = 0

    # Check tokens before and around problematic entity
    start_check = max(0, problematic_ent.start - 10)
    end_check = problematic_ent.end + 2

    for token in sent[start_check:end_check]:
        # Explicit negation markers
        if token.text.lower() in negation_markers or token.text.endswith("n't"):
            neg_count += 1

        # Words with negative prefixes (uncommon, improper, non-compliant)
        elif any(token.text.lower().startswith(prefix) for prefix in negative_affixes):
            # Only count if it's actually negative (not "uncle" or "inch")
            if len(token.text) > 4:  # Simple heuristic
                neg_count += 1

    return neg_count

def analyze_negation_scope(sent: Span, negation_tokens: List[Token],
                           problematic_ent: Span) -> Dict:
    """
    Determine if negation directly modifies the problematic language
    or modifies something else (like "assume", "believe", etc.)
    """
    results = {
        "has_direct_negation": False,
        "has_indirect_negation": False,
        "metalinguistic": False
    }

    prob_root = problematic_ent.root

    for neg_token in negation_tokens:
        # Check if negation is a direct dependent of problematic word
        if neg_token.head == prob_root:
            results["has_direct_negation"] = True

        # Check if negation modifies a parent verb (like "assume", "believe")
        elif neg_token.head.pos_ == "VERB" and neg_token.head.text.lower() in \
             ["assume", "believe", "think", "claim", "say", "suggest"]:
            results["has_indirect_negation"] = True

        # Check for metalinguistic markers (quotes, scare quotes)
        # Look for quotes around problematic term
        quote_nearby = any(t.text in ["'", '"', "'", """, """]
                          for t in sent[max(0, problematic_ent.start-2):problematic_ent.end+2])
        if quote_nearby:
            results["metalinguistic"] = True

    return results

def detect_partial_negation(sent: Span) -> bool:
    """
    Detect partial negations like "not all", "not every", "not always"
    These weaken claims but don't fully negate them.
    """
    text_lower = sent.text.lower()

    partial_patterns = [
        "not all", "not every", "not always",
        "not necessarily", "not entirely", "not completely"
    ]

    return any(pattern in text_lower for pattern in partial_patterns)

def advanced_negation_analysis(sent: Span, problematic_ents: List[Span]) -> Dict:
    """
    Comprehensive negation analysis returning detailed classification.

    Returns:
        Dict with keys: negation_type, multiplier, confidence, note
    """
    # Get all negation tokens from sentence
    negation_tokens = []
    for token in sent:
        if token.text.lower() in ["not", "never", "no", "neither", "nor"] or token.text.endswith("n't"):
            negation_tokens.append(token)

    # Also check for NEGATION entities from patterns
    negation_ents = [ent for ent in sent.ents if ent.label_ == "NEGATION"]
    for ent in negation_ents:
        negation_tokens.extend([t for t in ent])

    if not negation_tokens:
        return {
            "negation_type": "none",
            "multiplier": 1.0,
            "confidence": "high",
            "note": "No negation detected"
        }

    # Count negations in scope
    neg_count = 0
    for prob_ent in problematic_ents:
        neg_count += count_negations_in_scope(sent, prob_ent)

    # Check scope
    scope_analysis = analyze_negation_scope(
        sent,
        negation_tokens,
        problematic_ents[0] if problematic_ents else None
    )

    # Check for partial negation
    is_partial = detect_partial_negation(sent)


    # Case 2: Metalinguistic negation (refuting the framing)
    if scope_analysis["metalinguistic"]:
        return {
            "negation_type": "metalinguistic",
            "multiplier": 0.1,
            "confidence": "high",
            "note": "Negating the terminology/framing itself"
        }

    # Case 3: Direct negation of problematic language
    if scope_analysis["has_direct_negation"]:
        return {
            "negation_type": "direct_negation",
            "multiplier": 0.1,
            "confidence": "high",
            "note": "Directly negates the problematic statement"
        }

    # Case 4: Indirect negation (negating "assume", "believe", etc.)
    if scope_analysis["has_indirect_negation"]:
        return {
            "negation_type": "indirect_negation",
            "multiplier": 0.2,
            "confidence": "medium",
            "note": "Negates belief/assumption about the statement"
        }

    # Case 5: Partial negation ("not all")
    if is_partial:
        return {
            "negation_type": "partial_negation",
            "multiplier": 0.5,
            "confidence": "medium",
            "note": "Partial negation - weakens but doesn't eliminate bias"
        }

    # Case 6: Simple negation (odd count)
    if neg_count % 2 == 1:
        return {
            "negation_type": "simple_negation",
            "multiplier": 0.15,
            "confidence": "medium",
            "note": "Single negation detected"
        }

    # Default: unclear negation
    return {
        "negation_type": "unclear",
        "multiplier": 0.5,
        "confidence": "low",
        "note": "Negation present but scope unclear"
    }

def check_for_negation(sent: Span, problematic_ents: List[Span]) -> bool:
    """Check if sentence contains negation markers near problematic language."""
    negations = [ent for ent in sent.ents if ent.label_ == "NEGATION"]

    if not negations:
        return False

    # Check if negation appears before problematic language
    for neg in negations:
        for prob in problematic_ents:
            # Negation should appear within 5 tokens before the problematic language
            if 0 <= (prob.start - neg.end) <= 5:
                return True

    return False

def check_discourse_context(sent: Span) -> Dict[str, bool]:
    """Check for discourse markers that modify interpretation."""
    return {
        "has_contrast": any(ent.label_ == "CONTRAST" for ent in sent.ents),
        "has_historical": any(ent.label_ == "HISTORICAL" for ent in sent.ents),
        "has_attribution": any(ent.label_ == "ATTRIBUTION" for ent in sent.ents),
    }


def assess_specificity(sent: Span, demographics: List[Span]) -> float:
    """
    Assess whether demographic mention is specific (individual) or general (group).
    Returns specificity_factor: 0.0 (very general) to 1.0 (very specific)
    """
    # Check for specific patient markers
    if any(ent.label_ == "SPECIFIC_PATIENT" for ent in sent.ents):
        return 1.0  # Very specific - individual patient

    # Check for plural forms (more general)
    for demo in demographics:
        if demo.root.tag_ in ["NNS", "NNPS"]:  # Plural nouns
            # Check for determiners
            for token in demo:
                if token.dep_ == "det" and token.text.lower() in ["the", "these", "those"]:
                    return 0.6  # Specific group
            return 0.2  # General group (e.g., "Black patients" without article)
        else:
            # Singular - check for determiners
            for token in demo:
                if token.dep_ == "det" and token.text.lower() in ["a", "an", "the", "this", "that"]:
                    return 0.9  # Specific individual
            return 0.4  # Singular but no determiner

    return 0.5  # Default

def _categorize_confidence(risk_score: float) -> str:
    """Categorize confidence level based on risk score."""
    if risk_score >= 4:
        return "high_risk"
    elif risk_score >= 2:
        return "medium_risk"
    elif risk_score >= 0.5:
        return "low_risk"
    else:
        return "minimal_risk"


def _get_llm_priority(risk_score: float) -> str:
    """Determine LLM review priority."""
    if risk_score >= 4:
        return "immediate_individual"
    elif risk_score >= 2:
        return "batch_review"
    else:
        return "skip"

def detect_bias_with_context(doc) -> List[Dict]:
    """
    Advanced bias detection using dependency parsing, discourse analysis,
    and sophisticated negation handling.
    """
    bias_flags = []

    for sent in doc.sents:
        # Extract entities by type
        demographics = [ent for ent in sent.ents
                       if ent.label_ in ["AGE_MENTION", "RACE_ETHNICITY", "SOCIOECONOMIC"]]
        generalizing = [ent for ent in sent.ents if ent.label_ == "GENERALIZING_LANG"]
        deficit = [ent for ent in sent.ents if ent.label_ == "DEFICIT_LANG"]
        bias_indicators = [ent for ent in sent.ents if ent.label_ == "BIAS_INDICATOR"]
        legitimate_contexts = [ent for ent in sent.ents if ent.label_ == "LEGITIMATE_CONTEXT"]

        if not demographics:
            continue

        # Get all problematic language
        all_problematic = generalizing + deficit + bias_indicators
        if not all_problematic:
            continue

        # === MULTI-FACTOR ANALYSIS ===

        # 1. Advanced negation analysis
        negation_stats = advanced_negation_analysis(sent, all_problematic)

        # 2. Check discourse context
        discourse = check_discourse_context(sent)

        # 3. Assess specificity
        specificity = assess_specificity(sent, demographics)

        # 4. Check dependency relationships
        max_dependency_weight = 0.0
        has_subject_relationship = False
        for demo in demographics:
            for prob in all_problematic:
                is_subj, dep_weight = check_dependency_relationship(demo, prob)
                if is_subj:
                    has_subject_relationship = True
                max_dependency_weight = max(max_dependency_weight, dep_weight)

        # === CALCULATE BASE RISK ===
        base_risk = 0

        if bias_indicators:
            base_risk += 5  # Explicit bias language
        if deficit:
            base_risk += 3
        if generalizing:
            base_risk += 2

        # === APPLY MODIFIERS ===

        # Start with base risk
        risk_score = base_risk
        risk_factors = []

        # Apply negation multiplier
        risk_score *= negation_stats["multiplier"]
        if negation_stats["negation_type"] != "none":
            risk_factors.append(f"negation_{negation_stats['negation_type']}")

        # Only continue with other modifiers if not heavily negated
        if negation_stats["multiplier"] >= 0.3:
            # Dependency modifier (0.2 to 1.0)
            risk_score *= max_dependency_weight
            if has_subject_relationship:
                risk_factors.append("demographic_is_subject")
            elif max_dependency_weight < 0.5:
                risk_factors.append("weak_relationship")

            # Specificity modifier (general = higher risk)
            # Invert: 0.2 (general) becomes 1.0, 1.0 (specific) becomes 0.2
            specificity_modifier = 1.2 - specificity
            risk_score *= specificity_modifier
            if specificity > 0.7:
                risk_factors.append("specific_patient_reference")
            elif specificity < 0.3:
                risk_factors.append("broad_generalization")

            # Discourse modifiers
            if discourse["has_contrast"]:
                risk_score *= 0.3
                risk_factors.append("contrast_marker")
            if discourse["has_historical"]:
                risk_score *= 0.2
                risk_factors.append("historical_discussion")
            if discourse["has_attribution"]:
                risk_score *= 0.4
                risk_factors.append("attributed_to_others")

            # Legitimate context
            if legitimate_contexts:
                risk_score *= 0.5
                risk_factors.append("legitimate_context")

        # Only flag if meaningful risk remains
        if risk_score >= 0.5:
            bias_flags.append({
                "sentence": sent.text.strip(),
                "start_char": sent.start_char,
                "end_char": sent.end_char,
                "demographics": [{"text": e.text, "label": e.label_} for e in demographics],
                "problematic_language": {
                    "generalizing": [e.text for e in generalizing],
                    "deficit": [e.text for e in deficit],
                    "bias_indicators": [e.text for e in bias_indicators],
                },
                "context_analysis": {
                    "negation_type": negation_stats["negation_type"],
                    "negation_multiplier": negation_stats["multiplier"],
                    "negation_note": negation_stats["note"],
                    "has_contrast": discourse["has_contrast"],
                    "has_historical": discourse["has_historical"],
                    "has_attribution": discourse["has_attribution"],
                    "specificity_score": round(specificity, 2),
                    "dependency_weight": round(max_dependency_weight, 2),
                    "has_subject_relationship": has_subject_relationship,
                },
                "risk_score": round(risk_score, 2),
                "base_risk": base_risk,
                "risk_factors": risk_factors,
                "confidence": _categorize_confidence(risk_score),
                "needs_llm_review": risk_score >= 2,
                "llm_priority": _get_llm_priority(risk_score)
            })

    bias_flags.sort(key=lambda x: x["risk_score"], reverse=True)
    return bias_flags

