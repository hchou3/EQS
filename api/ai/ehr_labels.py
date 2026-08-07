import spacy
# ============================================================================
#  CONTEXT PATTERNS
# ============================================================================

legitimate_context_patterns = [
    # Epidemiological/Research contexts
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "studies"}, {"LOWER": {"IN": ["show", "indicate", "suggest", "demonstrate"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "research"}, {"LOWER": {"IN": ["shows", "indicates", "suggests", "demonstrates"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "data"}, {"LOWER": {"IN": ["shows", "indicates", "suggests", "demonstrates"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": {"IN": ["statistically", "epidemiologically"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": {"IN": ["prevalence", "incidence", "mortality", "morbidity"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "according"}, {"LOWER": "to"}]},

    # Clinical guidelines
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "screening"}, {"LOWER": {"IN": ["recommendations", "guidelines"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "clinical"}, {"LOWER": {"IN": ["guidelines", "protocol", "protocols"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "risk"}, {"LOWER": {"IN": ["stratification", "assessment", "factors"]}}]},

    # Social determinants
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "social"}, {"LOWER": {"IN": ["determinants", "factors"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": {"IN": ["structural", "systemic"]}}, {"LOWER": {"IN": ["barriers", "factors", "racism", "inequality"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "access"}, {"LOWER": "to"}, {"LOWER": {"IN": ["care", "healthcare", "treatment"]}}]},
    {"label": "LEGITIMATE_CONTEXT", "pattern": [{"LOWER": "health"}, {"LOWER": {"IN": ["disparities", "inequities", "equity"]}}]},
]

# ============================================================================
# DISCOURSE MARKERS (Negation, Contrast, Historical)
# ============================================================================

discourse_marker_patterns = [
    # Negation
    {"label": "NEGATION", "pattern": [{"LOWER": {"IN": ["not", "never", "no", "neither", "nor"]}}]},
    {"label": "NEGATION", "pattern": [{"LOWER": "should"}, {"LOWER": "not"}]},
    {"label": "NEGATION", "pattern": [{"LOWER": "do"}, {"LOWER": "not"}]},
    {"label": "NEGATION", "pattern": [{"LOWER": "does"}, {"LOWER": "not"}]},
    {"label": "NEGATION", "pattern": [{"LOWER": {"IN": ["incorrect", "wrong", "false"]}}, {"LOWER": "to"}]},
    {"label": "NEGATION", "pattern": [{"LOWER": "myth"}, {"LOWER": "that"}]},
    {"label": "NEGATION", "pattern": [{"LOWER": "stereotype"}, {"LOWER": "that"}]},

    # Contrast/Comparison
    {"label": "CONTRAST", "pattern": [{"LOWER": {"IN": ["unlike", "contrary", "however", "whereas", "although", "though"]}}]},
    {"label": "CONTRAST", "pattern": [{"LOWER": "while"}, {"LOWER": "some"}]},
    {"label": "CONTRAST", "pattern": [{"LOWER": "in"}, {"LOWER": "contrast"}]},
    {"label": "CONTRAST", "pattern": [{"LOWER": "on"}, {"LOWER": "the"}, {"LOWER": {"IN": ["contrary", "other"]}}]},

    # Historical/Hypothetical
    {"label": "HISTORICAL", "pattern": [{"LOWER": {"IN": ["historically", "previously", "formerly", "traditionally"]}}]},
    {"label": "HISTORICAL", "pattern": [{"LOWER": "in"}, {"LOWER": "the"}, {"LOWER": "past"}]},
    {"label": "HISTORICAL", "pattern": [{"LOWER": "used"}, {"LOWER": "to"}]},

    # Attribution to others (not author's view)
    {"label": "ATTRIBUTION", "pattern": [{"LOWER": "some"}, {"LOWER": {"IN": ["believe", "think", "assume", "claim"]}}]},
    {"label": "ATTRIBUTION", "pattern": [{"LOWER": "it"}, {"LOWER": "is"}, {"LOWER": {"IN": ["believed", "thought", "assumed"]}}]},
]

# ============================================================================
# SPECIFICITY MARKERS
# ============================================================================

specificity_patterns = [
    # Specific patient references (low risk)
    {"label": "SPECIFIC_PATIENT", "pattern": [{"LOWER": "the"}, {"LOWER": {"IN": ["patient", "individual", "person"]}}]},
    {"label": "SPECIFIC_PATIENT", "pattern": [{"LOWER": {"IN": ["this", "that"]}}, {"LOWER": {"IN": ["patient", "individual", "person"]}}]},
    {"label": "SPECIFIC_PATIENT", "pattern": [{"LOWER": "a"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "yr"]}}, {"LOWER": "old"}]},
]

bias_indicator_patterns = [
    {"label": "BIAS_INDICATOR", "pattern": [{"LOWER": "due"}, {"LOWER": "to"}, {"LOWER": {"IN": ["their", "the"]}},
                                            {"LOWER": {"IN": ["culture", "race", "ethnicity", "background"]}}]},
    {"label": "BIAS_INDICATOR", "pattern": [{"LOWER": "because"}, {"LOWER": "of"}, {"LOWER": {"IN": ["their", "the"]}},
                                            {"LOWER": {"IN": ["culture", "race", "ethnicity", "background"]}}]},
    {"label": "BIAS_INDICATOR", "pattern": [{"LOWER": {"IN": ["inherently", "naturally", "innately"]}}]},
    {"label": "BIAS_INDICATOR", "pattern": [{"LOWER": "cultural"}, {"LOWER": {"IN": ["beliefs", "practices", "tendencies"]}}]},
]

race_ethnicity_patterns = [
    {"label": "RACE_ETHNICITY", "pattern": [{"LOWER": {"IN": ["black", "african", "african-american", "afro-american"]}}]},
    {"label": "RACE_ETHNICITY", "pattern": [{"LOWER": "white"}, {"LOWER": {"IN": ["patient", "patients", "individual", "individuals"]}}]},
    {"label": "RACE_ETHNICITY", "pattern": [{"LOWER": {"IN": ["hispanic", "latino", "latina", "latinx"]}}]},
    {"label": "RACE_ETHNICITY", "pattern": [{"LOWER": {"IN": ["asian", "pacific", "islander"]}}]},
    {"label": "RACE_ETHNICITY", "pattern": [{"LOWER": "native"}, {"LOWER": "american"}]},
    {"label": "RACE_ETHNICITY", "pattern": [{"LOWER": {"IN": ["caucasian", "indigenous", "minority", "minorities"]}}]},
]

socioeconomic_patterns = [
    {"label": "SOCIOECONOMIC", "pattern": [{"LOWER": "low"}, {"LOWER": {"IN": ["income", "ses", "socioeconomic"]}}]},
    {"label": "SOCIOECONOMIC", "pattern": [{"LOWER": {"IN": ["poor", "impoverished", "disadvantaged", "underserved"]}}]},
    {"label": "SOCIOECONOMIC", "pattern": [{"LOWER": {"IN": ["uninsured", "medicaid"]}}]},
    {"label": "SOCIOECONOMIC", "pattern": [{"LOWER": "inner"}, {"LOWER": "city"}]},
]

age_patterns = [
    {"label": "AGE_MENTION", "pattern": [{"LOWER": {"IN": ["elderly", "geriatric", "senior", "aged"]}}]},
    {"label": "AGE_MENTION", "pattern": [{"LOWER": "older"}, {"LOWER": {"IN": ["adult", "patient", "adults", "patients"]}}]},
    {"label": "AGE_MENTION", "pattern": [{"LOWER": "young"}, {"LOWER": {"IN": ["adult", "patient"]}}]},
]

generalizing_patterns = [
    {"label": "GENERALIZING_LANG", "pattern": [{"LOWER": {"IN": ["all", "every", "always", "never", "typically", "usually", "generally", "tend", "tends"]}}]},
    {"label": "GENERALIZING_LANG", "pattern": [{"LOWER": "most"}]},
]

deficit_patterns = [
    {"label": "DEFICIT_LANG", "pattern": [{"LOWER": {"IN": ["unable", "incapable", "cannot", "can't", "won't"]}}]},
    {"label": "DEFICIT_LANG", "pattern": [{"LOWER": {"IN": ["lacking", "deficient", "insufficient"]}}]},
    {"label": "DEFICIT_LANG", "pattern": [{"LOWER": {"IN": ["limited", "poor"]}},
                                          {"LOWER": {"IN": ["understanding", "comprehension", "ability", "compliance"]}}]},
    {"label": "DEFICIT_LANG", "pattern": [{"LOWER": {"IN": ["confused", "disoriented"]}}]},
    {"label": "DEFICIT_LANG", "pattern": [{"LOWER": {"IN": ["refuses", "refused", "unwilling", "non-compliant", "noncompliant"]}}]},
]

def create_bias_detection_ruler(model_name):
    """Create spaCy pipeline with all pattern matching rules."""
    nlp = spacy.load(model_name)
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