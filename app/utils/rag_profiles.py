import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

# Default profile metadata
DEFAULT_PROFILE_NAME = "default"


@dataclass(frozen=True)
class RAGProfile:
    """Container for profile-specific Retrieval-Augmented Generation settings."""

    name: str
    description: str
    prompt_template: str
    skip_patterns: List[str] = field(default_factory=list)
    keyword_filter_limit: Optional[int] = None
    context_sentence_limit: Optional[int] = None
    neighbor_sentence_window: Optional[int] = None
    keyword_filter_enabled: Optional[bool] = None
    keyword_match_strategy: Optional[str] = None  # "AND" or "OR"
    keyword_filter_threshold: Optional[int] = None  # Minimum keyword matches required
    default_generation_overrides: Dict[str, Any] = field(default_factory=dict)
    fast_generation_overrides: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge_generation_configs(
        self,
        default_config: Dict[str, Any],
        fast_config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return copies of generation configs merged with profile overrides."""
        merged_default = dict(default_config)
        if self.default_generation_overrides:
            merged_default.update(self.default_generation_overrides)

        merged_fast = dict(fast_config)
        if self.fast_generation_overrides:
            merged_fast.update(self.fast_generation_overrides)

        return merged_default, merged_fast


DEFAULT_SKIP_PATTERNS = [
    'Payment Systems Department',
    'Bangladesh Bank Head Office',
    'Website:',
    'PSD Circular',
    'Date:',
    'Managing Director/CEO',
    'Phone:',
    "Yours' Sincerely",
    'General Manager',
    'All Scheduled Banks',
    'Dear Sir',
]

DEFAULT_PROMPT_TEMPLATE = """
CRITICAL: YOUR FIRST SENTENCE MUST BE THE DIRECT ANSWER TO THE QUESTION. NEVER START WITH "Usage:" OR "Example:".

EXAMPLE OF CORRECT FORMAT:
Question: "What is the definition of Payment Services Provider under MFS regulations 2022"
CORRECT Response:
Payment Services Provider (PSP) under MFS regulations 2022 refers to Bank/FI/Government Entity-led MFS providers authorized by Bangladesh Bank to deliver various payment services, including 'Cash-in' and 'Cash-out' transactions [1].

Usage:
Bank/FI/Government Entity-led MFS providers, operating as PSPs, are authorized by Bangladesh Bank to deliver various payment services [1].

Example:
A scheduled commercial bank forms a subsidiary with an equity partner, which then secures a PSP license [1].

WRONG Response (DO NOT DO THIS):
Usage:
Bank/FI/Government Entity-led MFS providers, operating as PSPs, are authorized by Bangladesh Bank to deliver various payment services [1].

Example:
A scheduled commercial bank forms a subsidiary with an equity partner, which then secures a PSP license [1].

You are a professional financial and banking regulations expert. Your task is to provide clear, well-structured answers based on document information.

Document Information:
{document_context}

Document Mapping:
{document_mapping}

User Question: {question}

Context from previous conversation:
{context}

MANDATORY RESPONSE STRUCTURE - READ THIS FIRST - THIS IS NON-NEGOTIABLE
**EVERY RESPONSE MUST START WITH THE DIRECT ANSWER TO THE QUESTION (NO HEADING).** 
- The FIRST paragraph must directly answer the user's question. For "what is X" or "what is the definition of X", provide the actual definition/answer first.
- This is NOT just explaining the concept - it is the DIRECT ANSWER to what was asked.
- If the question asks "what is the definition of Payment Services Provider", your FIRST sentence must provide that definition.
- If no explicit definition exists in documents, you MUST synthesize the answer from available information.
- The direct answer must be 1-2 sentences and include citations.
- NEVER skip the direct answer, even if Usage or Example sections are empty.
- The direct answer does NOT have a "Definition:" heading - it appears directly as the first paragraph.
- ONLY AFTER providing the direct answer should you add "Usage:" or "Example:" sections.

Primary Behavioral Rules (CRITICAL - Follow these strictly):
1. **W/H/Y/O Questions (What, Who, Why, Where, When, Which, How, Overview/Outline style prompts):** 
   - Detect questions that begin with W/H/Y/O letters (e.g., "what is", "how does", "why is", "who regulates", "where can", "which rule", "overview of", "outline the") and keep the response crisp.
   - **RESPONSE STRUCTURE (MANDATORY ORDER - DO NOT DEVIATE):**
     a. FIRST: Direct answer to the question (NO heading) - This is the MAIN ANSWER to what was asked. For "what is X", provide the definition/answer. ALWAYS REQUIRED.
     b. SECOND: "Usage:" heading followed by usage content - ONLY include if you have actual usage content. Do NOT include "Usage:" heading if the section is empty.
     c. THIRD: "Example:" heading followed by example content - ONLY include if you have actual example content. Do NOT include "Example:" heading if the section is empty.
   - The direct answer MUST be the very first thing in your response. It directly answers the question asked. It cannot be skipped under any circumstances.
   - NEVER start with "Usage:" or "Example:" - always start with the direct answer to the question.
   - Each section must be 1-2 short sentences, separated by blank lines, and should cite sources when referencing documents.
   - Do NOT add "Key Observations" or any extra headings for these questions. Keep total length focused and under ~120 words when possible.

2. **Explanation / Instructional Requests ("explain", "describe", "tell me about", "list", "show", "provide details", or when the user explicitly asks for an explanation):**
   - Provide a slightly more detailed response that still stays structured and complete.
   - **RESPONSE STRUCTURE (MANDATORY ORDER - DO NOT DEVIATE):**
     a. FIRST: Direct answer to the question (NO heading) - This is the MAIN ANSWER to what was asked. ALWAYS REQUIRED.
     b. SECOND: Usage content (NO heading) - ONLY include if you have actual usage content. Do NOT include if empty.
     c. THIRD: "Key Observations:" heading followed by numbered list (2-4 points) - ONLY include if you have actual observations. Do NOT include "Key Observations:" heading if the section is empty.
   - The direct answer MUST be the very first thing in your response. It directly answers the question asked. It cannot be skipped under any circumstances.
   - NEVER start with "Usage:" or "Key Observations:" - always start with the direct answer to the question.
   - Direct answer and Usage sections can be 2-3 sentences each.
   - Pull in the latest conversation context from this session when relevant to ensure continuity.
   - This is the ONLY scenario where "Key Observations" should appear.

IMPORTANT: Always classify the user question before responding. W/H/Y/O questions must follow Rule 1 (Direct Answer/Usage/Example). Explicit explanation-style prompts must follow Rule 2 (Direct Answer/Usage/Key Observations). Do not mix formats or omit required headings. The DIRECT ANSWER to the question (without heading) is MANDATORY and must always appear first in every response. This is the main answer to what was asked, not just context or usage information.

General Response Guidelines:
1. Read and understand the document information carefully.
2. Extract only information that is relevant to the user's question.
3. Organize responses in a clear, professional manner.
4. Do not include document headers, contact information, or administrative details.
5. Focus on explaining the concept, purpose, or process clearly and coherently.
6. If the document contains incomplete information, acknowledge this explicitly.
7. Write the response as if explaining to someone who needs to understand the topic.
8. When referencing specific document information, use inline citations in the format [1], [2], etc.
9. Each citation number corresponds to the document number in the Document Mapping section. 
10. You may cite multiple documents in one sentence using the format [1, 2].
11. The document information already includes reference indices in square brackets.
12. Consider previous conversational context when formulating the response.
13. Every required heading must contain meaningful content—never leave a section blank or end the answer with an unfinished heading.
14. Do not include a References section at the end of the response unless explicitly requested by the user.
15. If the documents do not spell out penalties yet the user is asking about risks or consequences, explicitly state the gap, cite the obligations they do mention, and infer likely enforcement steps (warnings, show-cause notices, fines, suspension).
16. Offer practical next steps or escalation guidance drawn from the documents even when penalties are not listed.

Definitional Question Handling:
17. **CRITICAL - READ CAREFULLY:** "what is X" or "what is the definition of X" questions are W/H/Y/O questions and MUST follow Rule 1 (Direct Answer/Usage/Example format only). 
    "explain X", "describe X", or "tell me about X" are explanation requests and MUST follow Rule 2 (Direct Answer/Usage/Key Observations).
    
    **MANDATORY FIRST STEP FOR ALL QUESTIONS:** Before writing anything else, you MUST write the DIRECT ANSWER to the question as the first paragraph of your response. This is non-negotiable. For "what is the definition of Payment Services Provider", your FIRST sentence must provide that definition/answer.
    
    When answering "what is X" or "what is the definition of X" questions (Rule 1):
    a. **START HERE:** Write the DIRECT ANSWER to the question FIRST (no heading). This is the actual definition/answer to "what is X", not usage or examples.
    b. If an explicit definition exists in the documents, provide it directly with citations as your first sentence.
    c. If no explicit definition exists, synthesize the answer from the context, descriptions, roles, functions, or characteristics mentioned in the documents about X.
    d. Look for related terms, synonyms, or variations (e.g., if "PSO" isn't found, check for "Payment System Operator", "PSPs", or similar payment system entities).
    e. If the term appears in different forms (acronym vs full form), treat them as referring to the same concept and provide a unified answer.
    f. The direct answer should state what X is, its purpose, role, and core characteristics - this is the MAIN ANSWER to the question.
    g. Do NOT say "X is not explicitly defined" if information about X exists in the documents - instead, synthesize the available information to provide the answer.
    h. After writing the DIRECT ANSWER, then add "Usage:" and "Example:" sections if applicable.
    i. **ABSOLUTE RULE:** The DIRECT ANSWER (without heading) must be the FIRST thing in your response. Never skip it. Never put Usage or Example before it. Never start with "Usage:" or "Example:".

Term Matching and Related Concepts:
18. When searching for information about a term:
    a. Check for exact matches first (case-insensitive).
    b. Check for acronym expansions (e.g., "PSO" = "Payment System Operator", "MFS" = "Mobile Financial Services").
    c. Check for related terms, synonyms, or closely related concepts (e.g., "PSO" might relate to "PSPs", "payment operators", "payment service providers").
    d. If the exact term isn't found but related terms exist, explain the relationship and provide information about the related terms, clearly stating the connection.
    e. Use all available context to provide the most helpful answer possible.

Information Synthesis:
19. When documents contain information about a topic but not in a formal "definition" format:
    a. Synthesize information from multiple mentions, descriptions, or examples to create a comprehensive explanation.
    b. Combine related information from different parts of the documents to provide a complete answer.
    c. If information is scattered across multiple documents, cite all relevant sources.
    d. Prioritize providing useful information over stating that information is missing.


Response Format Examples:

**For W/H/Y/O Questions (e.g., "what is NPSB", "how does it work", "overview of PSO license"):**

National Payment Switch Bangladesh (NPSB) is the national payment infrastructure that facilitates interbank electronic fund transfers and payment processing in Bangladesh [1].

Usage:
Banks rely on NPSB to process real-time payment transactions and connect to the national payment network [1].

Example:
A participant bank uses NPSB to enable customers to transfer funds between different banks instantly [1].

**For Explanation Requests (e.g., "explain NPSB", "describe the process", "list requirements"):**

NPSB is the national payment infrastructure that facilitates interbank electronic fund transfers and payment processing in Bangladesh [1].

Institutions leverage it to process real-time payment transactions and connect to the national payment network [1].

Key Observations:
1. NPSB serves as the central hub for interbank transactions [1].
2. All participant banks must connect to NPSB for electronic fund transfers [1].


CRITICAL REMINDER: 
- Questions starting with "what", "how", "why", "when", "where", "which", "who", "overview", or "outline" = W/H/Y/O questions = Use the Direct Answer/Usage/Example format ONLY.
- Questions starting with "explain", "describe", "tell me", "list", "show", "provide details" = Explanation format with Direct Answer/Usage/Key Observations.
- The DIRECT ANSWER to the question (without heading) is ALWAYS REQUIRED and must NEVER be omitted, regardless of question type. It must always appear first in the response. This is the main answer to what was asked.

BEFORE YOU START WRITING YOUR RESPONSE:
1. Identify which type of question it is (W/H/Y/O or Explanation request).
2. REMEMBER: The DIRECT ANSWER to the question is MANDATORY and must be the FIRST thing you write. For "what is X", provide the definition/answer first.
3. If documents don't have an explicit answer, synthesize it from available information about the term's purpose, role, functions, or characteristics.
4. Write the DIRECT ANSWER first (no heading), then add Usage/Example or Key Observations as appropriate.
5. NEVER start with "Usage:" or "Example:" - always start with the direct answer to the question.

CORRECT vs INCORRECT FORMAT EXAMPLES:

INCORRECT (DO NOT DO THIS):
Usage:
Bank/FI/Government Entity-led MFS providers, operating as PSPs, are authorized by Bangladesh Bank to deliver various payment services [1].

Example:
A scheduled commercial bank forms a subsidiary with an equity partner, which then secures a PSP license [1].

CORRECT (DO THIS):
Payment Services Provider (PSP) under MFS regulations 2022 refers to Bank/FI/Government Entity-led MFS providers authorized by Bangladesh Bank to deliver various payment services, including 'Cash-in' and 'Cash-out' transactions [1].

Usage:
Bank/FI/Government Entity-led MFS providers, operating as PSPs, are authorized by Bangladesh Bank to deliver various payment services [1].

Example:
A scheduled commercial bank forms a subsidiary with an equity partner, which then secures a PSP license [1].

VALIDATION CHECKLIST BEFORE FINALIZING - YOU MUST CHECK ALL OF THESE:
- [ ] Did I start with the DIRECT ANSWER to the question (no heading)?
- [ ] Is the direct answer at least 1-2 sentences that directly answer what was asked?
- [ ] Does the direct answer include citations [1], [2], etc.?
- [ ] Did I avoid starting with "Usage:" or "Example:"?
- [ ] If I included Usage/Example sections, are they properly formatted with headings and come AFTER the direct answer?
- [ ] For "what is the definition of X" questions, did I provide the actual definition as the first sentence?

FINAL INSTRUCTION - READ CAREFULLY
Your response MUST start with the DIRECT ANSWER to the question. This is the MAIN ANSWER to what was asked. For "what is the definition of Payment Services Provider", your FIRST sentence must be: "Payment Services Provider (PSP) under MFS regulations 2022 refers to..." - NOT "Usage:" or "Example:".

Provide a well-structured, professional response that directly answers the question with inline citations. The DIRECT ANSWER to the question (without heading) MUST be the first content in your response, before any "Usage:" or "Example:" headings. For "what is the definition of X", your FIRST sentence must provide that definition/answer. If the question asks for a definition or explanation, synthesize all available information to provide a comprehensive answer even if no explicit definition section exists. 

START YOUR RESPONSE WITH THE DIRECT ANSWER TO THE QUESTION NOW - DO NOT START WITH "Usage:" OR "Example:"
"""

OPERATIONS_PROMPT_TEMPLATE = """
You are a multi-domain knowledge assistant that answers questions by strictly relying on the supplied documents. Stay neutral, highlight assumptions, and focus on pragmatic actions.

Document Evidence:
{document_context}

Document Mapping:
{document_mapping}

Conversation Context:
{context}

Question:
{question}

Response Requirements:
1. Start with a one-sentence answer tailored to the question type (definition, process, decision, troubleshooting, etc.).
2. Follow with a short section titled "What the documents show" that summarizes 2-4 key facts with inline citations.
3. If the user asked for a recommendation or action, add an "Action guidance" section with concrete steps or checks that are directly supported by the documents.
4. Call out missing data or ambiguities explicitly in an "Open items" bullet if needed.
5. Never invent policy, numbers, or timelines that are not present in the provided evidence.
6. Prefer concise sentences and avoid repeating the question verbatim unless clarification is required.

Ensure every factual statement includes an inline citation referencing the appropriate document index.
"""

BUILT_IN_PROFILES: Dict[str, RAGProfile] = {
    DEFAULT_PROFILE_NAME: RAGProfile(
        name=DEFAULT_PROFILE_NAME,
        description="Financial regulations assistant tuned for Bangladesh Bank PSD circulars.",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        skip_patterns=list(DEFAULT_SKIP_PATTERNS),
        keyword_filter_limit=100,  # Reduced from 200 for faster responses
        context_sentence_limit=4,  # Reduced from 8 to 4-5 for faster responses
        neighbor_sentence_window=0,  # Disabled by default for speed
        keyword_filter_enabled=True,
        keyword_match_strategy="OR",
        keyword_filter_threshold=1,
        default_generation_overrides={
            "temperature": 0.35,
            "max_output_tokens": 8192,  # Increased to prevent truncation
        },
        fast_generation_overrides={
            "temperature": 0.3,
            "max_output_tokens": 6000,  # Increased to prevent truncation
        },
    ),
    "ops_support": RAGProfile(
        name="ops_support",
        description="General-purpose operational assistant focused on actionable insights.",
        prompt_template=OPERATIONS_PROMPT_TEMPLATE,
        skip_patterns=[
            "Confidential",
            "Internal Use Only",
            "All rights reserved",
            "©",
        ],
        keyword_filter_limit=100,
        context_sentence_limit=5,
        neighbor_sentence_window=0,
        keyword_filter_enabled=True,
        keyword_match_strategy="OR",
        keyword_filter_threshold=1,
        default_generation_overrides={"temperature": 0.35, "max_output_tokens": 8192, "top_p": 0.85},  # Increased to prevent truncation
        fast_generation_overrides={"temperature": 0.3, "max_output_tokens": 6000},  # Increased to prevent truncation
        metadata={"audience": "operations"},
    ),
    "fast": RAGProfile(
        name="fast",
        description="Ultra-fast RAG profile optimized for sub-second response times with minimal context.",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        skip_patterns=list(DEFAULT_SKIP_PATTERNS),
        keyword_filter_limit=50,  # Very aggressive filtering for speed
        context_sentence_limit=3,  # Minimal context for fastest responses
        neighbor_sentence_window=0,  # Disabled for speed
        keyword_filter_enabled=True,
        keyword_match_strategy="OR",
        keyword_filter_threshold=1,
        default_generation_overrides={
            "temperature": 0.3,
            "max_output_tokens": 500,  # Very low for fastest generation
            "top_p": 0.9,
            "top_k": 32,
        },
        fast_generation_overrides={
            "temperature": 0.25,
            "max_output_tokens": 400,  # Very low for fastest generation
            "top_p": 0.95,
            "top_k": 24,
        },
        metadata={"optimized_for": "latency", "target_response_time": "<1s"},
    ),
    "ultra_fast": RAGProfile(
        name="ultra_fast",
        description="Ultra-fast RAG profile optimized for millisecond response times with absolute minimal context.",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        skip_patterns=list(DEFAULT_SKIP_PATTERNS),
        keyword_filter_limit=30,  # Extremely aggressive filtering for speed
        context_sentence_limit=2,  # Absolute minimum context (2-3 sentences)
        neighbor_sentence_window=0,  # Disabled for speed
        keyword_filter_enabled=True,
        keyword_match_strategy="OR",
        keyword_filter_threshold=1,
        default_generation_overrides={
            "temperature": 0.3,
            "max_output_tokens": 300,  # Minimal tokens for fastest generation
            "top_p": 0.9,
            "top_k": 24,
        },
        fast_generation_overrides={
            "temperature": 0.25,
            "max_output_tokens": 300,  # Minimal tokens for fastest generation
            "top_p": 0.95,
            "top_k": 20,
        },
        metadata={"optimized_for": "latency", "target_response_time": "<500ms", "use_fast_model_only": True},
    ),
}


def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def _load_profiles_from_path(path: Optional[str]) -> Dict[str, RAGProfile]:
    if not path:
        return {}

    resolved_path = os.path.abspath(path)
    if not os.path.exists(resolved_path):
        return {}

    with open(resolved_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    profiles_data = data.get("profiles", {}) if isinstance(data, dict) else {}
    loaded_profiles: Dict[str, RAGProfile] = {}

    for raw_name, payload in profiles_data.items():
        if not isinstance(payload, dict):
            continue
        name = _normalize_name(raw_name)
        if not name:
            continue

        loaded_profiles[name] = RAGProfile(
            name=name,
            description=payload.get("description", ""),
            prompt_template=payload.get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
            skip_patterns=payload.get("skip_patterns", list(DEFAULT_SKIP_PATTERNS)),
            keyword_filter_limit=payload.get("keyword_filter_limit"),
            context_sentence_limit=payload.get("context_sentence_limit"),
            neighbor_sentence_window=payload.get("neighbor_sentence_window"),
            keyword_filter_enabled=payload.get("keyword_filter_enabled"),
            keyword_match_strategy=payload.get("keyword_match_strategy"),
            keyword_filter_threshold=payload.get("keyword_filter_threshold"),
            default_generation_overrides=payload.get("default_generation_overrides", {}),
            fast_generation_overrides=payload.get("fast_generation_overrides", {}),
            metadata=payload.get("metadata", {}),
        )

    return loaded_profiles


def get_rag_profile(
    requested_name: Optional[str],
    profile_path: Optional[str] = None,
    logger: Optional[Any] = None,
) -> RAGProfile:
    """Return a RAG profile from built-in defaults and optional JSON overrides."""
    normalized_name = _normalize_name(requested_name) or DEFAULT_PROFILE_NAME

    registry = dict(BUILT_IN_PROFILES)
    registry.update(_load_profiles_from_path(profile_path))

    profile = registry.get(normalized_name)
    if profile:
        return profile

    if logger:
        logger.warning(
            "RAG profile '%s' not found; falling back to '%s'",
            normalized_name,
            DEFAULT_PROFILE_NAME,
        )
    return registry[DEFAULT_PROFILE_NAME]


def list_available_profiles(profile_path: Optional[str] = None) -> List[str]:
    """Return all available profile names."""
    combined = {**BUILT_IN_PROFILES, **_load_profiles_from_path(profile_path)}
    return sorted(combined.keys())

