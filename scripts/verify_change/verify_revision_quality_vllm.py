#!/usr/bin/env python3
"""
Verify LLM capability to distinguish between real revisions and LLM-generated fixes.
Uses a locally hosted vLLM endpoint.

This script evaluates whether LLMs can distinguish between:
- "true good" = Camera ready paper (real revisions)
- "fake good" = De-planted error paper (LLM-generated fixes without real substance)

Given a pair of papers (original flawed + revised), the LLM scores how well revisions
were made on a scale of 1-9 (9 is best).

Usage:
    python verify_revision_quality_vllm.py --data_dir ../sampled_data_verify_change/no_appendix \
                                          --vllm_endpoint http://localhost:8000/v1/chat/completions \
                                          --comparison_type true_good_vs_fake_good
"""

import os
import json
import argparse
import time
import re
import threading
import signal
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# OpenReview API
try:
    import openreview
    OPENREVIEW_AVAILABLE = True
except ImportError:
    OPENREVIEW_AVAILABLE = False
    print("WARNING: openreview not installed. Install with: pip install openreview-py")

# Global flag for graceful shutdown
should_exit = threading.Event()
results_csv_lock = threading.Lock()

# Set style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
COLOR_MATCH = "#2ecc71"
COLOR_MISMATCH = "#e74c3c"
COLOR_NEUTRAL = "#95a5a6"
COLOR_WHITE = "#ffffff"

# Category label mapping for plots (short versions)
CATEGORY_LABELS = {
    '1a': 'Baselines',
    '1b': 'Scope',
    '1c': 'Ablation',
    '1d': 'Metrics',
    '2a': 'Design',
    '2b': 'Lacks Theory',
    '2c': 'Math Error',
    '3a': 'Novelty',
    '3b': 'Overclaims',
    '4a': 'Clarity',
    '4b': 'Reproducibility',
    '5a': 'Limitations',
    '5b': 'Ethical',
}

# Category label mapping for plots (full versions)
CATEGORY_LABELS_FULL = {
    '1a': 'Insufficient Baselines/Comparisons',
    '1b': 'Weak or Limited Scope of Experiments',
    '1c': 'Lack of Necessary Ablation or Analysis',
    '1d': 'Flawed Evaluation Metrics or Setup',
    '2a': 'Fundamental Technical Limitation',
    '2b': 'Missing or Incomplete Theoretical Foundation',
    '2c': 'Technical or Mathematical Error',
    '3a': 'Insufficient Novelty / Unacknowledged Prior Work',
    '3b': 'Overstated Claims or Mismatch Between Claim and Evidence',
    '4a': 'Lack of Clarity / Ambiguity',
    '4b': 'Missing Implementation or Methodological Details',
    '5a': 'Unacknowledged Technical Limitations',
    '5b': 'Unaddressed Ethical or Societal Impact',
}

# --- Environment & API Configuration ---
load_dotenv()

# --- Constants ---
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2

# Default vLLM endpoint
DEFAULT_VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"

# --- Pydantic Models ---
class RevisionScore(BaseModel):
    """Pydantic model for revision quality score."""
    score: int = Field(
        description="Revision quality score from 1 to 9, where 9 indicates excellent revisions that substantially address the flaw.",
        ge=1, le=9
    )
    reasoning: str = Field(
        description="Brief explanation (2-3 sentences) of the score, focusing on how well the revisions address the identified flaw."
    )

class DualRevisionScore(BaseModel):
    """Pydantic model for dual revision scores (Quality and Verifiability)."""
    quality_score: int = Field(
        description="Quality score from 1 to 9, assessing how well the revision addresses the flaw.",
        ge=1, le=9
    )
    verifiability_score: int = Field(
        description="Verifiability score from 1 to 9, assessing trustworthiness and reproducibility of the evidence.",
        ge=1, le=9
    )
    explanation: str = Field(
        description="Brief analysis of both scores. If Quality is High but Verifiability is Low (e.g., Q=8, V=2), explicitly warn that the results look suspicious or fabricated."
    )

# --- Prompt Templates ---
def create_verification_prompt(
    original_paper: str,
    revised_paper: str,
    flaw_description: str,
    flaw_location: str = "",
    change_details: List[Dict] = None,
    ablation_name: str = None,
    dual_scores: bool = False,
    rebuttal_text: Optional[str] = None,
    snippets_only: bool = False
) -> str:
    """Create prompt for verifying revision quality."""
    # Ablation: include location if ablation_name is "with_location"
    include_location = (ablation_name == "with_location")
    location_text = f"\n**Flaw Location:** {flaw_location}\n" if (flaw_location and include_location) else ""
    
    # Extract original and revised content from change_details if available
    original_flawed_content = ""
    revised_content = ""
    if change_details and len(change_details) > 0:
        # Use the first change's content
        first_change = change_details[0]
        original_flawed_content = first_change.get('planted_error_content', '')
        revised_content = first_change.get('deplanted_error_content', '')
    
    # Add specific sections that were changed if we have the content
    changed_sections_text = ""
    if original_flawed_content or revised_content:
        changed_sections_text = "\n**Specific Sections That Were Changed:**\n\n"
        if original_flawed_content:
            changed_sections_text += f"Original (flawed) content: {original_flawed_content}\n\n"
        if revised_content:
            changed_sections_text += f"Revised content: {revised_content}\n\n"
    
    # Add rebuttal text if provided
    rebuttal_section = ""
    if rebuttal_text and rebuttal_text.strip():
        rebuttal_section = f"\n**Author's Rebuttal to Reviewers:**\n\n{rebuttal_text}\n\n**Note:** The above rebuttal shows how the authors responded to reviewer concerns. Use this context to better understand the flaw and assess whether the revisions in the paper adequately address the concerns raised.\n\n"
    
    # When snippets_only is True, we only use the changed sections, not the full papers
    if snippets_only:
        if not changed_sections_text:
            raise ValueError("snippets_only mode requires change_details with original and revised content")
        # Don't include full papers in the prompt
        papers_section = ""
    else:
        # Include full papers in the prompt
        papers_section = f"\nOriginal Paper (with flaw): {original_paper}\n\nRevised Paper: {revised_paper}"
    
    # Use dual scores prompt if requested
    if dual_scores:
        prompt = f"""You are evaluating a revised research paper. You must assess the revision on two distinct dimensions:

1. **Quality of Revision:** Does the change logically and substantively address the reviewer's concern?

2. **Verifiability & Integrity:** Is the new evidence supported by sufficient methodological detail, error analysis, and realistic data to be trusted?

**Evaluation Approach:**

Reviewers often encounter two specific failure modes:

* **Lazy Truth:** The author admits the flaw but only promises future fixes or makes vague arguments without data. (Low Quality / High Verifiability)

* **Fabricated Effort:** The author adds impressive-looking tables or claims that lack necessary experimental details, appear statistically "too perfect," or are impossible to reproduce based on the text. (High Claimed Quality / Low Verifiability)

You must verify what is actually present in the revised paper by comparing it directly with the original version.

**Check systematically for:**

1.  **Verifiability Issues:** Fabricated/Unrealistic Results (e.g., no trade-offs, perfect alignment), Vague Experimental Details (missing n-values, hyperparameters), Cherry-Picking.

2.  **Quality Issues:** Low-Effort Changes (e.g., "we will address this"), Lack of Understanding, Poor Integration, Insufficient Depth.

**Quality Score (1-9)**

Does the content of the revision theoretically solve the problem?

* **1:** Flaw ignored. Revisions are completely irrelevant or purely cosmetic.

* **2:** Flaw acknowledged but deferred. Authors promise "future work" or use "will address" instead of doing the work now.

* **3:** Minimal effort. Shallow changes or generic sentences added that do not resolve the core issue.

* **4:** Partially addressed but superficial. Content added misses the point or lacks necessary depth.

* **5:** Moderately addressed. Meaningful improvements made, but noticeable gaps in coverage or analysis remain.

* **6:** Reasonably solid. Most expectations met, though minor details or deeper analysis are missing.

* **7:** Well addressed. Substantive changes clearly target the core issue with relevant evidence.

* **8:** Very well addressed. Strong evidence, thorough explanation, and clear understanding of the flaw.

* **9:** Exemplary. Comprehensive, rigorous, and fully supported response that perfectly integrates into the paper.

**Verifiability Score(1-9)**

Do you trust the evidence provided? Is the "How" described as clearly as the "What"?

* **1:** **Critical Flag.** Results are statistically impossible, contradictory, or completely invented.

* **2:** **Highly Suspicious.** Major claims made with zero methodological details (no n-values, no settings, no source).

* **3:** **Suspicious.** Results look "too perfect" (monotonic improvements everywhere) or lack expected error/variance analysis.

* **4:** Low Verifiability. Evidence is present but vague; it is hard to determine exactly how results were generated.

* **5:** Mixed. Some experimental details provided, but key reproducibility information (splits, hyperparameters) is missing.

* **6:** Passable. General methodology is described, but specific details needed for reproduction are unclear.

* **7:** Trustworthy. Methods are clear enough to understand; results look organic and realistic.

* **8:** High Integrity. Detailed methodology provided, error bars present, and limitations/trade-offs discussed.

* **9:** **Reproducible.** Complete transparency; full metrics, exact settings, and rigorous error analysis provided.

**Assessment Output:**

Provide:

* **Quality Score:** [1-9]

* **Verifiability Score:** [1-9]

* **Explanation:** A brief analysis. If Quality is High but Verifiability is Low (e.g., Q=8, V=2), explicitly warn that the results look suspicious or fabricated.

**Input:**

Identified Flaw: {flaw_description}
{location_text}{changed_sections_text}{rebuttal_section}{papers_section}

Please respond with a valid JSON object containing "quality_score", "verifiability_score", and "explanation" fields.
"""
        return prompt
    
    prompt = f"""You are evaluating how well a revised research paper addresses a specific flaw that was previously identified by reviewers. Your task is to provide a careful, critical, and professional assessment based on the actual content of the papers, not on the authors' claims alone.

**Evaluation Approach:**
Authors may sometimes make revisions that sound plausible but do not fully resolve the underlying issue. You must verify what is actually present in the revised paper by comparing it directly with the original version.

**Check systematically for:**

1. Fabricated or Unrealistic Results

2. Incomplete or Vague Experimental Details

3. Cherry-Picking and Selective Reporting

4. Over-Exaggeration and Unsubstantiated Claims

5. Low-Effort and Superficial Changes

6. Lack of Genuine Understanding

7. Insufficient Depth and Rigor

8. Poor Integration and Coherence

9. Discrepancies between Claims and Evidence

**Scoring (1-9):**

Assign a score from 1 to 9 based on how well the revisions address the flaw:

• 1: Invalid or Ignored. The flaw is ignored, or results are statistically implausible. No credible evidence is provided.
• 2: Methodologically Void. Claims lack supporting data (e.g., no n-values). Resolution is deferred to future work.
• 3: Superficial or Anomalous. Revisions are cosmetic. Results lack expected variance or error analysis, indicating reporting anomalies.
• 4: Ambiguous Verification. Evidence is present but vague; the generation of results is obscured. Content lacks depth.
• 5: Partially Reproducible. Experimental context is provided, but critical reproducibility parameters (e.g., splits, hyperparameters) are omitted.
• 6: Generally Valid. Methodology is described in broad terms. Expectations are met, though specific details for reproduction are absent.
• 7: Verifiable and Substantive. Methods are transparent and results appear organic. Changes substantively resolve the flaw.
• 8: Rigorous and Transparent. Detailed methodology is provided, including error bars and limitation analysis. Evidence strongly supports the resolution.
• 9: Fully Reproducible and Comprehensive. Complete transparency is achieved (exact metrics, settings, data). The response fully resolves the issue.

**Assessment Output:**

Provide:

- A numerical revision score from 1 to 9.

- A clear explanation (in a few sentences) identifying problems, commenting on depth/rigor, and summarizing if the fix is substantive or superficial.

**Input:**

Identified Flaw: {flaw_description}

{location_text}{changed_sections_text}{rebuttal_section}{papers_section}

Please respond with a valid JSON object containing "score" and "reasoning" fields.
"""
    
    return prompt

# --- OpenReview API Functions ---
# Default venue ID for NeurIPS 2024 (can be overridden)
DEFAULT_VENUE_ID = "NeurIPS.cc/2024/Conference"

# Cache file for rebuttals (stored in output directory)
REBUTTAL_CACHE_FILENAME = "rebuttals_cache.json"

def collect_all_rebuttals_recursive(reply, reply_type="Rebuttal", all_rebuttals=None):
    """
    Recursively collect all rebuttals from a reply and its nested replies.
    
    Args:
        reply: A reply note (dict or note object)
        reply_type: Type of reply to look for (default: "Rebuttal")
        all_rebuttals: List to accumulate rebuttals (for recursion)
    
    Returns:
        List of rebuttal texts
    """
    if all_rebuttals is None:
        all_rebuttals = []
    
    # Handle both dict and note object formats
    if isinstance(reply, dict):
        invitations = reply.get('invitations', [])
        if not invitations:
            inv = reply.get('invitation', '')
            if inv:
                invitations = [inv] if isinstance(inv, str) else inv
        details = reply.get('details', {})
    else:
        invitations = getattr(reply, 'invitations', [])
        if not invitations:
            inv = getattr(reply, 'invitation', '')
            if inv:
                invitations = [inv] if isinstance(inv, str) else inv
        details = getattr(reply, 'details', {})
    
    # Check if this reply is a rebuttal
    is_rebuttal = any(
        isinstance(inv, str) and inv.endswith(reply_type)
        for inv in invitations
    )
    
    if is_rebuttal:
        rebuttal_text = extract_rebuttal_text(reply)
        if rebuttal_text:
            all_rebuttals.append(rebuttal_text)
    
    # Recursively check nested replies
    nested_replies = []
    if isinstance(details, dict):
        nested_replies = details.get('replies', [])
    elif hasattr(details, 'get'):
        nested_replies = details.get('replies', [])
    elif hasattr(details, 'replies'):
        nested_replies = details.replies if details.replies else []
    
    for nested_reply in nested_replies:
        collect_all_rebuttals_recursive(nested_reply, reply_type, all_rebuttals)
    
    return all_rebuttals

def extract_rebuttal_text(rebuttal_note) -> Optional[str]:
    """
    Extract rebuttal text from a rebuttal note object or dict.
    
    Args:
        rebuttal_note: OpenReview note object or dict
        
    Returns:
        Rebuttal text as string, or None if not found
    """
    # Handle both dict and note object formats
    if isinstance(rebuttal_note, dict):
        content = rebuttal_note.get('content', {})
    else:
        content = getattr(rebuttal_note, 'content', {})
    
    if isinstance(content, dict):
        # Try common field names
        for field in ['rebuttal', 'text', 'reply', 'response']:
            if field in content:
                value = content[field]
                if isinstance(value, dict):
                    value = value.get('value', '')
                if isinstance(value, str) and value.strip():
                    return value.strip()
        
        # If no specific field, try to get all text content
        all_text = []
        for key, value in content.items():
            if isinstance(value, str) and len(value) > 50:
                all_text.append(value)
            elif isinstance(value, dict) and 'value' in value:
                val = value['value']
                if isinstance(val, str) and len(val) > 50:
                    all_text.append(val)
        if all_text:
            return '\n\n'.join(all_text)
    elif isinstance(content, str):
        return content.strip() if content.strip() else None
    
    return None

def load_rebuttals_cache(cache_path: Path) -> Dict[str, Optional[str]]:
    """
    Load rebuttals from cache file.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Dictionary mapping paper_id -> rebuttal text (or None), or empty dict if cache doesn't exist
    """
    if not cache_path.exists():
        return {}
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            # Cache format: {"venue_id": "...", "rebuttals": {...}}
            return cache_data.get('rebuttals', {})
    except Exception as e:
        print(f"⚠️ Warning: Could not load rebuttals cache from {cache_path}: {e}", flush=True)
        return {}

def save_rebuttals_cache(cache_path: Path, rebuttals: Dict[str, Optional[str]], venue_id: str):
    """
    Save rebuttals to cache file.
    
    Args:
        cache_path: Path to cache file
        rebuttals: Dictionary mapping paper_id -> rebuttal text (or None)
        venue_id: Venue ID used for fetching
    """
    try:
        cache_data = {
            'venue_id': venue_id,
            'rebuttals': rebuttals,
            'cached_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved rebuttals cache to {cache_path}", flush=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not save rebuttals cache to {cache_path}: {e}", flush=True)

def fetch_rebuttals_batch(
    paper_ids: List[str], 
    venue_id: str = DEFAULT_VENUE_ID,
    cache_path: Optional[Path] = None,
    overwrite_cache: bool = False
) -> Dict[str, Optional[str]]:
    """
    Fetch rebuttals for multiple papers using get_all_notes (efficient bulk fetch).
    Uses cache to avoid redundant API calls.
    
    Args:
        paper_ids: List of OpenReview paper IDs (forum IDs)
        venue_id: OpenReview venue ID (e.g., "NeurIPS.cc/2024/Conference")
        cache_path: Optional path to cache file. If provided, will load from/save to cache.
        
    Returns:
        Dictionary mapping paper_id -> rebuttal text (or None)
    """
    if not OPENREVIEW_AVAILABLE:
        print("⚠️ WARNING: openreview library not available. Install with: pip install openreview-py")
        return {pid: None for pid in paper_ids}
    
    # Initialize result dictionary
    rebuttals = {pid: None for pid in paper_ids}
    
    # Try to load from cache if cache path is provided (unless overwriting)
    cached_rebuttals = {}
    if cache_path and not overwrite_cache:
        if cache_path.exists():
            print(f"Checking cache at {cache_path}...", flush=True)
            cached_rebuttals = load_rebuttals_cache(cache_path)
            # Check if cached venue_id matches
            if cached_rebuttals:
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        cached_venue_id = cache_data.get('venue_id', '')
                        if cached_venue_id == venue_id:
                            # Use cached rebuttals for papers that are in cache
                            for pid in paper_ids:
                                if pid in cached_rebuttals:
                                    rebuttals[pid] = cached_rebuttals[pid]
                            cached_count = sum(1 for r in rebuttals.values() if r is not None)
                            print(f"✅ Loaded {cached_count} rebuttals from cache ({len(cached_rebuttals)} total in cache)", flush=True)
                        else:
                            print(f"⚠️ Cache venue_id mismatch (cached: {cached_venue_id}, requested: {venue_id}). Fetching fresh data...", flush=True)
                            cached_rebuttals = {}
                except Exception as e:
                    print(f"⚠️ Warning: Could not read cache metadata: {e}", flush=True)
                    cached_rebuttals = {}
            else:
                print(f"⚠️ Cache file exists but is empty or invalid", flush=True)
        else:
            print(f"Cache file not found at {cache_path}, will create new cache", flush=True)
    elif overwrite_cache:
        print(f"⚠️ Overwrite cache flag enabled, ignoring existing cache and fetching fresh data...", flush=True)
    
    # Find papers that need to be fetched (not in cache or None in cache, or if overwriting)
    if overwrite_cache:
        papers_to_fetch = list(paper_ids)
    else:
        papers_to_fetch = [pid for pid in paper_ids if pid not in cached_rebuttals or cached_rebuttals[pid] is None]
    
    if not papers_to_fetch:
        print(f"✅ All {len(paper_ids)} rebuttals found in cache, skipping API call", flush=True)
        return rebuttals
    
    print(f"Fetching {len(papers_to_fetch)} rebuttals from API ({(len(papers_to_fetch)/len(paper_ids)*100):.1f}% not in cache)...", flush=True)
    
    # Initialize OpenReview client
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    
    try:
        print(f"Fetching all submissions with replies from {venue_id}...", flush=True)
        
        # Get all submissions with replies in one call
        try:
            submissions = client.get_all_notes(
                invitation=f'{venue_id}/-/Submission',
                details='replies'
            )
        except (KeyError, AttributeError) as e:
            # Handle case where API response doesn't have expected format
            print(f"⚠️ API response format issue ({type(e).__name__}: {e}). Trying alternative approach...", flush=True)
            # Try using get_notes directly with limit
            try:
                submissions = []
                offset = 0
                limit = 1000
                while True:
                    batch = client.get_notes(
                        invitation=f'{venue_id}/-/Submission',
                        details='replies',
                        offset=offset,
                        limit=limit
                    )
                    if not batch:
                        break
                    submissions.extend(batch)
                    if len(batch) < limit:
                        break
                    offset += limit
                print(f"✅ Fetched {len(submissions)} submissions using alternative method", flush=True)
            except Exception as e2:
                print(f"⚠️ Alternative method also failed: {e2}", flush=True)
                raise e  # Re-raise original error
        
        print(f"✅ Fetched {len(submissions)} submissions", flush=True)
        
        # Filter for rebuttals and map to forum IDs
        reply_type = "Rebuttal"
        rebuttal_count = 0
        newly_fetched = 0
        
        for submission in submissions:
            forum_id = submission.forum if hasattr(submission, 'forum') else submission.id
            
            # Skip if this submission is not in our papers_to_fetch list
            if forum_id not in papers_to_fetch:
                continue
            
            # Get replies from submission details
            if not hasattr(submission, 'details') or not submission.details:
                continue
            
            replies = submission.details.get('replies', [])
            if not replies:
                continue
            
            # Filter for rebuttals (recursively to get all nested rebuttals)
            submission_rebuttals = []
            for reply in replies:
                # Recursively collect all rebuttals from this reply and its nested replies
                rebuttals_from_reply = collect_all_rebuttals_recursive(reply, reply_type)
                submission_rebuttals.extend(rebuttals_from_reply)
                rebuttal_count += len(rebuttals_from_reply)
            
            # Combine all rebuttals for this submission
            if submission_rebuttals:
                rebuttals[forum_id] = '\n\n---\n\n'.join(submission_rebuttals)
                newly_fetched += 1
        
        print(f"✅ Found {rebuttal_count} rebuttals across {newly_fetched} newly fetched papers", flush=True)
        
        # Merge with cached rebuttals and save to cache
        if cache_path:
            # Update cache with newly fetched rebuttals
            all_rebuttals = {**cached_rebuttals, **rebuttals}
            save_rebuttals_cache(cache_path, all_rebuttals, venue_id)
        
    except Exception as e:
        print(f"⚠️ Error fetching rebuttals: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # If we have cached rebuttals, return those even if API call failed
        if cached_rebuttals:
            print(f"⚠️ Using cached rebuttals due to API error", flush=True)
            # Merge cached rebuttals with any newly fetched ones
            for pid in paper_ids:
                if pid in cached_rebuttals and (pid not in rebuttals or rebuttals[pid] is None):
                    rebuttals[pid] = cached_rebuttals[pid]
    
    return rebuttals

# --- API Call Functions ---
def call_vllm_with_retries(
    endpoint: str,
    prompt: str,
    response_model: BaseModel,
    max_retries: int = MAX_RETRIES,
    temperature: float = 0.0,
    timeout: int = 300
) -> Optional[BaseModel]:
    """Call vLLM endpoint with retries and structured output."""
    
    # Get JSON schema from Pydantic model
    schema = response_model.model_json_schema()
    
    # Create system message to instruct JSON output
    system_message = f"""You are a helpful assistant that responds in JSON format. 
You must respond with a valid JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Ensure your response is valid JSON and matches the schema exactly."""
    
    for attempt in range(max_retries):
        try:
            # Prepare request payload (OpenAI-compatible format)
            payload = {
                "model": "vllm",  # Model name doesn't matter for vLLM
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"}  # Request JSON mode
            }
            
            # Make HTTP request
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            # Clean up JSON response (remove markdown code blocks if present)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON and validate with Pydantic
            response_dict = json.loads(content)
            return response_model(**response_dict)
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                tqdm.write(f"  ⚠️ Request error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                tqdm.write(f"  ⚠️ Error after {max_retries} attempts: {e}")
                return None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt < max_retries - 1:
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                tqdm.write(f"  ⚠️ JSON parsing error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                tqdm.write(f"  ⚠️ Error parsing response after {max_retries} attempts: {e}")
                tqdm.write(f"  Response content: {content[:500] if 'content' in locals() else 'N/A'}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                tqdm.write(f"  ⚠️ Unexpected error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                tqdm.write(f"  ⚠️ Error after {max_retries} attempts: {e}")
                return None
    
    return None

# --- File Reading Functions ---
def remove_tables_from_markdown(text: str) -> str:
    """
    Remove all tables (both HTML and markdown) from the text.
    
    Tables are identified by:
    1. HTML tables: <table>...</table> blocks (including nested divs)
    2. Markdown tables: Lines starting with | and containing multiple | characters
       - Separator lines with |, dashes (-), and optionally colons (:)
       - Subsequent table rows until a non-table line is encountered
    """
    if not text:
        return text
    
    # First, remove HTML tables using regex
    # Match <div>...</div> blocks containing <table>...</table>
    # This handles cases like <div id="tab:main0" markdown="1">\n<table>...</table>\n</div>
    html_table_pattern = r'<div[^>]*>.*?<table[^>]*>.*?</table>.*?</div>'
    text = re.sub(html_table_pattern, '', text, flags=re.DOTALL)
    
    # Also handle standalone <table>...</table> blocks (without div wrapper)
    standalone_table_pattern = r'<table[^>]*>.*?</table>'
    text = re.sub(standalone_table_pattern, '', text, flags=re.DOTALL)
    
    # Now remove markdown-style tables
    lines = text.split('\n')
    result_lines = []
    in_table = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if this line is a table row (starts with | and has multiple |)
        is_table_row = stripped.startswith('|') and stripped.count('|') >= 2
        
        # Check if this line is a table separator (contains |, -, and optionally :)
        is_separator = False
        if stripped.startswith('|'):
            # Check for separator pattern: |---| or |:---| or |---:| or |:---:|
            parts = [p.strip() for p in stripped.split('|') if p.strip()]
            if parts:
                # Check if all parts are dashes with optional colons
                is_separator = all(
                    re.match(r'^:?-+:?$', part) for part in parts
                )
        
        if is_table_row or is_separator:
            # We're in a table - skip this line
            in_table = True
            continue
        else:
            # Not a table line
            if in_table:
                # We just exited a table, continue normally
                in_table = False
            result_lines.append(line)
    
    return '\n'.join(result_lines)

def read_paper_markdown(paper_path: Path, remove_tables: bool = False) -> Optional[str]:
    """Read paper markdown file from various possible locations."""
    # Try structured_paper_output/paper.md first (for latest/)
    structured_path = paper_path / "structured_paper_output" / "paper.md"
    if structured_path.exists():
        try:
            with open(structured_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if remove_tables:
                    content = remove_tables_from_markdown(content)
                return content
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading {structured_path}: {e}")
    
    # Try paper.md directly
    paper_md = paper_path / "paper.md"
    if paper_md.exists():
        try:
            with open(paper_md, 'r', encoding='utf-8') as f:
                content = f.read()
                if remove_tables:
                    content = remove_tables_from_markdown(content)
                return content
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading {paper_md}: {e}")
    
    # If paper_path is already a .md file, read it directly
    if paper_path.is_file() and paper_path.suffix == '.md':
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if remove_tables:
                    content = remove_tables_from_markdown(content)
                return content
        except Exception as e:
            tqdm.write(f"  ⚠️ Error reading {paper_path}: {e}")
    
    return None

def read_modifications_summary(csv_path: Path) -> Optional[Dict]:
    """Read modifications summary CSV and extract flaw information."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        
        # Get first row (assuming one flaw per paper for now)
        row = df.iloc[0]
        
        # Parse modifications JSON
        modifications = json.loads(row['llm_generated_modifications'])
        
        # Extract target headings and locations
        locations = []
        for mod in modifications:
            heading = mod.get('target_heading', '')
            if heading:
                locations.append(heading)
        
        location_text = "; ".join(locations) if locations else ""
        
        return {
            'flaw_id': row['flaw_id'],
            'flaw_description': row['flaw_description'],
            'flaw_location': location_text,
            'num_modifications': row['num_modifications']
        }
    except Exception as e:
        tqdm.write(f"  ⚠️ Error reading modifications summary: {e}")
        return None

def read_fix_summary(csv_path: Path) -> Optional[Dict]:
    """Read fix_summary CSV from de-planted_error folder and extract detailed change locations."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        
        # Get first row (assuming one flaw per paper for now)
        row = df.iloc[0]
        
        # Parse modifications JSON
        modifications = json.loads(row['llm_generated_modifications'])
        
        # Extract detailed change information
        change_details = []
        for mod in modifications:
            heading = mod.get('target_heading', '')
            planted_content = mod.get('planted_error_content', '')
            deplanted_content = mod.get('deplanted_error_content', '')
            explanation = mod.get('explanation', '')
            success = mod.get('success', False)
            
            if heading:
                change_details.append({
                    'target_heading': heading,
                    'planted_error_content': planted_content[:1000] if planted_content else '',  # Limit length
                    'deplanted_error_content': deplanted_content[:1000] if deplanted_content else '',  # Limit length
                    'explanation': explanation,
                    'success': success
                })
        
        return {
            'flaw_id': row['flaw_id'],
            'flaw_description': row['flaw_description'],
            'change_details': change_details,
            'num_modifications': row['num_modifications']
        }
    except Exception as e:
        tqdm.write(f"  ⚠️ Error reading fix summary: {e}")
        return None

# --- Scoring Functions ---
def score_revision_quality(
    original_paper: str,
    revised_paper: str,
    flaw_description: str,
    flaw_location: str,
    endpoint: str,
    model_name: str = "vllm",
    change_details: List[Dict] = None,
    ablation_name: str = None,
    dual_scores: bool = False,
    rebuttal_text: Optional[str] = None,
    snippets_only: bool = False,
    temperature: float = 0.0
) -> Optional:
    """Score the quality of revisions using vLLM endpoint."""
    prompt = create_verification_prompt(
        original_paper=original_paper,
        revised_paper=revised_paper,
        flaw_description=flaw_description,
        flaw_location=flaw_location,
        change_details=change_details,
        ablation_name=ablation_name,
        dual_scores=dual_scores,
        rebuttal_text=rebuttal_text,
        snippets_only=snippets_only
    )
    
    response_model = DualRevisionScore if dual_scores else RevisionScore
    
    response = call_vllm_with_retries(
        endpoint=endpoint,
        prompt=prompt,
        response_model=response_model,
        max_retries=MAX_RETRIES,
        temperature=temperature
    )
    
    return response

def process_paper_pair(
    category: str,
    paper_folder: str,
    flaw_id: str,
    original_paper_path: Path,
    revised_paper_path: Path,
    flaw_info: Dict,
    output_dir: Path,
    task_idx: int,
    model_name: str,
    endpoint: str,
    change_details: List[Dict] = None,
    ablation_name: str = None,
    dual_scores: bool = False,
    rebuttal_text: Optional[str] = None,
    snippets_only: bool = False,
    remove_tables: bool = False,
    temperature: float = 0.0
) -> Optional[Dict]:
    """Process a single paper pair and score revision quality."""
    try:
        # When snippets_only is True, we don't need to read full papers
        # But we still need placeholder strings for the function signature
        if snippets_only:
            if not change_details:
                tqdm.write(f"  ⚠️ Skipping {paper_folder}/{flaw_id}: snippets_only requires change_details")
                return None
            # Use empty strings as placeholders since we won't use them in the prompt
            original_paper = ""
            revised_paper = ""
        else:
            # Read papers (with optional table removal)
            original_paper = read_paper_markdown(original_paper_path, remove_tables=remove_tables)
            revised_paper = read_paper_markdown(revised_paper_path, remove_tables=remove_tables)
            
            if not original_paper or not revised_paper:
                return None
        
        # Score revision quality
        score_result = score_revision_quality(
            original_paper=original_paper,
            revised_paper=revised_paper,
            flaw_description=flaw_info['flaw_description'],
            flaw_location=flaw_info['flaw_location'],
            endpoint=endpoint,
            model_name=model_name,
            change_details=change_details,
            ablation_name=ablation_name,
            dual_scores=dual_scores,
            rebuttal_text=rebuttal_text,
            snippets_only=snippets_only,
            temperature=temperature
        )
        
        if not score_result:
            return None
        
        result = {
            'category': category,
            'paper_folder': paper_folder,
            'flaw_id': flaw_id,
            'flaw_description': flaw_info['flaw_description'],
        }
        
        if dual_scores:
            # Dual scores: quality_score, verifiability_score, explanation
            result.update({
                'quality_score': score_result.quality_score,
                'verifiability_score': score_result.verifiability_score,
                'explanation': score_result.explanation
            })
        else:
            # Single score: score, reasoning
            result.update({
            'score': score_result.score,
            'reasoning': score_result.reasoning
            })
        
        return result
        
    except Exception as e:
        tqdm.write(f"  ❌ Error processing {paper_folder}/{flaw_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Statistics and Plotting Functions ---
# Note: These functions are identical to verify_revision_quality.py
# They are included here to keep this script standalone

# Import the statistics and plotting functions from the original script
# Since they don't depend on Gemini API, we can reuse them
import sys
from pathlib import Path as PathLib

# Add the parent directory to path to import shared functions
_original_script_path = PathLib(__file__).parent / "verify_revision_quality.py"
if _original_script_path.exists():
    # Import shared functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_revision_quality", _original_script_path)
    verify_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_module)
    
    # Import the functions we need
    compute_treatment_effect = verify_module.compute_treatment_effect
    detect_lazy_authors = verify_module.detect_lazy_authors
    _create_cohen_d_plot = verify_module._create_cohen_d_plot
    create_plots = verify_module.create_plots
    save_result_incrementally = verify_module.save_result_incrementally
    signal_handler = verify_module.signal_handler
else:
    raise ImportError(f"Could not find verify_revision_quality.py at {_original_script_path}")

# --- Main Processing Functions ---
# Note: process_category needs to be modified to use endpoint instead of API keys



def process_category(
    data_dir: Path,
    category: str,
    model_name: str,
    comparison_type: str,
    output_dir: Path,
    endpoint: str,
    max_workers: int = 5,
    processed_keys: Optional[set] = None,
    results_csv: Optional[Path] = None,
    ablation_name: str = None,
    dual_scores: bool = False,
    use_v1_as_flawed: bool = False,
    rebuttals: Optional[Dict[str, Optional[str]]] = None,
    snippets_only: bool = False,
    remove_tables: bool = False,
    temperature: float = 0.0,
) -> List[Dict]:
    """Process all papers in a category."""
    category_path = data_dir / "NeurIPS2024" / category
    
    if not category_path.exists():
        print(f"⚠️ Category {category} not found at {category_path}")
        return []
    
    # Determine which folders to compare
    if comparison_type == "true_good_vs_fake_good":
        if use_v1_as_flawed:
            original_folder = "v1"
            true_good_folder = "latest"
            fake_good_folder = "de-planted_error"
        else:
            original_folder = "planted_error"
            true_good_folder = "latest"
            fake_good_folder = "de-planted_error"
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")
    
    # For reading CSV files, always use planted_error folder (where modifications_summary.csv is stored)
    csv_source_folder = "planted_error"
    csv_source_dir = category_path / csv_source_folder
    
    # For original paper, use v1 or planted_error based on flag
    original_dir = category_path / original_folder
    if not original_dir.exists():
        print(f"⚠️ {original_folder} folder not found for category {category} at {original_dir}")
        return []
    
    if not csv_source_dir.exists():
        print(f"⚠️ {csv_source_folder} folder not found for category {category} at {csv_source_dir} (needed for CSV files)")
        return []
    
    results = []
    
    # Find all paper folders from CSV source (planted_error) to get the list of papers
    csv_paper_folders = [d for d in csv_source_dir.iterdir() if d.is_dir()]
    print(f"Found {len(csv_paper_folders)} paper folders in {csv_source_folder} (for CSV files)", flush=True)
    
    if len(csv_paper_folders) == 0:
        print(f"  ⚠️ No paper folders found in {csv_source_dir}", flush=True)
        return []
    
    # Debug counters
    papers_with_csv = 0
    papers_with_original = 0
    papers_with_true_good = 0
    papers_with_fake_good = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for csv_paper_folder in csv_paper_folders:
            paper_folder_name = csv_paper_folder.name
            
            # Read modifications summary from CSV source folder (planted_error)
            # Pattern 1: {paper_folder_name}_modifications_summary.csv
            csv_path = csv_paper_folder / f"{paper_folder_name}_modifications_summary.csv"
            if not csv_path.exists():
                # Pattern 2: Extract base name (before first underscore + numbers) and try
                # e.g., xjyU6zmZD7_2401_04486 -> xjyU6zmZD7
                base_name = paper_folder_name.split('_')[0] if '_' in paper_folder_name else paper_folder_name
                csv_path = csv_paper_folder / f"{base_name}_modifications_summary.csv"
                if not csv_path.exists():
                    # Pattern 3: Find any CSV file matching the pattern
                    csv_files = list(csv_paper_folder.glob("*_modifications_summary.csv"))
                    if csv_files:
                        csv_path = csv_files[0]
                    else:
                        print(f"  ⚠️ Skipping {paper_folder_name}: No modifications_summary.csv found in {csv_paper_folder}", flush=True)
                        continue
            
            # Verify csv_path exists before reading
            if not csv_path.exists():
                print(f"  ⚠️ Skipping {paper_folder_name}: CSV file not found at {csv_path}", flush=True)
                continue
            
            flaw_info = read_modifications_summary(csv_path)
            if not flaw_info:
                print(f"  ⚠️ Skipping {paper_folder_name}: Could not read flaw info from {csv_path}", flush=True)
                continue
            
            papers_with_csv += 1
            flaw_id = flaw_info['flaw_id']
            
            # Extract paper ID for rebuttal lookup
            paper_id = paper_folder_name.split('_')[0] if '_' in paper_folder_name else paper_folder_name
            rebuttal_text = rebuttals.get(paper_id) if rebuttals else None
            
            # Read detailed change locations if ablation_name contains "with_location" or snippets_only is True
            change_details = None
            if (ablation_name and "with_location" in ablation_name) or snippets_only:
                # Try to read fix_summary CSV from de-planted_error folder
                fake_good_base = category_path / fake_good_folder / paper_folder_name
                base_name = paper_folder_name.split('_')[0] if '_' in paper_folder_name else paper_folder_name
                fix_summary_path = fake_good_base / f"{base_name}_fix_summary.csv"
                
                if fix_summary_path.exists():
                    fix_summary_info = read_fix_summary(fix_summary_path)
                    if fix_summary_info and fix_summary_info.get('change_details'):
                        change_details = fix_summary_info['change_details']
                
                # If snippets_only is True but change_details is still None, skip this paper
                if snippets_only and not change_details:
                    print(f"  ⚠️ Skipping {paper_folder_name}: snippets_only requires change_details but fix_summary not found at {fix_summary_path}", flush=True)
                    continue
            
            # Get original paper path - different structure for v1 vs planted_error
            original_paper_folder = original_dir / paper_folder_name
            if not original_paper_folder.exists():
                # Skip if original paper folder doesn't exist
                print(f"  ⚠️ Skipping {paper_folder_name}: Original paper folder not found at {original_paper_folder}", flush=True)
                continue
            
            if use_v1_as_flawed:
                # v1 structure: v1/{paper_folder}/structured_paper_output/paper.md
                original_path = original_paper_folder / "structured_paper_output" / "paper.md"
                if not original_path.exists():
                    # Try to find any .md file in structured_paper_output
                    structured_dir = original_paper_folder / "structured_paper_output"
                    if structured_dir.exists():
                        md_files = list(structured_dir.glob("*.md"))
                        if md_files:
                            original_path = md_files[0]
                        else:
                            continue
                    else:
                        continue
            else:
                # planted_error structure: planted_error/{paper_folder}/flawed_papers/{flaw_id}.md
                original_path = original_paper_folder / "flawed_papers" / f"{flaw_id}.md"
                if not original_path.exists():
                    # Try to find any .md file
                    flawed_papers_dir = original_paper_folder / "flawed_papers"
                    if flawed_papers_dir.exists():
                        md_files = list(flawed_papers_dir.glob("*.md"))
                        if md_files:
                            original_path = md_files[0]
                        else:
                            continue
                    else:
                        continue
            
            # Verify original_path exists and is readable
            if not read_paper_markdown(original_path):
                print(f"  ⚠️ Skipping {paper_folder_name}: Original paper markdown not readable at {original_path}", flush=True)
                continue
            
            papers_with_original += 1
            
            # Process true good (camera ready)
            true_good_base = category_path / true_good_folder / paper_folder_name
            if not true_good_base.exists():
                print(f"  ⚠️ Skipping {paper_folder_name}: True good folder not found at {true_good_base}", flush=True)
                continue
            
            # Check if structured_paper_output exists (read_paper_markdown will look for paper.md inside it)
            true_good_structured = true_good_base / "structured_paper_output"
            
            if true_good_structured.exists():
                papers_with_true_good += 1
                # Pass the paper folder, not the structured_paper_output subdirectory
                true_good_path = true_good_base
                key_true = (category, paper_folder_name, flaw_id, 'true_good')
                if not processed_keys or key_true not in processed_keys:
                    future = executor.submit(
                    process_paper_pair,
                    category=category,
                    paper_folder=paper_folder_name,
                    flaw_id=flaw_id,
                    original_paper_path=original_path,
                    revised_paper_path=true_good_path,
                    flaw_info=flaw_info,
                    output_dir=output_dir,
                    task_idx=len(futures),
                    model_name=model_name,
                            temperature=temperature,
                            change_details=change_details,
                            endpoint=endpoint,
                            ablation_name=ablation_name,
                            dual_scores=dual_scores,
                            rebuttal_text=rebuttal_text,
                            snippets_only=snippets_only,
                            remove_tables=remove_tables
                    )
                    future.revision_type = 'true_good'
                    future.paper_folder_name = paper_folder_name
                    future.flaw_id = flaw_id
                    futures.append(future)
            
            # Process fake good (de-planted error)
            fake_good_base = category_path / fake_good_folder / paper_folder_name
            fake_good_path = fake_good_base / "flawed_papers" / f"{flaw_id}.md"
            
            if fake_good_path.exists():
                papers_with_fake_good += 1
                key_fake = (category, paper_folder_name, flaw_id, 'fake_good')
                if not processed_keys or key_fake not in processed_keys:
                    future = executor.submit(
                    process_paper_pair,
                    category=category,
                    paper_folder=paper_folder_name,
                    flaw_id=flaw_id,
                    original_paper_path=original_path,
                    revised_paper_path=fake_good_path,
                    flaw_info=flaw_info,
                    output_dir=output_dir,
                    task_idx=len(futures),
                    model_name=model_name,
                            temperature=temperature,
                            change_details=change_details,
                            endpoint=endpoint,
                            ablation_name=ablation_name,
                            dual_scores=dual_scores,
                            rebuttal_text=None,  # Don't use rebuttal for fake_good (LLM-generated fix)
                            snippets_only=snippets_only,
                            remove_tables=remove_tables
                    )
                    future.revision_type = 'fake_good'
                    future.paper_folder_name = paper_folder_name
                    future.flaw_id = flaw_id
                    futures.append(future)
            else:
                # Try to find any .md file
                fake_flawed_papers_dir = fake_good_base / "flawed_papers"
                if fake_flawed_papers_dir.exists():
                    md_files = list(fake_flawed_papers_dir.glob("*.md"))
                    if md_files:
                        papers_with_fake_good += 1
                        fake_good_path = md_files[0]
                        key_fake = (category, paper_folder_name, flaw_id, 'fake_good')
                        if not processed_keys or key_fake not in processed_keys:
                            future = executor.submit(
                                process_paper_pair,
                                category=category,
                                paper_folder=paper_folder_name,
                                flaw_id=flaw_id,
                                original_paper_path=original_path,
                                revised_paper_path=fake_good_path,
                                flaw_info=flaw_info,
                                output_dir=output_dir,
                                task_idx=len(futures),
                                model_name=model_name,
                                temperature=temperature,
                                change_details=change_details,
                                endpoint=endpoint,
                                ablation_name=ablation_name,
                                dual_scores=dual_scores,
                                rebuttal_text=None,  # Don't use rebuttal for fake_good (LLM-generated fix)
                                snippets_only=snippets_only,
                                remove_tables=remove_tables
                            )
                            future.revision_type = 'fake_good'
                            future.paper_folder_name = paper_folder_name
                            future.flaw_id = flaw_id
                            futures.append(future)
        
        print(f"  Submitted {len(futures)} tasks for processing", flush=True)
        print(f"  Debug: {papers_with_csv} papers with CSV, {papers_with_original} with original in {original_folder}, "
              f"{papers_with_true_good} with true_good in {true_good_folder}, {papers_with_fake_good} with fake_good in {fake_good_folder}", flush=True)
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {category}"):
            # Check if we should exit gracefully
            if should_exit.is_set():
                tqdm.write(f"  ⚠️ Exiting gracefully due to interrupt signal...")
                # Cancel remaining futures if possible
                for f in futures:
                    f.cancel()
                break
            
            result = future.result()
            if result:
                result['revision_type'] = future.revision_type
                results.append(result)
                
                # Save incrementally to CSV if provided
                if results_csv is not None:
                    try:
                        save_result_incrementally(result, results_csv)
                    except Exception as e:
                        tqdm.write(f"  ⚠️ Warning: Could not save result incrementally: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Verify LLM capability to distinguish between real revisions and LLM-generated fixes (vLLM endpoint version)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing NeurIPS2024 data"
    )
    parser.add_argument(
        "--vllm_endpoint",
        type=str,
        default=DEFAULT_VLLM_ENDPOINT,
        help=f"vLLM endpoint URL (default: {DEFAULT_VLLM_ENDPOINT})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vllm",
        help="Model name for output directory naming (default: vllm)"
    )
    parser.add_argument(
        "--comparison_type",
        type=str,
        default="true_good_vs_fake_good",
        choices=["true_good_vs_fake_good"],
        help="Type of comparison to perform"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories to process (default: all found in data_dir)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data_dir/llm_verification_{model_name})"
    )
    parser.add_argument(
        "--detect_lazy_authors",
        action="store_true",
        help="Detect cases where camera-ready version might not actually fix the error (true_good score < 5)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, reuse existing verification_scores.csv and only process missing paper/revision pairs"
    )
    parser.add_argument(
        "--ablation_name",
        type=str,
        default=None,
        help="Name of ablation study (e.g., 'no_location'). Output folder will be llm_verification_{model_name}_{ablation_name}"
    )
    parser.add_argument(
        "--dual_scores",
        action="store_true",
        help="Use dual scores mode (Quality and Verifiability). Uses DualRevisionScore model with separate scores."
    )
    parser.add_argument(
        "--use_v1_as_flawed",
        action="store_true",
        help="Use v1/ folder as the flawed paper instead of planted_error/. Compares v1 vs latest (true_good) and v1 vs de-planted_error (fake_good)."
    )
    parser.add_argument(
        "--include_rebuttals",
        action="store_true",
        help="Include author rebuttals from OpenReview API v2 in the verification prompt. Fetches all rebuttals at the beginning of the run."
    )
    parser.add_argument(
        "--overwrite_rebuttals_cache",
        action="store_true",
        help="Overwrite existing rebuttals cache and fetch fresh data from OpenReview API (recursively collects all nested rebuttals)."
    )
    parser.add_argument(
        "--snippets_only",
        action="store_true",
        help="Use only the changed snippets (before/after) for verification, without full papers. Requires fix_summary CSV files with change_details."
    )
    parser.add_argument(
        "--remove_tables",
        action="store_true",
        help="Remove all markdown tables from both original and revised papers before verification."
    )
    parser.add_argument(
        "--use_full_category_names",
        action="store_true",
        help="Use full category names (e.g., 'Insufficient Baselines/Comparisons') instead of short names (e.g., 'Baselines') on plot X-axis."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for vLLM generation (default: 0.0 for deterministic output)"
    )
    
    args = parser.parse_args()
    
    # Validate endpoint URL
    endpoint = args.vllm_endpoint
    if not endpoint.startswith(('http://', 'https://')):
        raise ValueError(f"Invalid endpoint URL: {endpoint}. Must start with http:// or https://")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name_safe = args.model_name.replace('.', '_').replace('-', '_').replace('/', '_').replace(':', '_')
        if args.ablation_name:
            output_dir = data_dir / f"llm_verification_{model_name_safe}_{args.ablation_name}"
        else:
            output_dir = data_dir / f"llm_verification_{model_name_safe}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Register signal handler for graceful shutdown (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    print("ℹ️  Press Ctrl+C to gracefully save progress and exit\n")
    
    # Find categories - data structure: data_dir/NeurIPS2024/{category_id}/
    neurips_path = data_dir / "NeurIPS2024"
    if not neurips_path.exists():
        raise ValueError(f"NeurIPS2024 directory not found in {data_dir}. Expected: {neurips_path}")
    
    print(f"✅ Found NeurIPS2024 directory: {neurips_path}", flush=True)
    
    if args.categories:
        categories = args.categories
    else:
        categories = [d.name for d in neurips_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        categories.sort()
    
    print(f"✅ Found {len(categories)} categories: {categories}", flush=True)
    
    # Verify category structure (only show if there are issues)
    for cat in categories:
        cat_path = neurips_path / cat
        if cat_path.exists():
            planted_error_path = cat_path / "planted_error"
            latest_path = cat_path / "latest"
            deplanted_path = cat_path / "de-planted_error"
            v1_path = cat_path / "v1"
            
            # Check required folders based on flags
            if args.use_v1_as_flawed:
                # When using v1, we need: v1, latest, de-planted_error, and planted_error (for CSV)
                required_folders_ok = (v1_path.exists() and latest_path.exists() and 
                                      deplanted_path.exists() and planted_error_path.exists())
                if not required_folders_ok:
                    print(f"⚠️ Category {cat} missing some folders:", flush=True)
                    print(f"    - v1: {'✅' if v1_path.exists() else '❌'}", flush=True)
                    print(f"    - planted_error (for CSV): {'✅' if planted_error_path.exists() else '❌'}", flush=True)
                    print(f"    - latest: {'✅' if latest_path.exists() else '❌'}", flush=True)
                    print(f"    - de-planted_error: {'✅' if deplanted_path.exists() else '❌'}", flush=True)
            else:
                # Default: need planted_error, latest, de-planted_error
                if not (planted_error_path.exists() and latest_path.exists() and deplanted_path.exists()):
                    print(f"⚠️ Category {cat} missing some folders:", flush=True)
                    print(f"    - planted_error: {'✅' if planted_error_path.exists() else '❌'}", flush=True)
                    print(f"    - latest: {'✅' if latest_path.exists() else '❌'}", flush=True)
                    print(f"    - de-planted_error: {'✅' if deplanted_path.exists() else '❌'}", flush=True)
    
    print("="*80, flush=True)
    print("Revision Quality Verification (vLLM)", flush=True)
    print("="*80, flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(f"vLLM Endpoint: {endpoint}", flush=True)
    print(f"Model name (for output): {args.model_name}", flush=True)
    if args.ablation_name:
        print(f"Ablation: {args.ablation_name}", flush=True)
    print(f"Comparison type: {args.comparison_type}", flush=True)
    if args.use_v1_as_flawed:
        print(f"Using v1/ as flawed paper (v1 vs latest for true_good, v1 vs de-planted_error for fake_good)", flush=True)
    else:
        print(f"Using planted_error/ as flawed paper (planted_error vs latest for true_good, planted_error vs de-planted_error for fake_good)", flush=True)
    if args.snippets_only:
        print(f"⚠️ Snippets-only mode: Using only changed sections, not full papers", flush=True)
    if args.remove_tables:
        print(f"⚠️ Table removal mode: All markdown tables will be removed from papers", flush=True)
    print(f"Categories to process: {categories}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Temperature: {args.temperature}", flush=True)
    print(flush=True)
    
    max_workers = args.max_workers
    print(f"Max workers: {max_workers}", flush=True)
    print(flush=True)
    
    # If resuming, load existing results and build set of processed keys
    existing_df: Optional[pd.DataFrame] = None
    processed_keys = set()
    results_csv = output_dir / "verification_scores.csv"
    if args.resume and results_csv.exists():
        try:
            existing_df = pd.read_csv(results_csv)
            for _, row in existing_df.iterrows():
                rev_type = row.get("revision_type", "unknown")
                processed_keys.add(
                    (row["category"], row["paper_folder"], row["flaw_id"], rev_type)
                )
            print(f"Resuming from existing results: {len(existing_df)} rows, {len(processed_keys)} distinct paper/revision pairs")
        except Exception as e:
            print(f"⚠️ Could not read existing results from {results_csv}: {e}")
            existing_df = None
            processed_keys = set()
    
    # Fetch rebuttals if flag is enabled
    rebuttals = {}
    if args.include_rebuttals:
        if not OPENREVIEW_AVAILABLE:
            print("⚠️ WARNING: --include_rebuttals specified but openreview library not available. Install with: pip install openreview-py", flush=True)
        else:
            print("Fetching rebuttals from OpenReview API v2 using get_all_notes...", flush=True)
            # Collect all paper IDs from all categories
            paper_ids = set()
            for category in categories:
                category_path = data_dir / "NeurIPS2024" / category
                csv_source_dir = category_path / "planted_error"
                if csv_source_dir.exists():
                    csv_paper_folders = [d for d in csv_source_dir.iterdir() if d.is_dir()]
                    for csv_paper_folder in csv_paper_folders:
                        paper_folder_name = csv_paper_folder.name
                        paper_id = paper_folder_name.split('_')[0] if '_' in paper_folder_name else paper_folder_name
                        paper_ids.add(paper_id)
            
            if paper_ids:
                print(f"Found {len(paper_ids)} unique paper IDs. Fetching all rebuttals in bulk...", flush=True)
                # Use default venue ID for NeurIPS 2024
                venue_id = DEFAULT_VENUE_ID
                # Set up cache path in output directory
                cache_path = output_dir / REBUTTAL_CACHE_FILENAME
                rebuttals = fetch_rebuttals_batch(
                    list(paper_ids), 
                    venue_id=venue_id, 
                    cache_path=cache_path,
                    overwrite_cache=args.overwrite_rebuttals_cache
                )
                found_count = sum(1 for r in rebuttals.values() if r is not None)
                print(f"✅ Total rebuttals available: {found_count}/{len(paper_ids)} papers", flush=True)
            else:
                print("⚠️ No paper IDs found to fetch rebuttals for", flush=True)
            print(flush=True)
    
    # Process all categories (only missing pairs if --resume is enabled)
    all_results = []
    for category in categories:
        # Check if we should exit gracefully
        if should_exit.is_set():
            print("\n⚠️ Exiting gracefully due to interrupt signal...", flush=True)
            break
        
        print(f"Processing category: {category}", flush=True)
        results = process_category(
            data_dir=data_dir,
            category=category,
            model_name=args.model_name,
            comparison_type=args.comparison_type,
            output_dir=output_dir,
            endpoint=endpoint,
            max_workers=max_workers,
            processed_keys=processed_keys,
            results_csv=results_csv,  # Pass CSV path for incremental saving
            ablation_name=args.ablation_name,
            dual_scores=args.dual_scores,
            use_v1_as_flawed=args.use_v1_as_flawed,
            rebuttals=rebuttals if args.include_rebuttals else None,
            snippets_only=args.snippets_only,
            remove_tables=args.remove_tables,
            temperature=args.temperature
        )
        all_results.extend(results)
        print(f"  Completed {category}: {len(results)} results")
        print()
    
    # Convert new results to DataFrame
    if all_results:
        df_new = pd.DataFrame(all_results)
    else:
        df_new = pd.DataFrame()
        print("No new results collected in this run.")
    
    # Merge with existing results if resuming, or use existing results if no new ones
    if existing_df is not None:
        if not df_new.empty:
            combined_df = pd.concat([existing_df, df_new], ignore_index=True)
            # Drop duplicates based on unique key
            combined_df.drop_duplicates(
                subset=["category", "paper_folder", "flaw_id", "revision_type"],
                keep="first",
                inplace=True,
            )
        else:
            # No new results, but use existing results for plotting
            combined_df = existing_df
            print(f"Using existing results ({len(combined_df)} rows) for statistics and plotting...")
    else:
        if df_new.empty:
            print("No results available! Cannot generate plots.")
            return
        combined_df = df_new
    
    # Check if we have any data to work with
    if combined_df.empty:
        print("⚠️ No results available! Cannot generate plots.")
        return
    
    # Save raw results (combined)
    combined_df.to_csv(results_csv, index=False)
    print(f"✅ Saved raw results: {results_csv}")
    
    # Save full LLM responses for further analysis (based on combined results)
    # Check if dual scores mode
    dual_scores_mode = 'quality_score' in combined_df.columns and 'verifiability_score' in combined_df.columns
    
    llm_responses = []
    for _, row in combined_df.iterrows():
        response = {
            'category': row['category'],
            'paper_folder': row['paper_folder'],
            'flaw_id': row['flaw_id'],
            'revision_type': row.get('revision_type', 'unknown'),
            'flaw_description': row['flaw_description'],
        }
        
        if dual_scores_mode:
            # Dual scores mode
            response.update({
                'quality_score': row.get('quality_score'),
                'verifiability_score': row.get('verifiability_score'),
                'explanation': row.get('explanation', '')
            })
        else:
            # Single score mode
            response.update({
                'score': row.get('score'),
                'reasoning': row.get('reasoning', '')
            })
        
        llm_responses.append(response)
    
    responses_json = output_dir / "llm_responses.json"
    with open(responses_json, 'w', encoding='utf-8') as f:
        json.dump(llm_responses, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved full LLM responses: {responses_json}")
    
    # Compute statistics on combined results
    print("\nComputing statistics...")
    stats = compute_treatment_effect(combined_df)
    
    if stats:
        dual_scores_mode = stats.get('dual_scores_mode', False)
        
        if dual_scores_mode:
            # Dual scores mode: print statistics for Quality and Verifiability separately
            quality_stats = stats['quality']
            verifiability_stats = stats['verifiability']
            
            print(f"\nOverall Treatment Effect Statistics - Quality:")
            print(f"  Number of pairs: {quality_stats['n_pairs']}")
            print(f"  Mean difference (True Good - Fake Good): {quality_stats['mean_difference']:.3f}")
            print(f"  Standard deviation of differences: {quality_stats['std_difference']:.3f}")
            print(f"  Treatment effect (mean / std): {quality_stats['treatment_effect']:.3f}")
            q_cohen_d_str = f"{quality_stats['cohen_d']:.3f}" if quality_stats['cohen_d'] is not None else 'N/A'
            q_ci_low_str = f"{quality_stats['cohen_d_ci_low']:.3f}" if quality_stats['cohen_d_ci_low'] is not None else 'N/A'
            q_ci_high_str = f"{quality_stats['cohen_d_ci_high']:.3f}" if quality_stats['cohen_d_ci_high'] is not None else 'N/A'
            print(f"  Cohen's d: {q_cohen_d_str}")
            print(f"  95% CI: [{q_ci_low_str}, {q_ci_high_str}]")
            
            print(f"\nOverall Treatment Effect Statistics - Verifiability:")
            print(f"  Number of pairs: {verifiability_stats['n_pairs']}")
            print(f"  Mean difference (True Good - Fake Good): {verifiability_stats['mean_difference']:.3f}")
            print(f"  Standard deviation of differences: {verifiability_stats['std_difference']:.3f}")
            print(f"  Treatment effect (mean / std): {verifiability_stats['treatment_effect']:.3f}")
            v_cohen_d_str = f"{verifiability_stats['cohen_d']:.3f}" if verifiability_stats['cohen_d'] is not None else 'N/A'
            v_ci_low_str = f"{verifiability_stats['cohen_d_ci_low']:.3f}" if verifiability_stats['cohen_d_ci_low'] is not None else 'N/A'
            v_ci_high_str = f"{verifiability_stats['cohen_d_ci_high']:.3f}" if verifiability_stats['cohen_d_ci_high'] is not None else 'N/A'
            print(f"  Cohen's d: {v_cohen_d_str}")
            print(f"  95% CI: [{v_ci_low_str}, {v_ci_high_str}]")
            
            # Print per-category statistics for Quality
            if 'category_stats' in quality_stats and not quality_stats['category_stats'].empty:
                print(f"\nPer-Category Statistics - Quality:")
                cat_stats = quality_stats['category_stats'].sort_values('category')
                for _, row in cat_stats.iterrows():
                    q_d_str = f"{row['cohen_d']:.3f}" if row['cohen_d'] is not None else 'N/A'
                    q_ci_low_str = f"{row['cohen_d_ci_low']:.3f}" if row['cohen_d_ci_low'] is not None else 'N/A'
                    q_ci_high_str = f"{row['cohen_d_ci_high']:.3f}" if row['cohen_d_ci_high'] is not None else 'N/A'
                    print(f"  {row['category']}:")
                    print(f"    n={row['n_pairs']}, mean_diff={row['mean_difference']:.3f}, "
                          f"Cohen's d={q_d_str}, "
                          f"95% CI=[{q_ci_low_str}, {q_ci_high_str}]")
            
            # Print per-category statistics for Verifiability
            if 'category_stats' in verifiability_stats and not verifiability_stats['category_stats'].empty:
                print(f"\nPer-Category Statistics - Verifiability:")
                cat_stats = verifiability_stats['category_stats'].sort_values('category')
                for _, row in cat_stats.iterrows():
                    v_d_str = f"{row['cohen_d']:.3f}" if row['cohen_d'] is not None else 'N/A'
                    v_ci_low_str = f"{row['cohen_d_ci_low']:.3f}" if row['cohen_d_ci_low'] is not None else 'N/A'
                    v_ci_high_str = f"{row['cohen_d_ci_high']:.3f}" if row['cohen_d_ci_high'] is not None else 'N/A'
                    print(f"  {row['category']}:")
                    print(f"    n={row['n_pairs']}, mean_diff={row['mean_difference']:.3f}, "
                          f"Cohen's d={v_d_str}, "
                          f"95% CI=[{v_ci_low_str}, {v_ci_high_str}]")
            
            # Print aggregated score statistics (if available)
            if 'aggregated' in stats:
                aggregated_stats = stats['aggregated']
                print(f"\nOverall Treatment Effect Statistics - Aggregated (Quality + Verifiability):")
                print(f"  Number of pairs: {aggregated_stats['n_pairs']}")
                print(f"  Mean difference (True Good - Fake Good): {aggregated_stats['mean_difference']:.3f}")
                print(f"  Standard deviation of differences: {aggregated_stats['std_difference']:.3f}")
                print(f"  Treatment effect (mean / std): {aggregated_stats['treatment_effect']:.3f}")
                a_cohen_d_str = f"{aggregated_stats['cohen_d']:.3f}" if aggregated_stats['cohen_d'] is not None else 'N/A'
                a_ci_low_str = f"{aggregated_stats['cohen_d_ci_low']:.3f}" if aggregated_stats['cohen_d_ci_low'] is not None else 'N/A'
                a_ci_high_str = f"{aggregated_stats['cohen_d_ci_high']:.3f}" if aggregated_stats['cohen_d_ci_high'] is not None else 'N/A'
                print(f"  Cohen's d: {a_cohen_d_str}")
                print(f"  95% CI: [{a_ci_low_str}, {a_ci_high_str}]")
                
                # Print per-category statistics for Aggregated
                if 'category_stats' in aggregated_stats and not aggregated_stats['category_stats'].empty:
                    print(f"\nPer-Category Statistics - Aggregated:")
                    cat_stats = aggregated_stats['category_stats'].sort_values('category')
                    for _, row in cat_stats.iterrows():
                        a_d_str = f"{row['cohen_d']:.3f}" if row['cohen_d'] is not None else 'N/A'
                        a_ci_low_str = f"{row['cohen_d_ci_low']:.3f}" if row['cohen_d_ci_low'] is not None else 'N/A'
                        a_ci_high_str = f"{row['cohen_d_ci_high']:.3f}" if row['cohen_d_ci_high'] is not None else 'N/A'
                        print(f"  {row['category']}:")
                        print(f"    n={row['n_pairs']}, mean_diff={row['mean_difference']:.3f}, "
                              f"Cohen's d={a_d_str}, "
                              f"95% CI=[{a_ci_low_str}, {a_ci_high_str}]")
            
            # Save statistics
            stats_json = output_dir / "statistics.json"
            stats_to_save = {
                'dual_scores_mode': True,
                'quality': {k: v for k, v in quality_stats.items() if k != 'category_stats'},
                'verifiability': {k: v for k, v in verifiability_stats.items() if k != 'category_stats'},
                'detailed_results': stats['detailed_results'].to_dict('records')
            }
            if 'category_stats' in quality_stats and not quality_stats['category_stats'].empty:
                stats_to_save['quality']['category_stats'] = quality_stats['category_stats'].to_dict('records')
            if 'category_stats' in verifiability_stats and not verifiability_stats['category_stats'].empty:
                stats_to_save['verifiability']['category_stats'] = verifiability_stats['category_stats'].to_dict('records')
            if 'aggregated' in stats:
                aggregated_stats = stats['aggregated']
                stats_to_save['aggregated'] = {k: v for k, v in aggregated_stats.items() if k != 'category_stats'}
                if 'category_stats' in aggregated_stats and not aggregated_stats['category_stats'].empty:
                    stats_to_save['aggregated']['category_stats'] = aggregated_stats['category_stats'].to_dict('records')
            with open(stats_json, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            print(f"✅ Saved statistics: {stats_json}")
            
            # Save detailed differences
            diff_csv = output_dir / "score_differences.csv"
            stats['detailed_results'].to_csv(diff_csv, index=False)
            print(f"✅ Saved differences: {diff_csv}")
            
            # Save category statistics for Quality
            if 'category_stats' in quality_stats and not quality_stats['category_stats'].empty:
                cat_csv = output_dir / "category_statistics_quality.csv"
                quality_stats['category_stats'].to_csv(cat_csv, index=False)
                print(f"✅ Saved category statistics (Quality): {cat_csv}")
            
            # Save category statistics for Verifiability
            if 'category_stats' in verifiability_stats and not verifiability_stats['category_stats'].empty:
                cat_csv = output_dir / "category_statistics_verifiability.csv"
                verifiability_stats['category_stats'].to_csv(cat_csv, index=False)
                print(f"✅ Saved category statistics (Verifiability): {cat_csv}")
            
            # Save category statistics for Aggregated (if available)
            if 'aggregated' in stats:
                aggregated_stats = stats['aggregated']
                if 'category_stats' in aggregated_stats and not aggregated_stats['category_stats'].empty:
                    cat_csv = output_dir / "category_statistics_aggregated.csv"
                    aggregated_stats['category_stats'].to_csv(cat_csv, index=False)
                    print(f"✅ Saved category statistics (Aggregated): {cat_csv}")
        else:
            # Single score mode: original code
            print(f"\nOverall Treatment Effect Statistics:")
            print(f"  Number of pairs: {stats['n_pairs']}")
            print(f"  Mean difference (True Good - Fake Good): {stats['mean_difference']:.3f}")
            print(f"  Standard deviation of differences: {stats['std_difference']:.3f}")
            print(f"  Treatment effect (mean / std): {stats['treatment_effect']:.3f}")
            cohen_d_str = f"{stats['cohen_d']:.3f}" if stats['cohen_d'] is not None else 'N/A'
            ci_low_str = f"{stats['cohen_d_ci_low']:.3f}" if stats['cohen_d_ci_low'] is not None else 'N/A'
            ci_high_str = f"{stats['cohen_d_ci_high']:.3f}" if stats['cohen_d_ci_high'] is not None else 'N/A'
            print(f"  Cohen's d: {cohen_d_str}")
            print(f"  95% CI: [{ci_low_str}, {ci_high_str}]")
    
            # Print per-category statistics
            if 'category_stats' in stats and not stats['category_stats'].empty:
                print(f"\nPer-Category Statistics:")
                cat_stats = stats['category_stats'].sort_values('category')
                for _, row in cat_stats.iterrows():
                    print(f"  {row['category']}:")
            
            # Save statistics for single score mode
            stats_json = output_dir / "statistics.json"
            stats_to_save = {k: v for k, v in stats.items() if k not in ['detailed_results', 'category_stats']}
            stats_to_save['detailed_results'] = stats['detailed_results'].to_dict('records')
            if 'category_stats' in stats and not stats['category_stats'].empty:
                stats_to_save['category_stats'] = stats['category_stats'].to_dict('records')
            with open(stats_json, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            print(f"✅ Saved statistics: {stats_json}")
            
            # Save detailed differences
            diff_csv = output_dir / "score_differences.csv"
            stats['detailed_results'].to_csv(diff_csv, index=False)
            print(f"✅ Saved differences: {diff_csv}")
                
            # Save category statistics
            if 'category_stats' in stats and not stats['category_stats'].empty:
                cat_csv = output_dir / "category_statistics.csv"
                stats['category_stats'].to_csv(cat_csv, index=False)
                print(f"✅ Saved category statistics: {cat_csv}")
        
        # Create plots
        print("\nGenerating plots...")
        create_plots(stats, output_dir, args.model_name, use_full_names=args.use_full_category_names)
    else:
        print("⚠️ Could not compute statistics (insufficient paired data)")
    
    # Detect lazy authors (true_good score < 5) - independent of stats computation
    if args.detect_lazy_authors:
        print("\nAnalyzing for lazy authors (true_good score < 5)...")
        lazy_analysis = detect_lazy_authors(combined_df)
        
        if lazy_analysis is not None and not lazy_analysis.empty:
            lazy_csv = output_dir / "lazy_authors_detection.csv"
            lazy_analysis.to_csv(lazy_csv, index=False)
            print(f"✅ Saved lazy authors detection: {lazy_csv}")
            
            print(f"\nLazy Authors Detection Summary:")
            print(f"  Total papers with true_good scores: {len(lazy_analysis)}")
            low_scores = lazy_analysis[lazy_analysis['true_good_score'] < 5]
            print(f"  Papers with true_good score < 5: {len(low_scores)} ({len(low_scores)/len(lazy_analysis)*100:.1f}%)")
            
            if len(low_scores) > 0:
                print(f"\n  Papers flagged as potentially lazy:")
                for _, row in low_scores.iterrows():
                    fake_good_str = f"{row['fake_good_score']:.1f}" if pd.notna(row['fake_good_score']) else 'N/A'
                    diff_str = f"{row['difference']:.1f}" if pd.notna(row['difference']) else 'N/A'
                    # Construct full path from the root data directory to the planted error paper
                    paper_rel_path = (
                        f"NeurIPS2024/{row['category']}/planted_error/"
                        f"{row['paper_folder']}/flawed_papers/{row['flaw_id']}.md"
                    )
                    paper_full_path = data_dir / paper_rel_path
                    print(f"    - {paper_full_path}: "
                          f"true_good={row['true_good_score']:.1f}, "
                          f"fake_good={fake_good_str}, "
                          f"diff={diff_str}")
        else:
            print("⚠️ Could not detect lazy authors (insufficient data)")
    
    print("\n" + "="*80)
    print("Verification complete!")
    print("="*80)

if __name__ == "__main__":
    main()