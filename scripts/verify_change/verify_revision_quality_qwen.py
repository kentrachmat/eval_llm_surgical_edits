#!/usr/bin/env python3
"""
Wrapper around verify_revision_quality.py to use Qwen API instead of Gemini.

This wrapper imports all functionality from verify_revision_quality.py and only
overrides the API-specific parts (API calls and key management), so changes to
the original script automatically work here.
"""

import os
import json
import sys
import time
from pathlib import Path
from openai import OpenAI
from typing import Optional, Dict, Tuple, List
from pydantic import BaseModel
from tqdm import tqdm
import threading

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the original module's functions and classes we need
import verify_revision_quality as orig

# Re-export everything from the original module
# This ensures all non-API functions work the same
from verify_revision_quality import *

# Load multiple paid Qwen API keys (QWEN_API_KEY_PAID_1, QWEN_API_KEY_PAID_2, ...)
QWEN_API_KEYS_PAID: Dict[str, str] = {}
for i in range(1, 10):
    key = os.getenv(f'QWEN_API_KEY_PAID_{i}')
    if key:
        QWEN_API_KEYS_PAID[f'PAID_{i}'] = key

# Load regular Qwen API keys (QWEN_API_KEY_1, QWEN_API_KEY_2, ...)
QWEN_API_KEYS: Dict[str, str] = {}
for i in range(1, 10):
    key = os.getenv(f'QWEN_API_KEY_{i}')
    if key:
        QWEN_API_KEYS[str(i)] = key

# Also try DASHSCOPE_API_KEY (default environment variable name)
if not QWEN_API_KEYS:
    dashscope_key = os.getenv('DASHSCOPE_API_KEY')
    if dashscope_key:
        QWEN_API_KEYS = {'SINGLE': dashscope_key}

if not QWEN_API_KEYS and not QWEN_API_KEYS_PAID:
    raise ValueError("No Qwen API keys found in environment variables. Set QWEN_API_KEY_1, QWEN_API_KEY_PAID_1, or DASHSCOPE_API_KEY")

# Print summary
if QWEN_API_KEYS_PAID:
    print(f"✅ Loaded {len(QWEN_API_KEYS_PAID)} paid Qwen API key(s): {list(QWEN_API_KEYS_PAID.keys())}")
print(f"✅ Loaded {len(QWEN_API_KEYS)} regular Qwen API keys")

# Qwen model RPM limits (adjust as needed)
QWEN_MODEL_RPM_LIMITS = {
    "qwen3-vl-235b-a22b-instruct": 60,
    "qwen-turbo": 30,
    "qwen-plus": 15,
    "qwen-max": 5,
    "qwen-max-longcontext": 5,
}

# Rate limiting tracking (separate from Gemini)
qwen_key_last_used: Dict[str, float] = {}
qwen_key_lock = threading.Lock()
QWEN_USE_PAID_KEY = False

def get_qwen_request_delay_for_model(model_name: str) -> float:
    """Calculate request delay in seconds based on Qwen model's RPM limit."""
    rpm_limit = QWEN_MODEL_RPM_LIMITS.get(model_name, 30)
    return 60.0 / rpm_limit

def wait_for_qwen_rate_limit(key_name: str, delay: float):
    """Wait if necessary to respect rate limits. Skip delay for paid key or if delay is 0."""
    is_paid_key = key_name in QWEN_API_KEYS_PAID or key_name.startswith('PAID')
    if is_paid_key or QWEN_USE_PAID_KEY or delay is None or delay <= 0:
        return
    
    with qwen_key_lock:
        current_time = time.time()
        if key_name in qwen_key_last_used:
            time_since_last_use = current_time - qwen_key_last_used[key_name]
            if time_since_last_use < delay:
                sleep_time = delay - time_since_last_use
                time.sleep(sleep_time)
        qwen_key_last_used[key_name] = time.time()

def get_qwen_api_key_for_task(task_idx: int, use_paid: bool = False) -> Tuple[str, str]:
    """Get Qwen API key for a task using round-robin. Use paid keys if use_paid=True."""
    if use_paid and QWEN_API_KEYS_PAID:
        paid_key_names = list(QWEN_API_KEYS_PAID.keys())
        key_name = paid_key_names[task_idx % len(paid_key_names)]
        return key_name, QWEN_API_KEYS_PAID[key_name]
    
    if not QWEN_API_KEYS:
        if QWEN_API_KEYS_PAID:
            paid_key_names = list(QWEN_API_KEYS_PAID.keys())
            key_name = paid_key_names[task_idx % len(paid_key_names)]
            return key_name, QWEN_API_KEYS_PAID[key_name]
        raise ValueError("No Qwen API keys available")
    
    key_names = list(QWEN_API_KEYS.keys())
    key_name = key_names[task_idx % len(key_names)]
    return key_name, QWEN_API_KEYS[key_name]

def call_qwen_with_retries(
    api_key: str,
    key_name: str,
    prompt: str,
    response_model: BaseModel,
    max_retries: int = orig.MAX_RETRIES,
    request_delay: float = None,
    model_name: str = "qwen-plus"
) -> Optional[BaseModel]:
    """Call Qwen API with retries and structured output using OpenAI-compatible interface."""
    if request_delay is None:
        request_delay = get_qwen_request_delay_for_model(model_name)
    
    # Get and clean the JSON schema
    raw_schema = response_model.model_json_schema()
    cleaned_schema = orig.clean_json_schema(raw_schema)
    
    # Initialize OpenAI client for Qwen (using DashScope compatible endpoint)
    base_urls = [
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ]
    
    for attempt in range(max_retries):
        last_error = None
        for base_url in base_urls:
            try:
                wait_for_qwen_rate_limit(key_name, request_delay)
                
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                
                # Qwen uses OpenAI-compatible chat completions API
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are a helpful assistant that always responds with valid JSON matching the provided schema.'
                        },
                        {
                            'role': 'user',
                            'content': f"""{prompt}

IMPORTANT: Respond with ONLY valid JSON that matches this schema:
{json.dumps(cleaned_schema, indent=2)}

Do not include any markdown formatting or code blocks. Return only the JSON object."""
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                
                # Parse JSON response
                response_text = response.choices[0].message.content.strip()
                
                # Clean up any markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                elif response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                response_dict = json.loads(response_text)
                return response_model(**response_dict)
                
            except Exception as e:
                last_error = e
                if base_url == base_urls[-1]:
                    break
                continue
        
        # Retry logic
        if attempt < max_retries - 1:
            backoff = orig.INITIAL_BACKOFF_SECONDS * (2 ** attempt)
            time.sleep(backoff)
        else:
            if "Schema" not in str(last_error):
                tqdm.write(f"  ⚠️ Error after {max_retries} attempts: {last_error}")
            return None
    
    return None

# Override the original module's API call function
orig.call_gemini_with_retries = call_qwen_with_retries
orig.get_api_key_for_task = get_qwen_api_key_for_task
orig.wait_for_rate_limit = wait_for_qwen_rate_limit

# Override API key storage and constants
orig.GEMINI_API_KEYS_PAID = QWEN_API_KEYS_PAID
orig.GEMINI_API_KEYS = QWEN_API_KEYS
orig.GEMINI_MODEL_RPM_LIMITS = QWEN_MODEL_RPM_LIMITS

# Override get_request_delay_for_model
orig.get_request_delay_for_model = get_qwen_request_delay_for_model

# Note: score_revision_quality doesn't need to be overridden because it calls
# call_gemini_with_retries, which we've already replaced with call_qwen_with_retries.
# Similarly, process_paper_pair calls get_api_key_for_task and score_revision_quality,
# both of which now use our Qwen functions

# Now all API functions are patched to use Qwen
# Just call the original main - it will use our overridden functions
if __name__ == "__main__":
    # The original main will use our patched functions automatically
    # We've already replaced:
    # - call_gemini_with_retries -> call_qwen_with_retries (uses OpenAI-compatible Qwen API)
    # - get_api_key_for_task -> get_qwen_api_key_for_task (uses QWEN_API_KEYS)
    # - wait_for_rate_limit -> wait_for_qwen_rate_limit (uses QWEN rate limits)
    # - get_request_delay_for_model -> get_qwen_request_delay_for_model (uses QWEN_MODEL_RPM_LIMITS)
    # - GEMINI_API_KEYS_PAID -> QWEN_API_KEYS_PAID (loaded from QWEN_API_KEY_PAID_*)
    # - GEMINI_API_KEYS -> QWEN_API_KEYS (loaded from QWEN_API_KEY_* or DASHSCOPE_API_KEY)
    # - GEMINI_MODEL_RPM_LIMITS -> QWEN_MODEL_RPM_LIMITS (qwen-specific limits)
    #
    # All other functions (statistics, plotting, file I/O, etc.) work unchanged
    
    orig.main()
