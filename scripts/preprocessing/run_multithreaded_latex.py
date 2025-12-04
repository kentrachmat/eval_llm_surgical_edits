#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ##############################################################################
# Python Script for Processing LaTeX Papers to Markdown
#
# Original Workflow:
# This code looks into a base directory for input CSV (not used in this script version)
# containing info of papers to be parsed into HTML/Markdown.
# Key steps:
# - Fetches paper information from OpenReview.
# - Searches for corresponding papers on arXiv.
# - Downloads LaTeX sources from arXiv.
# - Converts LaTeX to Markdown using a custom converter.
# - Handles bibliography, figures, and various LaTeX complexities.
# - Processes papers in parallel.
#
# Note: For papers without arXiv versions, the original notebook pointed to a
# Colab notebook for PDF parsing. That part is not included here.
# ##############################################################################

# --- Dependencies ---
# Make sure you have these Python packages installed:
# pip install arxiv openreview-py openai datasets PyPDF2 pdf2image pillow tqdm python-dotenv
#
# System Dependencies (ensure these are installed on your system):
# - poppler-utils (for pdf2image, e.g., `sudo apt-get install poppler-utils` on Debian/Ubuntu)
# - TeX Live (for pdflatex, bibtex, e.g., `sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra texlive-science texlive-publishers`)
# - Pandoc (version 3.x recommended)
# ---

from latex import *

import os
import json
import getpass
import requests
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import arxiv
import openreview
import re
import tarfile
# import bz2 # bz2 is part of tarfile's 'r:*' mode, direct import not always needed
import concurrent.futures
import traceback
import subprocess
import tempfile
from pathlib import Path
import argparse
import sys
import shutil
import urllib.parse
import time
import io
import xml.etree.ElementTree as ET
import signal
import threading
from dotenv import load_dotenv

# Attempt to import PyPDF2
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# Attempt to import BeautifulSoup
try:
    from bs4 import BeautifulSoup, Comment
except ImportError:
    BeautifulSoup = None
    Comment = None

# Attempt to import Pillow (PIL)
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# Attempt to import pdf2image
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# --- Global Variables & Configuration ---
# Load .env file if it exists
load_dotenv()

# API keys can be set as environment variables or passed via CLI
# These are default values if not overridden
DEFAULT_ELSEVIER_API_KEY = os.environ.get("ELSEVIER_API_KEY", os.environ.get("elsevier-api-key"))
DEFAULT_SPRINGER_API_KEY = os.environ.get("SPRINGER_API_KEY", os.environ.get("springer-api-key"))
DEFAULT_SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", os.environ.get("semantic-scholar-api-key"))
DEFAULT_OPENREVIEW_USERNAME = os.environ.get("OPENREVIEW_USERNAME")
DEFAULT_OPENREVIEW_PASSWORD = os.environ.get("OPENREVIEW_PASSWORD")

# Template for Markdown output (originally from %%writefile template.md)
DEFAULT_TEMPLATE_MD_CONTENT = """$if(title)$
# $title$
$endif$
$if(abstract)$

## Abstract

$abstract$

$endif$
$for(header-includes)$
$header-includes$

$endfor$
$for(include-before)$
$include-before$

$endfor$
$if(toc)$
$table-of-contents$

$endif$
$body$
$for(include-after)$

$include-after$
$endfor$
"""
DEFAULT_TEMPLATE_MD_PATH = "template.md"

# List of common venue style file name patterns (regex)
VENUE_STYLE_PATTERNS = [
    r"neurips_?\d{4}", r"icml_?\d{4}", r"iclr_?\d{4}(_conference)?", r"aistats_?\d{4}",
    r"collas_?\d{4}(_conference)?", r"cvpr_?\d{4}?", r"iccv_?\d{4}?", r"eccv_?\d{4}?",
    r"acl_?\d{4}?", r"emnlp_?\d{4}?", r"naacl_?\d{4}?", r"siggraph_?\d{4}?",
    r"chi_?\d{4}?", r"aaai_?\d{2,4}", r"ijcai_?\d{2,4}", r"uai_?\d{4}?",
    r"IEEEtran", r"ieeeconf", r"acmart"
]

PROCESSED_BBL_MARKER = "% L2M_PROCESSED_BBL_V1_DO_NOT_EDIT_MANUALLY_BELOW_THIS_LINE"
MAX_PANDOC_COMMENT_RETRIES = 10 # Max attempts to fix by commenting within a single strategy

# --- Pandoc Installation (Optional Helper) ---
def setup_pandoc(pandoc_version="3.1.9"): # Changed to a more common recent version
    """
    Downloads and installs a specific version of Pandoc.
    Note: This is for Debian/Ubuntu based systems.
    """
    if shutil.which("pandoc"):
        try:
            result = subprocess.run(["pandoc", "--version"], capture_output=True, text=True, check=True)
            print(f"[*] Pandoc already installed: {result.stdout.splitlines()[0]}")
            # Optionally, add version check here if a specific one is critical
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[!] Found pandoc, but 'pandoc --version' failed. Proceeding with setup.")

    print(f"[*] Attempting to install Pandoc version {pandoc_version}...")
    # For 3.1.9, the asset is pandoc-3.1.9-1-amd64.deb
    deb_filename = f"pandoc-{pandoc_version}-1-amd64.deb"
    download_url = f"https://github.com/jgm/pandoc/releases/download/{pandoc_version}/{deb_filename}"

    commands = [
        f"echo 'Downloading Pandoc {pandoc_version}...' && wget {download_url}",
        f"echo 'Installing {deb_filename}...' && sudo dpkg -i {deb_filename}",
        "echo 'Fixing any missing dependencies...' && sudo apt-get install -f -y",
        f"echo 'Cleaning up {deb_filename}...' && rm {deb_filename}",
        "echo 'Verifying Pandoc installation...' && pandoc --version"
    ]

    for cmd in commands:
        print(f"\nExecuting: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            if result.stdout: print("Output:\n", result.stdout)
            if result.stderr: print("Error output (if any):\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e.cmd}")
            print("Stderr:\n", e.stderr)
            print("Stdout:\n", e.stdout)
            print("[!] Pandoc installation might have failed. Please check errors or install manually.")
            return False
        except Exception as e_gen:
            print(f"[!] An unexpected error occurred during Pandoc setup: {e_gen}")
            return False
    print("\n[+] Pandoc installation script completed successfully!")
    return True



# --- LaTeX Download and Parse Class ---
class LaTeX_Download:
    def __init__(self, url, output_dir=None,
                 quiet_converter=0, 
                 max_workers=8,
                 template_path_converter=DEFAULT_TEMPLATE_MD_PATH,
                 openalex_email_converter="ai-reviewer@example.com", # Use a generic default
                 openalex_api_key_converter=None,
                 elsevier_api_key_converter=None, springer_api_key_converter=None,
                 semantic_scholar_api_key_converter=None, poppler_path_converter=None,
                 output_debug_tex_converter=False, thread_name=None):
        self.url = url
        self.output_dir = output_dir
        self._create_output_dir()

        self.quiet_converter = quiet_converter
        self.max_workers = max_workers
        self.template_path_converter = template_path_converter
        self.openalex_email_converter = openalex_email_converter
        self.openalex_api_key_converter = openalex_api_key_converter
        self.elsevier_api_key_converter = elsevier_api_key_converter
        self.springer_api_key_converter = springer_api_key_converter
        self.semantic_scholar_api_key_converter = semantic_scholar_api_key_converter
        self.poppler_path_converter = poppler_path_converter
        self.output_debug_tex_converter = output_debug_tex_converter
        self.thread_name = thread_name if thread_name else threading.current_thread().name


    def _create_output_dir(self):
        if self.output_dir is None:
            url_part = self.url.split("/")[-1]
            if not url_part or url_part == "e-print": url_part = self.url.split("/")[-2]
            self.output_dir = url_part.replace(".", "_").replace("v1","").replace("v2","")
            if not self.output_dir : self.output_dir = "downloaded_latex_default"
        self.output_dir = str(self.output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # print(f"[{self.thread_name}] Output dir for {self.url.split('/')[-1]} set to: {Path(self.output_dir).resolve()}")

    def download_e_print(self):
        e_print_url = self.url
        # print(f"[{self.thread_name}] Downloading: {e_print_url.split('/')[-1]}")
        output_dir_path = Path(self.output_dir)
        zip_folder_path = output_dir_path / "e-print_source"
        zip_file_name = "e-print_source_archive.tar.gz" # Assuming .tar.gz, adjust if needed
        zip_file_path = output_dir_path / zip_file_name

        try:
            zip_response = requests.get(e_print_url, timeout=60)
            zip_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[{self.thread_name}] Download failed for {e_print_url.split('/')[-1]}: {e}", file=sys.stderr)
            raise

        zip_folder_path.mkdir(parents=True, exist_ok=True)
        with open(zip_file_path, "wb") as f: f.write(zip_response.content)

        if os.path.getsize(zip_file_path) == 0:
            if zip_file_path.exists(): zip_file_path.unlink()
            raise IOError(f"[{self.thread_name}] Downloaded archive is empty: {e_print_url.split('/')[-1]}")

        try:
            with tarfile.open(zip_file_path, 'r:*') as tar_ref: # 'r:*' handles auto-compression
                members = [m for m in tar_ref.getmembers() if not (m.name.startswith('/') or '..' in m.name)]
                tar_ref.extractall(path=zip_folder_path, members=members)
            # print(f"[{self.thread_name}] Extracted: {e_print_url.split('/')[-1]} to {zip_folder_path}")
            return str(zip_folder_path)
        except tarfile.ReadError as e:
            print(f"[{self.thread_name}] Tar extract error for {e_print_url.split('/')[-1]}: {e}", file=sys.stderr)
            try:
                with open(zip_file_path, 'r', encoding='utf-8', errors='ignore') as f_check:
                    first_few_lines = "".join(f_check.readline() for _ in range(2)).lower()
                if "<!doctype html>" in first_few_lines or "<html" in first_few_lines :
                    print(f"[{self.thread_name}] Downloaded file {zip_file_path} appears to be an HTML page.", file=sys.stderr)
            except: pass
            raise
        except Exception as e_other_extract:
            print(f"[{self.thread_name}] Other extraction error for {e_print_url.split('/')[-1]}: {e_other_extract}", file=sys.stderr)
            raise

    def parse_md_from_eprint(self, extracted_source_path_str):
        project_path = Path(extracted_source_path_str).resolve()
        output_folder_for_md = Path(self.output_dir) / "structured_paper_output"

        if not project_path.is_dir():
            print(f"[{self.thread_name}] Error: Extracted project folder '{project_path}' not found.", file=sys.stderr)
            return False
        try:
            output_folder_for_md.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[{self.thread_name}] Error creating output dir for Markdown '{output_folder_for_md}': {e}", file=sys.stderr)
            return False

        mock_cli_args = argparse.Namespace(
            quiet=self.quiet_converter, 
            max_workers=self.max_workers,
            openalex_email=self.openalex_email_converter, openalex_api_key=self.openalex_api_key_converter,
            elsevier_api_key=self.elsevier_api_key_converter, springer_api_key=self.springer_api_key_converter,
            semantic_scholar_api_key=self.semantic_scholar_api_key_converter, poppler_path=self.poppler_path_converter,
            output_debug_tex=self.output_debug_tex_converter
        )
        
        resolved_template_path = None
        if self.template_path_converter:
            p_template = Path(self.template_path_converter)
            if p_template.is_file(): resolved_template_path = str(p_template.resolve())
            else: print(f"[{self.thread_name}] Template '{self.template_path_converter}' not found, using Pandoc default.", file=sys.stderr)

        # print(f"[{self.thread_name}] Calling process_project_folder for: {project_path.name}")
        success = process_project_folder(
            str(project_path), str(output_folder_for_md), mock_cli_args, resolved_template_path, thread_name=self.thread_name,
        )
        # if success: print(f"[{self.thread_name}] Markdown parsed for {project_path.name}")
        # else: print(f"[{self.thread_name}] Markdown parsing failed for {project_path.name}")
        return success

    def download_and_parse(self):
        extracted_source_path = None
        download_archive_path = Path(self.output_dir) / "e-print_source_archive.tar.gz"
        try:
            extracted_source_path = self.download_e_print()
            # print(f"[{self.thread_name}] Downloaded: {self.url.split('/')[-1]}")
            if not self.parse_md_from_eprint(extracted_source_path):
                # print(f"[{self.thread_name}] Failed to parse Markdown from e-print: {self.url.split('/')[-1]}", file=sys.stderr)
                raise Exception(f"Failed to parse Markdown from e-print: {self.url.split('/')[-1]}")
            return True
        except Exception as e:
            print(f"[{self.thread_name}] Full processing failed for {self.url.split('/')[-1]}: {e}", file=sys.stderr)
            return False
        finally:
            if extracted_source_path and Path(extracted_source_path).exists():
                shutil.rmtree(extracted_source_path, ignore_errors=True)
            if download_archive_path.exists():
                download_archive_path.unlink(missing_ok=True)

# --- Helper Functions for Parallel Processing ---
def process_project_folder(project_folder_str, output_folder_str, cli_args, resolved_template_path_str, thread_name=None):
    project_path = Path(project_folder_str).resolve()
    # Use a combination of thread name and paper identifier for unique log project name
    paper_identifier = project_path.name # e-print_source or similar
    if project_path.parent.name != "e-print_source": # If deeper, use parent
        paper_identifier = project_path.parent.name

    log_project_name = paper_identifier.replace('.', '_')

    if thread_name:
        log_project_name = f"{thread_name}_{log_project_name}"
    
    # print(f"Debug: Log project name set to: {log_project_name}")


    converter = LatexToMarkdownConverter(
        folder_path_str=str(project_path),
        quiet_level=cli_args.quiet,
        max_workers=cli_args.max_workers,
        template_path=resolved_template_path_str,
        openalex_email=cli_args.openalex_email,
        openalex_api_key=cli_args.openalex_api_key,
        elsevier_api_key=cli_args.elsevier_api_key,
        springer_api_key=cli_args.springer_api_key,
        semantic_scholar_api_key=cli_args.semantic_scholar_api_key,
        poppler_path=cli_args.poppler_path,
        output_debug_tex=cli_args.output_debug_tex,
        log_project_name=log_project_name
    )
    success = False
    try:
        if converter.convert_to_markdown(output_folder_str):
            success = True
    except Exception as e:
        converter._log(f"Unexpected error in process_project_folder for {project_path.name}: {e}\n{traceback.format_exc(limit=2)}", "error")
    return success

def _process_single_paper_id_worker(paper_id_candidate, or_client, ar_client, decision_type, base_output_dir_str, converter_settings_dict, current_time_str, arxiv_version):
    thread_name = threading.current_thread().name
    # print(f"[{thread_name}] Worker started for OR ID: {paper_id_candidate}")
    try:
        note = or_client.get_note(id=paper_id_candidate)
        paper_title_original = note.content.get('title', {})
        if not isinstance(paper_title_original, str):
            paper_title_original = paper_title_original.get('value', '')
        paper_title_lower = paper_title_original.lower()

        if not paper_title_lower:
            # print(f"[{thread_name}] No title for OR ID {paper_id_candidate}, worker skipping.")
            return None

        search_query = f'ti:"{paper_title_original}"'
        search_results = list(ar_client.results(arxiv.Search(
            query=search_query, max_results=5, sort_by=arxiv.SortCriterion.Relevance
        )))

        if not search_results:
            # print(f"[{thread_name}] No arXiv results for title '{paper_title_original}' (OR ID: {paper_id_candidate})")
            return None

        for result in search_results:
            if result.title.lower() == paper_title_lower:
                # Get the base arXiv ID (without version) from the search result.
                base_arxiv_id = re.sub(r'v\d+$', '', result.get_short_id())

                # Construct the specific version ID. If arxiv_version is 0 or less,
                # we download the latest version by not specifying a version number in the URL.
                if arxiv_version > 0:
                    arxiv_id_with_version = f"{base_arxiv_id}v{arxiv_version}"
                else:
                    arxiv_id_with_version = base_arxiv_id # Omit version for latest

                # Create a filesystem-friendly name for the paper, replacing / and .
                arxiv_id_for_path = arxiv_id_with_version.replace('/', '_').replace('.', '_')
                
                # print(f"[{thread_name}] Exact title match: arXiv:{arxiv_id_with_version} for OR ID {paper_id_candidate}")

                paper_specific_subfolder_name = f"{paper_id_candidate}_{arxiv_id_for_path}"
                paper_processing_output_dir = Path(base_output_dir_str) / decision_type / paper_specific_subfolder_name
                
                # Construct the e-print URL for the specific version. The URL needs slashes.
                e_print_url = f"https://arxiv.org/e-print/{arxiv_id_with_version.replace('_', '/')}"

                # print(f"[{thread_name}] Instantiating LaTeX_Download for {e_print_url.split('/')[-1]}")
                latex_downloader_instance = LaTeX_Download(
                    url=e_print_url,
                    output_dir=str(paper_processing_output_dir),
                    quiet_converter=converter_settings_dict['quiet_converter'],
                    max_workers=converter_settings_dict['max_parallel_workers'],
                    template_path_converter=converter_settings_dict['template_path_converter'],
                    openalex_email_converter=converter_settings_dict['openalex_email_converter'],
                    openalex_api_key_converter=converter_settings_dict['openalex_api_key'],
                    elsevier_api_key_converter=converter_settings_dict['elsevier_api_key'],
                    springer_api_key_converter=converter_settings_dict['springer_api_key'],
                    semantic_scholar_api_key_converter=converter_settings_dict['semantic_scholar_api_key'],
                    poppler_path_converter=converter_settings_dict['poppler_path_converter'],
                    output_debug_tex_converter=converter_settings_dict['output_debug_tex_converter'],
                    thread_name=thread_name,
                )

                if latex_downloader_instance.download_and_parse():
                    # print(f"[+] SUCCESS: OR ID {paper_id_candidate} (arXiv:{arxiv_id_with_version}) processed by {thread_name}. Output: {paper_processing_output_dir}")
                    return paper_id_candidate
                else:
                    # print(f"[{thread_name}] Download/parse failed for arXiv:{arxiv_id_with_version} (OR ID {paper_id_candidate}).")
                    pass # Continue to next arXiv result if any
            # else:
                # print(f"  [{thread_name}] ArXiv Title: '{result.title.lower()}' != OR Title: '{paper_title_lower}' (OR ID: {paper_id_candidate})")


        # print(f"[{thread_name}] No successful exact match and parse for OR ID {paper_id_candidate} ('{paper_title_original}') after checking all arXiv results.")
        return None
    except Exception as e:
        print(f"[!] Worker Error for OR ID {paper_id_candidate} by {thread_name}: {e}\n{traceback.format_exc(limit=2)}", file=sys.stderr)
        return None

def download_arxiv_papers(accepted_ids, rejected_ids, client,
                          output_dir="parsed_papers",
                          required_accepted=None, required_rejected=None,
                          max_parallel_workers=10,
                          quiet_converter=0,
                          template_path_converter=DEFAULT_TEMPLATE_MD_PATH,
                          openalex_email_converter="ai-reviewer@example.com",
                          openalex_api_key=None,
                          elsevier_api_key=None, springer_api_key=None,
                          semantic_scholar_api_key=None, poppler_path_converter=None,
                          output_debug_tex_converter=False,
                          arxiv_version=1):
    downloaded_accepted = []
    downloaded_rejected = []
    arxiv_search_client = arxiv.Client()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    converter_settings = {
        'quiet_converter': quiet_converter,
        'max_parallel_workers': max_parallel_workers,
        'template_path_converter': template_path_converter,
        'openalex_email_converter': openalex_email_converter,
        'openalex_api_key': openalex_api_key,
        'elsevier_api_key': elsevier_api_key,
        'springer_api_key': springer_api_key,
        'semantic_scholar_api_key': semantic_scholar_api_key,
        'poppler_path_converter': poppler_path_converter,
        'output_debug_tex_converter': output_debug_tex_converter
    }

    def _process_papers_category_parallel(paper_id_list, target_count, decision_label, successful_list_target, executor_instance, arxiv_version):
        actual_target_count = target_count if target_count is not None else len(paper_id_list)
        if actual_target_count == 0: return

        # Ensure tqdm is used correctly for progress tracking
        pbar = tqdm(total=actual_target_count, desc=f"Processing {decision_label} papers", unit="paper", position=0, leave=True)
        
        futures_map = {}
        submitted_count = 0

        for or_id in paper_id_list:
            if len(successful_list_target) >= actual_target_count:
                break # Already collected enough successful ones
            if submitted_count >= actual_target_count * 2 and target_count is not None: # Limit submissions if target is set
                 break


            future = executor_instance.submit(
                _process_single_paper_id_worker,
                or_id, client, arxiv_search_client, decision_label,
                output_dir, converter_settings, run_timestamp_str,
                arxiv_version
            )
            futures_map[future] = or_id
            submitted_count +=1
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures_map):
            original_or_id = futures_map[future]
            try:
                result_paper_id = future.result()
                if result_paper_id:
                    if len(successful_list_target) < actual_target_count:
                        successful_list_target.append(result_paper_id)
                        # pbar.update(1)
            except Exception as exc:
                # pbar.update(1)
                print(f"[!] Exception processing OR ID {original_or_id}: {exc}", file=sys.stderr)
            pbar.update(1)
            completed_count += 1
            if len(successful_list_target) >= actual_target_count:
                # Cancel remaining futures if enough papers are processed
                # for f_cancel in futures_map:
                #     if not f_cancel.done(): f_cancel.cancel()
                break # Exit as_completed loop
        
        if pbar.n < pbar.total and len(successful_list_target) >= actual_target_count :
             pbar.update(pbar.total - pbar.n) # Ensure progress bar completes if target met early
        pbar.close()

        if len(successful_list_target) < actual_target_count:
            print(f"[!] Warning: Only {len(successful_list_target)} of {actual_target_count} target {decision_label} papers processed successfully.")


    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_workers, thread_name_prefix='PaperProc') as executor:
        print(f"Processing accepted papers (max_workers={max_parallel_workers})...")
        _process_papers_category_parallel(accepted_ids, required_accepted, "accepted", downloaded_accepted, executor, arxiv_version)

        print(f"\nProcessing rejected papers (max_workers={max_parallel_workers})...")
        _process_papers_category_parallel(rejected_ids, required_rejected, "rejected", downloaded_rejected, executor, arxiv_version)

    print(f"\n--- Parallel Download and Parsing Summary ---")
    print(f"Target accepted: {required_accepted if required_accepted is not None else 'all'}. Successfully processed: {len(downloaded_accepted)}")
    print(f"Target rejected: {required_rejected if required_rejected is not None else 'all'}. Successfully processed: {len(downloaded_rejected)}")
    return {"accepted": downloaded_accepted, "rejected": downloaded_rejected}


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Download and parse LaTeX papers from OpenReview/arXiv.")
    parser.add_argument("--venue", default="ICLR.cc/2025/Conference", help="OpenReview venue ID (e.g., ICLR.cc/2024/Conference)")
    parser.add_argument("--output_dir", default="parsed_papers_output", help="Base directory to save downloaded and parsed papers.")
    parser.add_argument("--num_accepted", type=int, default=None, help="Number of accepted papers to process (None for all).")
    parser.add_argument("--num_rejected", type=int, default=None, help="Number of rejected papers to process (None for all).")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of parallel workers.")
    parser.add_argument("--arxiv_version", type=int, default=1, help="The version of the arXiv paper to download (e.g., 1, 2). Use 0 for the latest version. Defaults to 1.")
    parser.add_argument("-q", "--quiet_converter", action="count", default=0, help="Suppress info messages.")
    parser.add_argument("--template_md", default=DEFAULT_TEMPLATE_MD_PATH, help="Path to Pandoc Markdown template.")
    parser.add_argument("--openalex_email", default="ai-reviewer@gmail.com", help="Email for OpenAlex API (politeness).")
    parser.add_argument("--openalex-api-key", default=os.environ.get("OPENALEX_API_KEY"), help="Your OpenAlex API key. Can also be set via OPENALEX_API_KEY environment variable.")
    parser.add_argument("--elsevier_api_key", default=DEFAULT_ELSEVIER_API_KEY, help="Elsevier API Key.")
    parser.add_argument("--springer_api_key", default=DEFAULT_SPRINGER_API_KEY, help="Springer API Key.")
    parser.add_argument("--semantic_scholar_api_key", default=DEFAULT_SEMANTIC_SCHOLAR_API_KEY, help="Semantic Scholar API Key.")
    parser.add_argument("--poppler_path", default=None, help="Path to Poppler binaries (if not in system PATH).")
    parser.add_argument("--output_debug_tex", action="store_true", help="Output intermediate .tex files for debugging converter.")
    parser.add_argument("--openreview_username", default=DEFAULT_OPENREVIEW_USERNAME, help="OpenReview username.")
    parser.add_argument("--openreview_password", default=DEFAULT_OPENREVIEW_PASSWORD, help="OpenReview password (will prompt if not provided).")
    parser.add_argument("--setup_pandoc_ver", type=str, default=None, help="Version of Pandoc to attempt to install (e.g., 3.1.9). Skips if Pandoc is found.")


    args = parser.parse_args()

    # Setup Pandoc if requested
    if args.setup_pandoc_ver:
        if not setup_pandoc(args.setup_pandoc_ver):
            print("[!] Pandoc setup failed. The script might not work correctly.")
            # Decide if to exit or continue:
            # sys.exit(1) # Exit if Pandoc is critical and setup fails
    elif not shutil.which("pandoc"):
        print("[!] Pandoc not found in PATH and no setup version specified. Please install Pandoc.")
        # sys.exit(1)


    # Create template.md if it doesn't exist
    template_md_path = Path(args.template_md)
    if not template_md_path.exists():
        print(f"[*] '{template_md_path}' not found. Creating default template.")
        try:
            with open(template_md_path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_TEMPLATE_MD_CONTENT)
            print(f"[+] Default template written to '{template_md_path}'")
        except IOError as e:
            print(f"[-] Error writing default template: {e}. Please ensure you have a template.md or provide a valid path.", file=sys.stderr)
            # sys.exit(1) # Exit if template is critical

    # OpenReview client setup
    openreview_username = args.openreview_username
    openreview_password = args.openreview_password

    if not openreview_username:
        openreview_username = input("Enter OpenReview Username: ")
    
    if not openreview_password:
        # Check for openreview_account.json as a fallback
        openreview_config_path = Path("openreview_account.json")
        if openreview_config_path.exists():
            try:
                with open(openreview_config_path, 'r') as f:
                    acc_details = json.load(f)
                openreview_username = acc_details.get("username", openreview_username)
                openreview_password = acc_details.get("password")
                print("[*] Loaded OpenReview credentials from openreview_account.json")
            except Exception as e:
                print(f"[!] Error reading openreview_account.json: {e}. Will prompt for password.")
        
        if not openreview_password: # Still no password
             openreview_password = getpass.getpass("Enter OpenReview Password: ")


    try:
        openreview_client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=openreview_username,
            password=openreview_password
        )
        print(f"[+] OpenReview client authenticated for user: {openreview_username}")
    except Exception as e:
        print(f"[-] Failed to connect to OpenReview: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Get correct client version for this venue
    client_version = "v2"
    api_venue_group = openreview_client.get_group(args.venue)
    api_venue_domain = api_venue_group.domain
    if api_venue_domain:
        print("This venue is available for OpenReview Client V2. Proceeding...")
    else:
        print("This venue is not available for OpenReview Client V2. Switching to Client V1...")
        openreview_client = openreview.Client(
            baseurl='https://api.openreview.net', 
            username=openreview_username, 
            password=openreview_password
        )
        client_version = "v1"

    # Get paper IDs from OpenReview
    print(f"[*] Fetching paper IDs for venue: {args.venue}")
    if client_version == "v2":
        try:
            venue_group = openreview_client.get_group(args.venue)
            # submission_name = venue_group.content['submission_name']['value'] # Not directly used later

            # Accepted papers
            accepted_submissions_iterator = openreview_client.get_all_notes(content={'venueid': args.venue}, details='direct')
            all_accepted_ids = [item.id for item in accepted_submissions_iterator]
            print(f"[*] Found {len(all_accepted_ids)} accepted papers.")

            # Rejected papers
            rejected_venue_id_key = venue_group.content.get('rejected_venue_id', {}).get('value')
            if rejected_venue_id_key:
                rejected_submissions_iterator = openreview_client.get_all_notes(content={'venueid': rejected_venue_id_key}, details='direct')
                all_rejected_ids = [item.id for item in rejected_submissions_iterator]
                print(f"[*] Found {len(all_rejected_ids)} rejected papers.")
            else:
                all_rejected_ids = []
                print("[!] 'rejected_venue_id' not found for this venue. Skipping rejected papers.")

        except Exception as e:
            print(f"[-] Error fetching data from OpenReview for venue {args.venue}: {e}", file=sys.stderr)
            sys.exit(1)
    elif client_version == "v1":
        submissions = openreview_client.get_all_notes(invitation = f'{args.venue}/-/Blind_Submission', details='directReplies')
        blind_notes = {note.id: note for note in submissions}
        all_decision_notes = []
        for submission_id, submission in blind_notes.items():
                all_decision_notes = all_decision_notes + [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("Decision")]
        all_accepted_ids = []
        all_rejected_ids = []

        for decision_note in all_decision_notes:
            if 'Accept' in decision_note["content"]['decision']:
                all_accepted_ids.append(decision_note['forum'])
            else:
                all_rejected_ids.append(decision_note['forum'])

    else:
        raise ValueError(f"Unknown client version: {client_version}")

    # Filter if --num_accepted or --num_rejected is set
    if args.num_accepted is not None and args.num_accepted < len(all_accepted_ids):
        all_accepted_ids = all_accepted_ids[:args.num_accepted]
        print(f"[*] Limiting to first {args.num_accepted} accepted papers.")
    if args.num_rejected is not None and args.num_rejected < len(all_rejected_ids):
        all_rejected_ids = all_rejected_ids[:args.num_rejected]
        print(f"[*] Limiting to first {args.num_rejected} rejected papers.")


    # Ensure API keys are populated for the converter settings
    elsevier_key = args.elsevier_api_key if args.elsevier_api_key else DEFAULT_ELSEVIER_API_KEY
    springer_key = args.springer_api_key if args.springer_api_key else DEFAULT_SPRINGER_API_KEY
    s2_key = args.semantic_scholar_api_key if args.semantic_scholar_api_key else DEFAULT_SEMANTIC_SCHOLAR_API_KEY

    # Run the download and parsing process
    results = download_arxiv_papers(
        accepted_ids=all_accepted_ids,
        rejected_ids=all_rejected_ids,
        client=openreview_client,
        output_dir=args.output_dir,
        required_accepted=args.num_accepted,
        required_rejected=args.num_rejected,
        max_parallel_workers=args.max_workers,
        quiet_converter=args.quiet_converter,
        template_path_converter=args.template_md,
        openalex_email_converter=args.openalex_email,
        openalex_api_key=args.openalex_api_key,
        elsevier_api_key=elsevier_key,
        springer_api_key=springer_key,
        semantic_scholar_api_key=s2_key,
        poppler_path_converter=args.poppler_path,
        output_debug_tex_converter=args.output_debug_tex,
        arxiv_version=args.arxiv_version
    )

    print("\n--- Script Finished ---")
    print(f"Results: {results}")
    print(f"Output saved in: {Path(args.output_dir).resolve()}")

if __name__ == "__main__":
    # IMPORTANT: You need to paste the FULL LatexToMarkdownConverter class definition
    # above where the placeholder indicates. Otherwise, this script will not run.
    # This script assumes the class is fully defined.
    if LatexToMarkdownConverter.__name__ == "LatexToMarkdownConverter" and not hasattr(LatexToMarkdownConverter, '_fully_process_bbl'):
         print("CRITICAL ERROR: The LatexToMarkdownConverter class is not fully defined in the script.", file=sys.stderr)
         print("Please paste the complete class definition from your notebook into the script.", file=sys.stderr)
         print("The script will now exit.", file=sys.stderr)
         sys.exit(1)
    main()
