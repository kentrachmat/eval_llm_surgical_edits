#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import subprocess # For running external commands (pdflatex, bibtex, pandoc)
import tempfile # For creating temporary files
from pathlib import Path
import argparse
import sys
import shutil # For copying files and checking for executables (shutil.which)
import urllib.parse # For decoding URL-encoded paths in Markdown
import requests # For Semantic Scholar & OpenAlex APIs
import time      # For politely pausing between API calls
import io        # For PDF processing
import xml.etree.ElementTree as ET # For Elsevier API
import json # For Springer API

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
    Comment = None # Ensure Comment is also None if bs4 is not available

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

class LatexToMarkdownConverter:
    def __init__(self, folder_path_str, quiet_level=0, max_workers=1, template_path=None,
                 openalex_email="your-email@example.com", openalex_api_key = None, elsevier_api_key=None, springer_api_key=None, semantic_scholar_api_key=None,
                 poppler_path=None, output_debug_tex=False, log_project_name=None, ): # Added log_project_name
        self.folder_path = Path(folder_path_str).resolve()
        self.main_tex_path = None
        self.original_main_tex_content = ""
        self.quiet_level = quiet_level
        if self.quiet_level == 0:
            self.verbose = True
        else:
            self.verbose = False
        self.max_workers = max_workers
        self.template_path = template_path
        self.final_output_folder_path = None # Will be set in convert_to_markdown
        self.openalex_email = openalex_email
        self.openalex_api_key = openalex_api_key
        self.elsevier_api_key = elsevier_api_key
        self.springer_api_key = springer_api_key
        self.PdfReader = PdfReader
        self.BeautifulSoup = BeautifulSoup
        self.Comment = Comment
        self.PILImage = PILImage
        self.convert_from_path = convert_from_path
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.poppler_path = poppler_path
        self.output_debug_tex = output_debug_tex # Store the flag
        self.log_project_name = log_project_name # Store for logging context

        # Dependency checks and logs
        if self.PdfReader is None and self.verbose:
            self._log("PyPDF2 library not found. PDF parsing for abstracts will be disabled. Install with: pip install PyPDF2", "warn")
        if self.BeautifulSoup is None and self.verbose:
            self._log("BeautifulSoup4 library not found. HTML parsing for abstracts will be disabled. Install with: pip install beautifulsoup4", "warn")
        if self.PILImage is None and self.verbose:
            self._log("Pillow (PIL) library not found. Image conversion to PNG will be disabled. Install with: pip install Pillow", "warn")
        if self.convert_from_path is None and self.verbose:
            self._log("pdf2image library not found. PDF to PNG conversion using this library will be disabled. Install with: pip install pdf2image", "warn")
        if self.poppler_path and self.verbose:
            self._log(f"Using Poppler path: {self.poppler_path}", "info")
        elif self.convert_from_path and self.verbose:
            self._log("Poppler path not specified. pdf2image will try to find Poppler in system PATH. If PDF conversion fails, try providing --poppler-path.", "info")

        if self.openalex_email == "your-email@example.com" and self.verbose:
            self._log("OpenAlex email is set to default. For responsible API use, please provide your email via --openalex-email.", "warn")
        if not self.elsevier_api_key and self.verbose:
            self._log("Elsevier API key not provided. Abstract fetching from linkinghub.elsevier.com via API will be disabled.", "info")
        if not self.springer_api_key and self.verbose:
            self._log("Springer API key not provided. Abstract fetching from link.springer.com via API will be disabled.", "info")

    def _log(self, message, level="info"):
        if self.quiet_level >=2: return
        
        prefix_str = f"[{self.log_project_name}] " if self.log_project_name else ""
        if level == "error": print(f"{prefix_str}[-] Error: {message}", file=sys.stderr)
        elif level == "warn": print(f"{prefix_str}[!] Warning: {message}", file=sys.stderr)
        elif self.verbose:
            # Maintain original debug indentation if present, apply prefix
            if level == "debug" and message.startswith("    [*]"):
                 print(f"{prefix_str}{message}")
            elif level == "debug":
                 print(f"{prefix_str}    [*] {message}")
            elif level == "info": print(f"{prefix_str}[*] {message}")
            elif level == "success": print(f"{prefix_str}[+] {message}")

    def _write_debug_tex_if_needed(self, tex_content, base_filename="l2m_debug.tex"):
        # Writes debug TeX file if self.output_debug_tex is True.
        # Prefers self.final_output_folder_path, falls back to self.folder_path.
        if not self.output_debug_tex:
            return

        output_dir_candidate = self.final_output_folder_path if self.final_output_folder_path else self.folder_path
        
        if not output_dir_candidate:
            self._log(f"Cannot write {base_filename}: no valid output/project directory determined yet.", "warn")
            return

        try: # Ensure the chosen directory exists
            output_dir_candidate.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            self._log(f"Could not create directory {output_dir_candidate} for {base_filename}: {e_mkdir}", "warn")
            return

        debug_tex_file_path = output_dir_candidate / base_filename
        try:
            self._log(f"Writing TeX content for debugging to: {debug_tex_file_path}", "debug")
            with open(debug_tex_file_path, "w", encoding="utf-8") as debug_file:
                debug_file.write(tex_content)
            self._log(f"Successfully wrote {base_filename} to {debug_tex_file_path}", "info")
        except Exception as e_debug:
            self._log(f"Could not write {base_filename} to '{debug_tex_file_path}': {e_debug}", "warn")

    def _cleanup_debug_files(self):
        # Cleans up all l2m_debug_*.tex files if self.output_debug_tex is False.
        if self.output_debug_tex:
            return

        self._log("Cleaning up L2M debug TeX files...", "debug")
        patterns_to_delete = ["l2m_debug_*.tex"]
        folders_to_check = []
        if self.final_output_folder_path and self.final_output_folder_path.exists():
            folders_to_check.append(self.final_output_folder_path)
        # Also check project folder, in case output folder wasn't created or is different
        if self.folder_path and self.folder_path.exists() and \
           (not self.final_output_folder_path or self.folder_path != self.final_output_folder_path):
            folders_to_check.append(self.folder_path)

        for folder in folders_to_check:
            for pattern in patterns_to_delete:
                for f_path in folder.glob(pattern):
                    try:
                        f_path.unlink()
                        self._log(f"Deleted debug file: {f_path}", "debug")
                    except Exception as e_del:
                        self._log(f"Could not delete debug file {f_path}: {e_del}", "warn")
    
    # def _parse_pandoc_error_line_number(self, stderr_output: str) -> int | None:
    #     # Parses Pandoc's stderr for "Error at "source" (line X, ...)"
    #     # Returns the 1-indexed line number if found, otherwise None.
    #     match = re.search(r'Error at "source" \(line (\d+),', stderr_output)
    #     if match:
    #         try:
    #             line_num = int(match.group(1))
    #             self._log(f"Parsed Pandoc error: problem at line {line_num}.", "debug")
    #             return line_num
    #         except ValueError:
    #             self._log(f"Could not parse line number from Pandoc error string: '{match.group(1)}'", "warn")
    #     return None
    
    def _parse_pandoc_error_line_number(self, stderr_output: str) -> int | None:
        # Tries to parse Pandoc's stderr for error line numbers.
        # Iterates through a list of patterns, from most critical/specific to informational.
        # Returns the 1-indexed line number if found, otherwise None.

        patterns = [
            {
                'name': 'PandocSpecificFileError', # e.g., Error at "tmpXYZ.tex" (line L, column C):
                'regex': r'Error at "(?P<filename>[^"]+\.tex)" \(line (?P<line_num>\d+), column \d+\):',
                'log_message': "Parsed Pandoc specific file error: problem at line {line_num} in '{filename}'."
            },
            {
                'name': 'PandocGenericSourceErrorWithColumn', # e.g., Error at "source" (line L, column C):
                'regex': r'Error at "source" \(line (?P<line_num>\d+), column \d+\):',
                'log_message': "Parsed Pandoc generic 'source' error: problem at line {line_num}."
            },
            {
                'name': 'PandocGenericSourceError', # e.g., Error at "source" (line L,
                'regex': r'Error at "source" \(line (?P<line_num>\d+),',
                'log_message': "Parsed Pandoc generic 'source' error (legacy format): problem at line {line_num}."
            },
            {
                'name': 'PandocIncludeInfoError', # e.g., [INFO] Could not load include file ... at tmpXYZ.tex line L column C
                                                  # or [INFO] Could not load include file ... at source line L column C
                'regex': r'\[INFO\] Could not load include file .*? at (?P<filename>[^ ]+\.tex|source) line (?P<line_num>\d+) column \d+',
                'log_message': "Parsed Pandoc info: problem loading include at line {line_num} in '{filename}'."
            }
            # Add more patterns here if needed, in order of priority
        ]

        for p_info in patterns:
            match = re.search(p_info['regex'], stderr_output)
            if match:
                try:
                    line_num = int(match.group("line_num"))
                    filename_detail = match.groupdict().get('filename', 'unknown_file')
                    # We assume the line number is relevant to the main temporary file Pandoc processes.
                    # Pandoc's error messages usually give line numbers relative to the input file it's processing.
                    log_msg = p_info['log_message'].format(line_num=line_num, filename=filename_detail)
                    self._log(f"Line number extraction ({p_info['name']}): {log_msg}", "debug")
                    return line_num
                except ValueError:
                    self._log(f"Could not parse line number as int from Pandoc error string ({p_info['name']}): '{match.group('line_num')}'", "warn")
                except IndexError: # Should not happen if regex has "line_num" group
                    self._log(f"Regex for '{p_info['name']}' is missing 'line_num' group.", "warn")

        self._log(f"Could not find any known error line pattern in Pandoc stderr for line number extraction.", "debug")
        return None

    # --- (All other helper methods like find_main_tex_file, _comment_out_style_packages, _generate_bbl_content, etc. remain here) ---
    # --- (Make sure they are complete and correct as in the original provided script) ---
    def find_main_tex_file(self):
        self._log("Finding main .tex file...")
        for path in self.folder_path.rglob("*.tex"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
                if r'\begin{document}' in content:
                    self.main_tex_path, self.original_main_tex_content = path, content
                    self._log(f"Main .tex file found: {self.main_tex_path}", "success"); return True
            except Exception as e: self._log(f"Could not read file {path} due to {e}", "warn")
        self._log(f"No main .tex file with \\begin{{document}} found in '{self.folder_path}'.", "error"); return False

    def _get_project_sty_basenames(self):
        sty_basenames = []
        common_styles_to_ignore = [
            "article", "report", "book", "letter", "amsmath", "amssymb", "amsthm", "amsfonts", 
            "graphicx", "graphics", "xcolor", "color", "geometry", "fancyhdr", "hyperref", "url",
            "inputenc", "fontenc", "babel", "textcomp", "listings", "minted", "enumitem", "parskip",
            "setspace", "emptypage", "tocbibind", "titling", "titlesec", "etoolbox", "xparse",
            "expl3", "l3keys2e", "iftex", "ifluatex", "ifxetex", "natbib", "biblatex", "microtype",
            "soul", "ulem", "booktabs", "multirow", "multicol", "tabularx", "longtable", "array",
            "float", "caption", "subcaption", "subfigure", "threeparttable", "makecell", "colortbl",
            "xspace", "ragged2e", "calc", "refstyle", "varioref", "wasysym", "marvosym", "pifont",
            "algorithmicx", "algorithm", "algpseudocode", "algorithms", "tcolorbox", "framed", "mdframed",
            "trimspaces", "environ", "keyval", "kvoptions", "pgfopts", "pdfpages", "epstopdf", "grffile",
            "trig", "substr", "xstring", "comment", "verbatim", "nameref", "gettitlestring",
            "times", "helvet", "courier", "mathptmx", "newtxtext", "newtxmath", "sourcesanspro",
            "sourcecodepro", "sourceserifpro", "lmodern", "bera", "appendix", "cleveref", "csquotes",
            "standalone", "todonotes", "wrapfig", "subfiles", "import", "authblk"
        ]
        for sty_path in self.folder_path.rglob("*.sty"):
            try:
                relative_sty_path = sty_path.relative_to(self.folder_path)
            except ValueError:
                relative_sty_path = Path(sty_path.name)
            sty_identifier_with_path = str(relative_sty_path.with_suffix(""))
            if sty_path.stem.lower() not in common_styles_to_ignore:
                sty_basenames.append(sty_identifier_with_path)
        if sty_basenames:
            self._log(f"Identified project-specific/non-standard .sty files for potential commenting: {sty_basenames}", "debug")
        return sty_basenames

    def _comment_out_style_packages(self, tex_content: str, mode: str = "venue_only") -> str:
        modified_content = tex_content
        initial_content = tex_content
        if mode == "venue_only":
            patterns_to_process = VENUE_STYLE_PATTERNS
            target_regex_flags = re.IGNORECASE
            main_sub_flags = re.MULTILINE | re.IGNORECASE
        elif mode == "all_project":
            project_sty_paths = self._get_project_sty_basenames()
            if not project_sty_paths:
                self._log("No project-specific .sty files to comment out for 'all_project' mode.", "debug")
                return tex_content
            patterns_to_process = [re.escape(sty_path) for sty_path in project_sty_paths]
            target_regex_flags = 0
            main_sub_flags = re.MULTILINE | re.IGNORECASE
        else:
            self._log(f"Invalid mode '{mode}' for _comment_out_style_packages.", "error")
            return tex_content

        general_line_pattern = re.compile(
            r"^([^\%]*?)(\\usepackage(?:\[[^\]]*\])?\{([^{}]*)\}[^\n]*)$",
            main_sub_flags
        )
        for core_style_pattern_str in patterns_to_process:
            current_target_pkg_regex = re.compile(
                r"^(?:[a-zA-Z0-9_\-\./]+/)?(?:" + core_style_pattern_str + r")$",
                target_regex_flags
            )
            def replacement_evaluator(match_obj: re.Match) -> str:
                leading_text_on_line = match_obj.group(1)
                full_usepackage_command = match_obj.group(2)
                brace_content_string = match_obj.group(3)
                declared_packages = [pkg.strip() for pkg in brace_content_string.split(',')]
                for package_candidate in declared_packages:
                    if not package_candidate: continue
                    if current_target_pkg_regex.fullmatch(package_candidate):
                        self._log(f"Commenting out line due to matching package '{package_candidate}' with pattern '{core_style_pattern_str}' (mode: {mode})", "debug")
                        return f"{leading_text_on_line}% {full_usepackage_command}"
                return match_obj.group(0)
            modified_content = general_line_pattern.sub(replacement_evaluator, modified_content)
        if modified_content != initial_content:
            self._log(f"Commented out relevant \\usepackage commands (mode: {mode}).", "debug")
        else:
            self._log(f"No \\usepackage commands were modified for commenting (mode: {mode}).", "debug")
        return modified_content

    def _generate_bbl_content(self):
        if not self.main_tex_path:
            self._log("Cannot generate .bbl: Main .tex not identified.", "error"); return None
        main_file_stem = self.main_tex_path.stem
        specific_bbl_path = self.folder_path / f"{main_file_stem}.bbl"
        raw_bbl_content_to_process = None

        if specific_bbl_path.exists():
            self._log(f"Found existing specific .bbl file: {specific_bbl_path}", "debug")
            try:
                content = specific_bbl_path.read_text(encoding="utf-8", errors="ignore")
                if content.startswith(PROCESSED_BBL_MARKER + "\n"):
                    self._log("Using cached fully processed specific .bbl file.", "success")
                    return content.split(PROCESSED_BBL_MARKER + "\n", 1)[1]
                else:
                    self._log("Existing specific .bbl found but not marked as processed. Will use its raw content.", "info")
                    raw_bbl_content_to_process = content
            except Exception as e:
                self._log(f"Error reading existing specific .bbl file '{specific_bbl_path}': {e}. Will attempt generation/fallback.", "warn")
                raw_bbl_content_to_process = None
        
        if raw_bbl_content_to_process is None: # BBL not found or not cached or error reading cache
            self._log(f"Attempting to generate specific .bbl file: {specific_bbl_path.name}...", "info")
            bibtex_run_successful = False
            try:
                commands = [
                    ["pdflatex", "-interaction=nonstopmode", "-draftmode", self.main_tex_path.name],
                    ["bibtex", main_file_stem]
                ]
                for i, cmd_args in enumerate(commands):
                    self._log(f"Running BBL generation command: {' '.join(cmd_args)} in {self.folder_path}", "debug")
                    # Use cwd parameter instead of os.chdir
                    process = subprocess.run(cmd_args, capture_output=True, text=True, check=False, 
                                             encoding='utf-8', errors='ignore', cwd=self.folder_path)
                    if process.returncode != 0:
                        self._log(f"Error running {' '.join(cmd_args)}. STDERR: {process.stderr[:500]} STDOUT: {process.stdout[:500]}", "error")
                        if i == 0 and "pdflatex" in cmd_args[0]: 
                            self._log("Initial pdflatex run failed for BBL generation.", "error")
                        bibtex_run_successful = False
                        if specific_bbl_path.exists(): # Attempt to clean up potentially corrupt BBL
                            try: specific_bbl_path.unlink()
                            except Exception as e_unlink: self._log(f"Could not remove BBL file {specific_bbl_path} after error: {e_unlink}", "warn")
                        break 
                else: # No break means all commands in loop succeeded (or at least didn't error out in a way we catch here)
                    bibtex_run_successful = True # Assume success if loop completes

                if bibtex_run_successful and specific_bbl_path.exists():
                    self._log(f"Specific .bbl file generated successfully: {specific_bbl_path}", "debug")
                    raw_bbl_content_to_process = specific_bbl_path.read_text(encoding="utf-8", errors="ignore")
                elif bibtex_run_successful: # BibTeX ran but no BBL, maybe no citations
                    self._log(f"BibTeX ran but did not create '{specific_bbl_path.name}'. This might be normal if there are no citations.", "info")
                    raw_bbl_content_to_process = "" # Treat as empty BBL
                else: # BibTeX run failed or pdflatex before it failed
                    self._log(f"BibTeX run did not complete successfully for '{specific_bbl_path.name}'.", "warn")
                    raw_bbl_content_to_process = None


            except Exception as e:
                self._log(f"Exception during .bbl generation: {e}", "error")
                raw_bbl_content_to_process = None # Ensure it's None on exception
        
        # Fallback to any .bbl if specific one is still not available
        if raw_bbl_content_to_process is None:
            self._log(f"Specific .bbl file '{specific_bbl_path.name}' not found or generated. Searching for any .bbl file in '{self.folder_path}'...", "info")
            bbl_files_in_folder = list(self.folder_path.glob("*.bbl"))
            if bbl_files_in_folder:
                alternative_bbl_path = bbl_files_in_folder[0]
                # Prefer the one matching main_tex_stem if it exists among them, even if generation failed
                # (This part is already covered by specific_bbl_path logic if it exists)
                # For truly alternative, just pick the first one found.
                self._log(f"Found alternative .bbl file: '{alternative_bbl_path.name}'. Using its raw content.", "info")
                try:
                    raw_bbl_content_to_process = alternative_bbl_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    self._log(f"Error reading alternative .bbl file '{alternative_bbl_path.name}': {e}", "warn")
                    raw_bbl_content_to_process = None # Set to None if read fails
            else:
                self._log("No .bbl files (specific or alternative) found in the project folder. Proceeding without bibliography.", "warn")
                raw_bbl_content_to_process = "" # Treat as empty BBL if none found at all

        if raw_bbl_content_to_process is not None: # This means we have some BBL content (even if empty string)
            self._log("Starting full BBL processing (abstracts, keys) for caching...", "info")
            processed_bbl_string = self._fully_process_bbl(raw_bbl_content_to_process)
            try:
                # Cache even if raw_bbl_content_to_process was empty, to mark it as "processed"
                with open(specific_bbl_path, "w", encoding="utf-8") as f:
                    f.write(PROCESSED_BBL_MARKER + "\n")
                    f.write(processed_bbl_string)
                self._log(f"Successfully wrote processed BBL content to '{specific_bbl_path}' (cached).", "success")
            except Exception as e_write:
                self._log(f"Error writing processed BBL to cache file '{specific_bbl_path}': {e_write}", "warn")
            return processed_bbl_string
        
        self._log("Failed to obtain BBL content through any method.", "error")
        return None # Explicitly return None if all attempts fail
    def _latex_escape_abstract(self, text: str) -> str:
        if not text: return ""
        text = text.replace('\\', r'\textbackslash{}')
        text = text.replace('{', r'\{'); text = text.replace('}', r'\}')
        text = text.replace('&', r'\&'); text = text.replace('%', r'\%')
        text = text.replace('$', r'\$'); text = text.replace('#', r'\#')
        text = text.replace('_', r'\_'); text = text.replace('~', r'\textasciitilde{}')
        text = text.replace('^', r'\textasciicircum{}')
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _deinvert_abstract_openalex(self, inverted_index):
        if not inverted_index: return None
        max_pos = -1
        for positions in inverted_index.values():
            if positions: max_pos = max(max_pos, max(positions))
        if max_pos == -1: return ""
        length = max_pos + 1; abstract_list = [""] * length
        for word, positions in inverted_index.items():
            for pos in positions:
                if 0 <= pos < length: abstract_list[pos] = word
        return " ".join(abstract_list).strip()

    def _extract_text_from_pdf_content(self, pdf_content):
        if not self.PdfReader: return None
        try:
            reader = self.PdfReader(io.BytesIO(pdf_content)); text = ""
            for page in reader.pages: text += page.extract_text() or ""
            return text
        except Exception as e: self._log(f"PDF text extraction error: {e}", "warn"); return None

    def _find_abstract_in_pdf_text(self, text, max_chars_abstract=2500):
        if not text: return None
        text_lower = text.lower()
        abstract_match = re.search(r'\b(a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\b\.?\s*\n?', text_lower, re.IGNORECASE)
        if abstract_match:
            start_index = abstract_match.end()
            end_patterns = [r'\n\s*(keywords|key words|index terms)\b', r'\n\s*(1|I)\.\s*(Introduction|INTRODUCTION)\b', r'\n\s*(Introduction|INTRODUCTION)\b', r'\n\s*\n\s*\n']
            end_index = len(text)
            text_after_abstract_keyword = text[start_index:]
            for pattern in end_patterns:
                match = re.search(pattern, text_after_abstract_keyword, re.IGNORECASE)
                if match and match.start() < (end_index - start_index) : end_index = start_index + match.start()
            potential_abstract = text[start_index:min(end_index, start_index + max_chars_abstract)].strip()
            potential_abstract = re.sub(r'\s+', ' ', potential_abstract)
            if len(potential_abstract) > 50: self._log("PDF Abstract: Found abstract.", "debug"); return potential_abstract
        return None

    def _get_elsevier_abstract_from_linking_url(self, linking_hub_url: str, api_key: str) -> str | None:
        pii_match = re.search(r'/pii/([^/?]+)', linking_hub_url)
        if not pii_match:
            self._log(f"Elsevier API: Could not extract PII from URL: {linking_hub_url}", "warn")
            return "❌ Error: Could not extract PII from the provided URL."
        pii = pii_match.group(1)
        self._log(f"Elsevier API: Extracted PII: {pii} from URL: {linking_hub_url}", "debug")
        api_url = f"https://api.elsevier.com/content/article/pii/{pii}"
        headers = {'X-ELS-APIKey': api_key, 'Accept': 'application/xml'}
        self._log(f"Elsevier API: Constructed API URL: {api_url}", "debug")
        max_retries = 3; retry_delay_seconds = 2; server_error_codes = [500, 502, 503, 504]
        for attempt in range(max_retries):
            try:
                response = requests.get(api_url, headers=headers, timeout=30)
                response.raise_for_status()
                self._log(f"Elsevier API: Request successful (Attempt {attempt + 1}/{max_retries}).", "debug")
                try:
                    root = ET.fromstring(response.content)
                    namespaces = {'svapi': 'http://www.elsevier.com/xml/svapi/article/dtd', 'dc': 'http://purl.org/dc/elements/1.1/'}
                    coredata = root.find('svapi:coredata', namespaces)
                    if coredata is not None:
                        desc = coredata.find('dc:description', namespaces)
                        if desc is not None and desc.text:
                            self._log("Elsevier API: Abstract extracted.", "success")
                            return ' '.join(desc.text.strip().split())
                    self._log("Elsevier API: Abstract not found in XML.", "debug")
                    return "❌ Abstract not found in XML."
                except Exception as e_xml:
                    self._log(f"Elsevier API: XML parsing error: {e_xml}. Resp: {response.text[:200]}...", "warn")
                    return f"❌ XML parsing error: {e_xml}. Resp: {response.text[:200]}..."
            except requests.exceptions.HTTPError as http_err:
                self._log(f"Elsevier API: HTTP error (Attempt {attempt + 1}/{max_retries}): {http_err} - Status: {http_err.response.status_code}", "warn")
                if http_err.response.status_code in server_error_codes:
                    if attempt < max_retries - 1:
                        self._log(f"Elsevier API: Retrying in {retry_delay_seconds}s...", "info"); time.sleep(retry_delay_seconds); continue
                    else: self._log("Elsevier API: Max retries reached for server error.", "error"); return f"❌ HTTP error after retries: {http_err} - Status: {http_err.response.status_code}"
                else: return f"❌ HTTP error: {http_err} - Status: {http_err.response.status_code} - Resp: {http_err.response.text[:200]}..."
            except requests.exceptions.RequestException as req_err:
                self._log(f"Elsevier API: Request error (Attempt {attempt + 1}/{max_retries}): {req_err}", "warn")
                if attempt < max_retries - 1:
                    self._log(f"Elsevier API: Retrying in {retry_delay_seconds}s...", "info"); time.sleep(retry_delay_seconds); continue
                else: self._log("Elsevier API: Max retries reached for request exception.", "error"); return f"❌ Request error after retries: {req_err}"
            break
        return "❌ Elsevier API: Failed after all retries or due to unexpected issue."

    def _get_springer_abstract_from_url(self, springer_url: str, api_key: str) -> str | None:
        doi_match = re.search(r'/(?:chapter|article|book)/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', springer_url, re.IGNORECASE)
        if not doi_match:
            self._log(f"Springer API: Could not extract DOI from URL: {springer_url}", "warn")
            return "❌ Error: Could not extract DOI from the provided URL."
        doi = doi_match.group(1)
        self._log(f"Springer API: Extracted DOI: {doi} from URL: {springer_url}", "debug")
        api_url = "https://api.springernature.com/meta/v2/json"
        params = {'q': f'doi:{doi}', 'api_key': api_key}
        self._log(f"Springer API: Constructed API URL: {api_url} with params: {params}", "debug")
        try:
            response = requests.get(api_url, params=params, timeout=20)
            response.raise_for_status()
            self._log("Springer API: API request successful.", "debug")
        except requests.exceptions.HTTPError as http_err:
            self._log(f"Springer API: HTTP error: {http_err} - Status: {response.status_code} - Resp: {response.text[:200]}...", "warn")
            return f"❌ HTTP error: {http_err} - Status: {response.status_code} - Resp: {response.text[:200]}..."
        except requests.exceptions.RequestException as err:
            self._log(f"Springer API: Request error: {err}", "warn"); return f"❌ Request error: {err}"
        try:
            data = response.json()
            if data.get("records") and len(data["records"]) > 0:
                first_record = data["records"][0]
                if "abstract" in first_record and first_record["abstract"]:
                    self._log("Springer API: Abstract extracted.", "success"); return first_record["abstract"].strip()
            total_results = data.get("result", [{}])[0].get("total", "unknown")
            if str(total_results) == "0":
                self._log(f"Springer API: No records found for DOI: {doi}. API returned 0 results.", "debug")
                return f"❌ No records found for DOI: {doi}. The API returned 0 results."
            self._log("Springer API: Abstract not found in API response.", "debug")
            return "❌ Abstract field not found or empty in API response."
        except Exception as e:
            self._log(f"Springer API: JSON parsing error: {e}. Resp: {response.text[:200]}...", "warn")
            return f"❌ JSON parsing error: {e}. Resp: {response.text[:200]}..."

    def _get_arxiv_abstract_from_page(self, arxiv_url: str) -> str | None:
        if not self.BeautifulSoup:
            self._log("arXiv Page Fetch: BeautifulSoup library not available. Cannot parse HTML.", "warn"); return None
        original_url_for_logging = arxiv_url
        if "arxiv.org/pdf/" in arxiv_url:
            arxiv_url = arxiv_url.replace("/pdf/", "/abs/").replace(".pdf", "")
            self._log(f"arXiv Page Fetch: Converted PDF URL '{original_url_for_logging}' to abstract URL '{arxiv_url}'.", "debug")
        elif not arxiv_url.startswith("https://arxiv.org/abs/"):
            self._log(f"arXiv Page Fetch: URL '{original_url_for_logging}' is not a standard arXiv abstract or PDF URL. Proceeding cautiously.", "warn")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            response = requests.get(arxiv_url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self._log(f"arXiv Page Fetch: Error fetching URL {arxiv_url}: {e}", "warn"); return None
        try:
            soup = self.BeautifulSoup(response.content, 'html.parser')
            meta_abstract = soup.find('meta', attrs={'name': 'citation_abstract'})
            if meta_abstract and meta_abstract.get('content'):
                self._log(f"arXiv Page Fetch: Found abstract in 'citation_abstract' meta tag for {arxiv_url}.", "debug")
                return meta_abstract.get('content').strip()
            meta_og_description = soup.find('meta', property='og:description')
            if meta_og_description and meta_og_description.get('content'):
                self._log(f"arXiv Page Fetch: Found abstract in 'og:description' meta tag for {arxiv_url}.", "debug")
                return meta_og_description.get('content').strip()
            blockquote_abstract = soup.find('blockquote', class_='abstract')
            if blockquote_abstract:
                abstract_text_content = blockquote_abstract.get_text(separator=' ', strip=True)
                cleaned_abstract = re.sub(r'^\s*Abstract:?\s*', '', abstract_text_content, flags=re.IGNORECASE).strip()
                if cleaned_abstract:
                    self._log(f"arXiv Page Fetch: Found abstract in blockquote.abstract for {arxiv_url}.", "debug")
                    return cleaned_abstract
            self._log(f"arXiv Page Fetch: Could not find abstract on page: {arxiv_url} using known meta tags or blockquote.", "debug")
            return None
        except Exception as e:
            self._log(f"arXiv Page Fetch: Error parsing HTML content from {arxiv_url}: {e}", "warn"); return None

    def _fetch_and_parse_html_for_abstract(self, url: str):
        if not self.BeautifulSoup or not self.Comment:
            self._log("HTML Parsing: BeautifulSoup or Comment not available.", "warn"); return None
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            if not url.startswith("file:///"):
                self._log(f"HTML Parsing: Invalid URL scheme: {url}", "debug"); return None
        self._log(f"HTML Parsing: Attempting to fetch from: {url}", "debug")
        effective_url = url
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; LatexToMarkdownConverter/1.3; +http://example.com/bot)'}
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            effective_url = response.url
            if url.startswith("file:///"): effective_url = url
            self._log(f"HTML Parsing: Effective URL after redirects: {effective_url}", "debug")
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                self._log(f"HTML Parsing: Content from {effective_url} not HTML (type: {content_type}).", "debug"); return None
            soup = self.BeautifulSoup(response.content, 'html.parser')
            abstract_text_candidate = None
            # (Rest of the HTML parsing logic from the original script)
            # Check for Springer API if it's a Springer link and HTML parsing fails initially
            if not abstract_text_candidate:
                if self.springer_api_key and "link.springer.com/" in effective_url:
                    api_abstract = self._get_springer_abstract_from_url(effective_url, self.springer_api_key)
                    if api_abstract and not api_abstract.startswith("❌"):
                        self._log(f"Springer API: Successfully fetched abstract from {effective_url} after HTML parse failed.", "success")
                        return api_abstract
                    elif api_abstract: self._log(f"Springer API: Attempt after HTML parse fail for {effective_url} also failed: {api_abstract}", "warn")
                    else: self._log(f"Springer API: Attempt after HTML parse fail for {effective_url} returned None.", "warn")

            # MHTML-like comment extraction
            if not abstract_text_candidate:
                self._log(f"HTML Abstract: Trying MHTML-like comment-based extraction for {effective_url}", "debug")
                abstract_start_comment = soup.find(string=lambda text: isinstance(text, self.Comment) and "Abstract." in text and "<!-- Abstract." in text and not "/ Abstract." in text)
                if abstract_start_comment:
                    self._log(f"HTML Abstract: Found '' comment in {effective_url}", "debug")
                    possible_outer_divs = soup.find_all('div', class_='columns is-centered has-text-centered')
                    for outer_div in possible_outer_divs:
                        column_div = outer_div.find('div', class_='column is-four-fifths')
                        if column_div:
                            h1_abstract = column_div.find('h1', class_='title is-3', string=re.compile(r'^\s*Abstract\s*$', re.I))
                            content_div = column_div.find('div', class_='content has-text-justified')
                            if h1_abstract and content_div and h1_abstract.find_next_sibling('div', class_='content has-text-justified') == content_div:
                                texts = []
                                for element in content_div.find_all(['p', 'ul'], recursive=False):
                                    if element.name == 'p': texts.append(element.get_text(separator=' ', strip=True))
                                    elif element.name == 'ul':
                                        for li_item in element.find_all('li', recursive=False): texts.append("- " + li_item.get_text(separator=' ', strip=True))
                                extracted_abstract = "\n".join(texts)
                                if extracted_abstract.strip():
                                    self._log(f"HTML Abstract: Extracted using MHTML-like comment structure from {effective_url}.", "success")
                                    return extracted_abstract
                else:
                    self._log(f"HTML Abstract: MHTML-like '' comment not found in {effective_url}", "debug")
            
            # Standard "Abstract" heading search
            if not abstract_text_candidate:
                abstract_heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'div'], string=re.compile(r'^\s*Abstract\s*$', re.I))
                if abstract_heading:
                    # (ACL style and general sibling logic as in original)
                    acl_style_element = None
                    if "acl-abstract" in " ".join(abstract_heading.get('class', [])): acl_style_element = abstract_heading
                    elif abstract_heading.parent and "acl-abstract" in " ".join(abstract_heading.parent.get('class', [])): acl_style_element = abstract_heading.parent
                    # ... (rest of ACL specific logic)
                    if acl_style_element:
                        abs_span = acl_style_element.find('span')
                        if abs_span:
                            text = abs_span.get_text(separator=' ', strip=True)
                            if len(text) > 70: self._log("HTML Abstract: Extracted from ACL-style span.", "debug"); return text
                        abs_p = acl_style_element.find('p')
                        if abs_p:
                            text = abs_p.get_text(separator=' ', strip=True)
                            if len(text) > 70: self._log("HTML Abstract: Extracted from ACL-style p.", "debug"); return text
                    
                    # General sibling content extraction
                    next_content_node = abstract_heading.find_next_sibling()
                    # ... (logic for div/section and general siblings)
                    for _ in range(3): # Limit search depth
                        if not next_content_node: break
                        if next_content_node.name in ['div', 'section'] and \
                           ((hasattr(next_content_node, 'attrs') and any(c in " ".join(next_content_node.attrs.get('class',[])) for c in ['content', 'abstract-content', 'entry-content', 'abstract__content'])) or \
                           not hasattr(next_content_node, 'attrs') or not next_content_node.attrs.get('class', None)):
                            collected_texts = []
                            for child_el in next_content_node.find_all(['p', 'ul', 'div'], recursive=True):
                                if child_el.find_parent(['table', 'figure', 'figcaption', 'nav', 'header', 'footer']): continue
                                text = child_el.get_text(separator=' ', strip=True)
                                if text: collected_texts.append(text)
                                if sum(len(t) for t in collected_texts) > 2500 : break
                            final_abstract_text = " ".join(collected_texts).strip()
                            if len(final_abstract_text) > 70:
                                self._log("HTML Abstract: Extracted from div/section after 'Abstract' heading.", "debug"); return final_abstract_text
                            break 
                        next_content_node = next_content_node.find_next_sibling() if hasattr(next_content_node, 'find_next_sibling') else None
                    # Fallback to general siblings if specific div/section not found
                    collected_general_texts = []
                    current_sib = abstract_heading.find_next_sibling()
                    try:
                        while current_sib:
                            if current_sib.name in ['h1','h2','h3','h4','h5'] or \
                            (hasattr(current_sib, 'attrs') and any(val in current_sib.attrs.get(attr, '').lower() for attr in ['id', 'class'] for val in ['reference', 'citation', 'biblio'])):
                                break 
                            text = current_sib.get_text(separator=' ', strip=True)
                            if text: collected_general_texts.append(text)
                            if sum(len(t) for t in collected_general_texts) > 2500: break
                            current_sib = current_sib.find_next_sibling()
                        final_general_text = " ".join(collected_general_texts).strip()
                        if len(final_general_text) > 70:
                            self._log("HTML Abstract: Extracted from general siblings after 'Abstract' heading.", "debug"); return final_general_text
                    except Exception:
                        pass
            # "Introduction" section search
            if not abstract_text_candidate:
                # (As in original)
                introduction_heading_element = None
                for heading_tag_name in ['h1', 'h2', 'h3']:
                    headings = soup.find_all(heading_tag_name, string=re.compile(r'^\s*Introduction\s*$', re.I))
                    if headings:
                        article_body = soup.find('article', class_='markdown-body')
                        if article_body:
                            intro_in_article = article_body.find(heading_tag_name, string=re.compile(r'^\s*Introduction\s*$', re.I))
                            if intro_in_article: introduction_heading_element = intro_in_article; break
                        if not introduction_heading_element: introduction_heading_element = headings[0]; break
                if introduction_heading_element:
                    # (Logic to collect content after Introduction)
                    start_node = introduction_heading_element.parent if introduction_heading_element.parent.name == 'div' and 'markdown-heading' in introduction_heading_element.parent.get('class', []) else introduction_heading_element
                    collected_texts, char_count, max_chars_for_intro = [], 0, 2500
                    for sibling in start_node.find_next_siblings():
                        if char_count >= max_chars_for_intro: break
                        if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] or (sibling.name == 'div' and "markdown-heading" in sibling.get('class', [])): break
                        current_text = ""
                        if sibling.name == 'p': current_text = sibling.get_text(separator=' ', strip=True)
                        elif sibling.name == 'ul': current_text = "\n".join([f"- {li.get_text(separator=' ', strip=True)}" for li in sibling.find_all('li', recursive=False)])
                        if current_text:
                            if char_count + len(current_text) > max_chars_for_intro: current_text = current_text[:max_chars_for_intro - char_count] + "..."
                            collected_texts.append(current_text); char_count += len(current_text)
                    if collected_texts:
                        introduction_abstract = "\n\n".join(collected_texts).strip()
                        if len(introduction_abstract) > 70:
                            self._log(f"HTML Abstract: Extracted from 'Introduction' section for {effective_url}.", "success"); return introduction_abstract
            
            # General selectors
            if not abstract_text_candidate:
                # (As in original)
                general_selectors = ['div.abstract', 'section.abstract', 'article.abstract', 'div#abstract', 'section#abstract', 'div[class*="abstract-text"]', 'div[itemprop="description"]', 'div[class*="entry-summary"]', 'div.summary', 'article p']
                for selector in general_selectors:
                    elements = soup.select(selector)
                    if elements:
                        candidate_texts = []
                        for i, el in enumerate(elements):
                            text = el.get_text(separator=' ', strip=True)
                            if text: candidate_texts.append(text)
                            if selector == 'article p' and i >= 4: break 
                        full_text = " ".join(candidate_texts).strip()
                        if len(full_text) > 150: self._log(f"HTML Abstract: Found using general selector '{selector}'.", "debug"); return full_text
            
            # Meta tags (last resort)
            if not abstract_text_candidate:
                # (As in original)
                for meta_attr_spec in [{'name': 'description'}, {'property': 'og:description'}, {'name': 'twitter:description'}]:
                    meta_tag = soup.find('meta', attrs=meta_attr_spec)
                    if meta_tag and meta_tag.get('content'):
                        abs_text = meta_tag['content'].strip()
                        if len(abs_text.split(',')) > 5 and len(abs_text) < 300 : self._log(f"HTML Abstract: Meta tag {meta_attr_spec} looks like author list, skipping.", "debug"); continue
                        if len(abs_text) > 70: self._log(f"HTML Abstract: Found in meta tag {meta_attr_spec} (last resort).", "debug"); return abs_text
            self._log(f"HTML Parsing: No abstract found in {effective_url} via any heuristic after all attempts.", "debug")
            return None

        except requests.exceptions.HTTPError as http_err:
            log_url_info = f"original URL: {url}" + (f", effective (error) URL: {effective_url}" if url != effective_url else "")
            self._log(f"HTML Parsing: HTTP error {http_err.response.status_code} for {log_url_info}: {http_err}", "debug")
            # API fallback logic for 403 errors
            if http_err.response.status_code == 403:
                if self.elsevier_api_key and "sciencedirect.com/science/article/pii/" in effective_url:
                    # (Elsevier API call as in original)
                    api_abstract = self._get_elsevier_abstract_from_linking_url(effective_url, self.elsevier_api_key)
                    if api_abstract and not api_abstract.startswith("❌"): return api_abstract
                elif self.springer_api_key and "link.springer.com/" in effective_url:
                    # (Springer API call as in original)
                    api_abstract = self._get_springer_abstract_from_url(effective_url, self.springer_api_key)
                    if api_abstract and not api_abstract.startswith("❌"): return api_abstract
            return None
        except requests.exceptions.RequestException as e:
            log_url_info = f"original URL: {url}" + (f", effective URL during attempt: {effective_url}" if url != effective_url and effective_url != url else "")
            self._log(f"HTML Parsing: Request error for {log_url_info}: {e}", "warn"); return None
        return None # Should be unreachable if all paths return

    def _clean_title_for_search(self, title_str: str, attempt=1) -> str:
        if not title_str: return ""
        cleaned = title_str
        cleaned = re.sub(r'\\(?:emph|textbf|textit|texttt|textsc|mathrm|mathsf|mathcal|mathbf|bm)\s*\{(.*?)\}', r'\1', cleaned)
        cleaned = re.sub(r'(?:emph|textbf|textit|texttt|textsc|mathrm|mathsf|mathcal|mathbf|bm)\s*\{(.*?)\}', r'\1', cleaned)
        cleaned = cleaned.replace(r"\'e", "e").replace(r'\"u', 'ue').replace(r'\`a', 'a')
        cleaned = re.sub(r'\{([A-Za-z\d\-:]+)\}', r'\1', cleaned) # For things like {CNNs}
        cleaned = re.sub(r'\\url\{[^\}]+\}', '', cleaned); cleaned = re.sub(r'\\href\{[^\}]+\}\{[^\}]+\}', '', cleaned)
        cleaned = cleaned.replace(r'\&', '&').replace(r'\%', '%').replace(r'\$', '$').replace(r'\_', '_')
        cleaned = cleaned.replace('\n', ' ').strip()
        cleaned = re.sub(r'\{\\natexlab\{[^}]*\}\}', '', cleaned); cleaned = re.sub(r'\{\\noop\{[^}]*\}\}', '', cleaned)
        cleaned = re.sub(r'^In\s*:\s*(?=[A-Z])', '', cleaned).strip(); cleaned = re.sub(r'^In\s+(?=[A-Z])', '', cleaned).strip()
        cleaned = re.sub(r'[:?!;]+', ' ', cleaned) # Remove most punctuation, keep spaces
        cleaned = re.sub(r',', '', cleaned) # Remove commas specifically (often problematic for search)
        if attempt == 1: # More aggressive cleaning on first attempt
            cleaned = re.sub(r'[\s\(]+\d{4}[a-z]?\)?\s*$', '', cleaned).strip() # Remove trailing year like (2020) or 2020a
            cleaned = re.sub(r'\s+et\s+al\.?\s*$', '', cleaned, flags=re.IGNORECASE).strip() # Remove "et al."
        cleaned = cleaned.rstrip('.') # Remove trailing period
        cleaned = cleaned.lower() # Convert to lowercase
        cleaned = re.sub(r'\s+', ' ', cleaned).strip() # Normalize whitespace
        cleaned = re.sub(r"^[^\w\s]+", "", cleaned); cleaned = re.sub(r"[^\w\s]+$", "", cleaned) # Remove leading/trailing non-alphanumeric (but keep internal)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip() # Normalize whitespace again
        return cleaned.strip()

    def _fetch_abstract_from_openalex(self, title_query: str, authors_str: str = ""):
        if not title_query: return None
        for attempt in range(1, 3): # Try up to 2 cleaning attempts for title
            cleaned_title = self._clean_title_for_search(title_query, attempt=attempt)
            if not cleaned_title: continue
            self._log(f"OpenAlex(A{attempt}): Searching:'{cleaned_title}'", "debug")
            base_url = "https://api.openalex.org/works"; params = {"filter": f"title.search:{re.escape(cleaned_title)}", "mailto": self.openalex_email, "api_key": self.openalex_api_key}
            headers = {"User-Agent": f"LatexToMarkdownConverter/1.3 (mailto:{self.openalex_email})"}
            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=20)
                response.raise_for_status(); data = response.json(); results = data.get("results", [])
                if results:
                    for i, work in enumerate(results): # Check top 2 results
                        if i >= 2: break
                        if work.get("abstract"): self._log("OpenAlex: Direct abstract.", "success"); time.sleep(0.2); return work["abstract"]
                        if work.get("abstract_inverted_index"):
                            deinverted = self._deinvert_abstract_openalex(work["abstract_inverted_index"])
                            if deinverted: self._log("OpenAlex: Deinverted abstract.", "success"); time.sleep(0.2); return deinverted
                        
                        landing_page_url_val = work.get("primary_location", {}).get("landing_page_url")
                        # (Elsevier/Springer API fallbacks via landing page as in original)
                        if self.elsevier_api_key and isinstance(landing_page_url_val, str) and landing_page_url_val and "linkinghub.elsevier.com/retrieve/pii/" in landing_page_url_val:
                            elsevier_abs = self._get_elsevier_abstract_from_linking_url(landing_page_url_val, self.elsevier_api_key)
                            if elsevier_abs and not elsevier_abs.startswith("❌"):
                                self._log("OpenAlex (via Elsevier API): Abstract found.", "success"); time.sleep(0.2); return elsevier_abs
                        elif self.springer_api_key and isinstance(landing_page_url_val, str) and landing_page_url_val and "link.springer.com/" in landing_page_url_val:
                            springer_abs = self._get_springer_abstract_from_url(landing_page_url_val, self.springer_api_key)
                            if springer_abs and not springer_abs.startswith("❌"):
                                self._log("OpenAlex (via Springer API): Abstract found.", "success"); time.sleep(0.2); return springer_abs
                        # HTML parse fallback
                        # if self.BeautifulSoup and isinstance(landing_page_url_val, str) and landing_page_url_val and not landing_page_url_val.lower().endswith(".pdf"):
                        #     html_abs = self._fetch_and_parse_html_for_abstract(landing_page_url_val)
                        #     if html_abs: self._log("OpenAlex: HTML abstract from landing page.", "success"); time.sleep(0.2); return html_abs
                        # PDF parse fallback
                        pdf_url_candidate_from_landing = None
                        if isinstance(landing_page_url_val, str) and landing_page_url_val and landing_page_url_val.lower().endswith(".pdf"):
                            pdf_url_candidate_from_landing = landing_page_url_val
                        pdf_url = work.get("primary_location", {}).get("pdf_url") or pdf_url_candidate_from_landing
                        if not pdf_url: # Check other OA locations for PDF
                            for loc in work.get("locations", []):
                                loc_landing_page_url_val = loc.get("landing_page_url")
                                loc_pdf_url_candidate = None
                                if isinstance(loc_landing_page_url_val, str) and loc_landing_page_url_val and loc_landing_page_url_val.lower().endswith(".pdf"):
                                    loc_pdf_url_candidate = loc_landing_page_url_val
                                if loc.get("is_oa") and (loc.get("pdf_url") or loc_pdf_url_candidate):
                                    pdf_url = loc.get("pdf_url") or loc_pdf_url_candidate; break
                        if pdf_url:
                            try:
                                pdf_resp = requests.get(pdf_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'}, allow_redirects=True)
                                pdf_resp.raise_for_status()
                                if 'application/pdf' in pdf_resp.headers.get('Content-Type','').lower():
                                    pdf_text = self._extract_text_from_pdf_content(pdf_resp.content)
                                    if pdf_text:
                                        pdf_abs_candidate = self._find_abstract_in_pdf_text(pdf_text)
                                        if pdf_abs_candidate: self._log("OpenAlex: PDF abstract from primary/OA location.", "success"); time.sleep(0.2); return pdf_abs_candidate
                            except Exception as e_pdf: self._log(f"OpenAlex: PDF error for {pdf_url}: {e_pdf}", "warn")
                    self._log(f"OpenAlex(A{attempt}): Found papers for '{cleaned_title[:60]}' but no abstract.", "debug"); return None # No abstract found in top results for this title variant
            except Exception as e: self._log(f"OpenAlex(A{attempt}): API/Req error for '{cleaned_title[:60]}': {e}", "warn")
            time.sleep(1) # Polite delay
        return None # All title cleaning attempts failed

    def _fetch_abstract_from_semantic_scholar(self, title: str, authors_str: str = ""):
        if not title: return None
        for attempt in range(1, 6): # Try up to 5 cleaning attempts for title
            cleaned_title = self._clean_title_for_search(title, attempt=attempt)
            if not cleaned_title: continue
            self._log(f"S2(A{attempt}): Searching:'{cleaned_title}'", "debug")
            try:
                search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote_plus(cleaned_title)}&fields=title,abstract,url&limit=2"
                headers = {'User-Agent': 'LatexToMarkdownConverter/1.3'}
                if self.semantic_scholar_api_key:
                    headers['x-api-key'] = self.semantic_scholar_api_key
                    self._log(f"S2(A{attempt}): Using Semantic Scholar API key.", "debug")
                else: self._log(f"S2(A{attempt}): No Semantic Scholar API key provided. Making unauthenticated request.", "debug")
                
                response = requests.get(search_url, headers=headers, timeout=15)
                response.raise_for_status(); data = response.json()
                if data.get("data"):
                    for paper_data in data["data"]: # Check top results
                        if paper_data.get("abstract"):
                            self._log(f"S2(A{attempt}): Found abstract for '{cleaned_title}'.", "success"); time.sleep(0.3); return paper_data["abstract"]
                        s2_url = paper_data.get("url")
                        # (Elsevier/Springer/HTML fallbacks via S2 URL as in original)
                        if self.elsevier_api_key and s2_url and "linkinghub.elsevier.com/retrieve/pii/" in s2_url:
                            elsevier_abs = self._get_elsevier_abstract_from_linking_url(s2_url, self.elsevier_api_key)
                            if elsevier_abs and not elsevier_abs.startswith("❌"):
                                self._log("S2 (via Elsevier API): Abstract found.", "success"); time.sleep(0.3); return elsevier_abs
                        elif self.springer_api_key and s2_url and "link.springer.com/" in s2_url:
                            springer_abs = self._get_springer_abstract_from_url(s2_url, self.springer_api_key)
                            if springer_abs and not springer_abs.startswith("❌"):
                                self._log("S2 (via Springer API): Abstract found.", "success"); time.sleep(0.3); return springer_abs
                        elif self.BeautifulSoup and s2_url and not s2_url.lower().endswith(".pdf"):
                                html_abs = self._fetch_and_parse_html_for_abstract(s2_url)
                                if html_abs: self._log("S2 (via HTML parse): Abstract found.", "success"); time.sleep(0.3); return html_abs
                    self._log(f"S2(A{attempt}): Found papers for '{cleaned_title}' but no abstract (or via fallback URLs).", "debug"); return None
            except Exception as e: self._log(f"S2(A{attempt}): API error for '{cleaned_title}': {e}", "warn")
            # Time to sleep
            time.sleep(0.5*self.max_workers)
        return None

    def _extract_bibitem_components(self, bibitem_text_chunk: str) -> dict:
        """
        Extracts key, authors, title, and other details from a bibitem entry.
        This version is enhanced to correctly parse \href{url}{text} formats,
        extracting "text" as the title for API searches.
        """
        key_match = re.match(r"\\bibitem(?:\[[^\]]*\])?\{([^\}]+)\}", bibitem_text_chunk)
        key = key_match.group(1) if key_match else ""
        content_after_key = bibitem_text_chunk[key_match.end():].strip() if key_match else bibitem_text_chunk

        # Split out authors from the rest of the content
        author_parts = content_after_key.split(r'\newblock', 1)
        authors_str = author_parts[0].strip().rstrip('.,;')
        text_after_authors = author_parts[1].strip() if len(author_parts) > 1 else ""

        # Initialize components to be extracted
        title_raw = ""
        title_cleaned_for_api = ""
        details_after_title_str = ""
        is_url_title_flag = False  # Flag to indicate if a primary URL is present
        url_if_title_val = None

        # Isolate the main title block from subsequent details
        title_block_parts = text_after_authors.split(r'\newblock', 1)
        current_title_candidate_block = title_block_parts[0].strip()
        further_details_block = ("\\newblock " + title_block_parts[1].strip()) if len(title_block_parts) > 1 else ""

        # STRATEGY 1: Prioritize matching the \href{url}{text} pattern.
        # This regex handles potentially nested braces in the text part.
        href_match = re.match(r"^\\href\s*\{([^}]+)\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}", current_title_candidate_block)
        if href_match:
            url_if_title_val = href_match.group(1).strip()
            title_raw = href_match.group(2).strip()  # This is the descriptive text, which we use as the title
            title_cleaned_for_api = self._clean_title_for_search(title_raw)
            is_url_title_flag = True
            # Any content after the \href command in this block is part of the details
            details_after_title_str = (current_title_candidate_block[href_match.end():].strip() + " " + further_details_block).strip()

        else:
            # STRATEGY 2: If no \href, check for \url{...} or a raw http:// URL as the title.
            url_patterns = [r"^(?P<url>\\url\{([^}]+)\})", r"^(?P<url>https?://[^\s]+)"]
            url_is_title = False
            for pat_str in url_patterns:
                url_match = re.match(pat_str, current_title_candidate_block)
                if url_match:
                    # Check if there is other significant text in the block. If not, the URL is the title.
                    rest_of_block = current_title_candidate_block[url_match.end():].strip()
                    if not rest_of_block or re.match(r"^[,\s]*\d{4}\.?$", rest_of_block) or len(rest_of_block) < 15:
                        title_raw = url_match.group("url").strip()
                        url_content = url_match.group(2) if pat_str.startswith(r"^(?P<url>\\url") else title_raw
                        
                        # In this case, the URL itself is the item to be searched
                        title_cleaned_for_api = url_content
                        url_if_title_val = url_content
                        is_url_title_flag = True
                        url_is_title = True
                        details_after_title_str = (rest_of_block + " " + further_details_block).strip()
                        break  # Title found, exit loop

            if not url_is_title:
                # STRATEGY 3: Default to parsing a standard text title.
                title_end_delimiters = [
                    r"In\s+(?:Proc\.?(?:eedings)?|Workshop|Conference|Journal|Symposium)\b", r"\b(?:[A-Z][a-z]+)\s+(?:Press|Publishers|Verlag)\b",
                    r"Ph\.?D\.?\s+thesis\b", r"Master(?:'s)?\s+thesis\b", r"arXiv preprint arXiv:",
                    r"[,;\s]\(?(?P<year>\d{4})\)?(?=\W|$|\s*\\newblock|\s*notes\b)", r"\.\s+\d{4}\.", r"Article\s+No\.",
                    r"vol\.\s*\d+", r"pp\.\s*\d+", r"no\.\s*\d+", r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
                ]
                min_delimiter_idx = len(current_title_candidate_block)
                for pat in title_end_delimiters:
                    match = re.search(pat, current_title_candidate_block, re.IGNORECASE)
                    if match and match.start() < min_delimiter_idx:
                        # Avoid matching year if it's part of a short, non-terminated title
                        if 'year' in match.groupdict() and len(current_title_candidate_block[:match.start()].strip()) < 15 and not re.search(r'[,.;:]$', current_title_candidate_block[:match.start()].strip()):
                            pass
                        else:
                            min_delimiter_idx = match.start()

                title_raw = current_title_candidate_block[:min_delimiter_idx].strip().rstrip('.,;/')
                title_cleaned_for_api = self._clean_title_for_search(title_raw)
                details_after_title_str = (current_title_candidate_block[min_delimiter_idx:] + " " + further_details_block).strip()

        return {
            "key": key,
            "authors": authors_str,
            "title_raw": title_raw,
            "title_cleaned": title_cleaned_for_api,
            "is_url_title": is_url_title_flag,
            "url_if_title": url_if_title_val,
            "details_after_title": details_after_title_str.strip()
        }
        
    def _fully_process_bbl(self, raw_bbl_content: str) -> str:
        # (As in original, with API calls for abstracts)
        self._log("Starting full BBL processing (abstracts, keys)...", "info")
        if not raw_bbl_content: return ""
        bbl_preamble = ""; begin_thebibliography_match = re.search(r"\\begin{thebibliography}\{[^}]*\}", raw_bbl_content)
        if begin_thebibliography_match:
            bbl_preamble = raw_bbl_content[:begin_thebibliography_match.start()]
            external_cleanup_patterns_str = [r"\\providecommand\\{\\natexlab\\}\\[1\\]\\{#1\\}\\s*", r"\\providecommand\\{\\url\\}\\[1\\]\\{\\texttt\\{#1\\}\\}\\s*", r"\\providecommand\\{\\doi\\}\\[1\\]\\{doi:\s*#1\\}\\s*", r"\\expandafter\\ifx\\csname\s*urlstyle\\endcsname\\relax[\s\S]*?\\fi\s*", r"\\expandafter\\ifx\\csname\s*doi\\endcsname\\relax[\s\S]*?\\fi\s*"]
            for p_str in external_cleanup_patterns_str:
                try: bbl_preamble = re.sub(p_str, "", bbl_preamble, flags=re.DOTALL)
                except re.error as e_re: self._log(f"Regex error cleaning external BBL preamble with pattern '{p_str}': {e_re}", "warn")
            bbl_preamble = bbl_preamble.strip()
        bibliography_env_start_for_pandoc = r"\begin{thebibliography}{}"
        items_text_start_offset = begin_thebibliography_match.end() if begin_thebibliography_match else 0
        end_thebibliography_match = re.search(r"\\end{thebibliography}", raw_bbl_content)
        items_text_end_offset = end_thebibliography_match.start() if end_thebibliography_match else len(raw_bbl_content)
        bbl_items_text = raw_bbl_content[items_text_start_offset:items_text_end_offset].strip()
        if bbl_items_text: # Clean internal preamble commands if any
            internal_cleanup_patterns = [re.compile(r"^\s*\\providecommand\\{\\natexlab\\}\\[1\\]\\{#1\\}\\s*", flags=re.MULTILINE | re.DOTALL), re.compile(r"^\s*\\providecommand\\{\\url\\}\\[1\\]\\{\\texttt\\{#1\\}\\}\\s*", flags=re.MULTILINE | re.DOTALL), re.compile(r"^\s*\\providecommand\\{\\doi\\}\\[1\\]\\{doi:\s*#1\\}\\s*", flags=re.MULTILINE | re.DOTALL), re.compile(r"^\s*\\expandafter\\ifx\\csname\s*urlstyle\\endcsname\\relax[\s\S]*?\\fi\s*", flags=re.MULTILINE | re.DOTALL), re.compile(r"^\s*\\expandafter\\ifx\\csname\s*doi\\endcsname\\relax[\s\S]*?\\fi\s*", flags=re.MULTILINE | re.DOTALL)]
            cleaned_any_internal_preamble = False; original_bbl_items_text_len_for_log = len(bbl_items_text)
            while True:
                text_before_cleaning_pass = bbl_items_text
                for pattern_obj in internal_cleanup_patterns:
                    try: bbl_items_text = pattern_obj.sub("", bbl_items_text).strip()
                    except re.error as e_re_int: self._log(f"Regex error cleaning internal BBL preamble with pattern '{pattern_obj.pattern}': {e_re_int}", "warn")
                if bbl_items_text == text_before_cleaning_pass: break
                cleaned_any_internal_preamble = True
            if cleaned_any_internal_preamble: self._log(f"Cleaned internal BBL preamble. Remaining item text starts with: '{bbl_items_text[:100]}...'", "debug")
            elif original_bbl_items_text_len_for_log > 0 and not bbl_items_text.startswith(r"\bibitem") and bbl_items_text: self._log(f"BBL item text starts with non-bibitem content after internal preamble cleaning attempt: '{bbl_items_text[:100]}...'", "debug")
        
        processed_bibitems_parts = []
        if bbl_items_text:
            split_bibitems = re.split(r'(\\bibitem)', bbl_items_text)
            if split_bibitems and split_bibitems[0].strip() and not split_bibitems[0].startswith("\\bibitem"): split_bibitems.pop(0)
            elif split_bibitems and not split_bibitems[0].strip() and len(split_bibitems) > 1 : split_bibitems.pop(0)
            k = 0
            while k < len(split_bibitems):
                if split_bibitems[k] == r'\bibitem':
                    if k + 1 < len(split_bibitems): item_chunk = r'\bibitem' + split_bibitems[k+1].strip(); k += 2
                    else: self._log("Warning: Dangling \\bibitem in BBL processing.", "warn"); processed_bibitems_parts.append(r'\bibitem'); k += 1; continue
                else: self._log(f"Warning: Unexpected chunk in BBL processing, expected \\bibitem: '{split_bibitems[k][:50]}...'", "warn"); k += 1; continue
                
                components = self._extract_bibitem_components(item_chunk)
                reconstructed_item_parts = [f"\\bibitem{{{components['key']}}}", components['authors']]
                if components['authors'] and components['title_raw']: reconstructed_item_parts.append(r"\newblock")
                if components['title_raw']: reconstructed_item_parts.append(components['title_raw'])
                if components['details_after_title'].strip(): reconstructed_item_parts.append(components['details_after_title'])
                current_item_reconstructed = " ".join(p.strip() for p in reconstructed_item_parts if p.strip())
                current_item_reconstructed = re.sub(r'\s*\\newblock\s*', r' \\newblock ', current_item_reconstructed).strip()
                current_item_reconstructed = re.sub(r'\s+', ' ', current_item_reconstructed)
                
                abstract_text = None
                # (ArXiv abstract logic as in original)
                if 'arxiv' in item_chunk.lower():
                    arxiv_id_local = None
                    specific_match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)', item_chunk, re.IGNORECASE)
                    if specific_match: arxiv_id_local = specific_match.group(1)
                    else:
                        general_match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', item_chunk)
                        if general_match: arxiv_id_local = general_match.group(1)
                    if arxiv_id_local and re.fullmatch(r'\d{4}\.\d{4,5}(v\d+)?', arxiv_id_local):
                        arxiv_abs_url = f"https://arxiv.org/abs/{arxiv_id_local}"
                        abstract_text = self._get_arxiv_abstract_from_page(arxiv_abs_url)
                        if abstract_text: self._log(f"Bibitem {components['key']}: Abstract successfully extracted from arXiv {arxiv_abs_url}.", "success")

                if not abstract_text: # Fallback to other sources
                    title_for_api = components["title_cleaned"]
                    if components["is_url_title"] and isinstance(components["url_if_title"], str) and components["url_if_title"]:
                        # (API calls for Elsevier/Springer from URL title as in original)
                        current_url_to_check = components["url_if_title"]
                        if "linkinghub.elsevier.com/retrieve/pii/" in current_url_to_check and self.elsevier_api_key:
                            api_abs = self._get_elsevier_abstract_from_linking_url(current_url_to_check, self.elsevier_api_key)
                            if api_abs and not api_abs.startswith("❌"): abstract_text = api_abs
                        elif "link.springer.com/" in current_url_to_check and self.springer_api_key:
                            api_abs = self._get_springer_abstract_from_url(current_url_to_check, self.springer_api_key)
                            if api_abs and not api_abs.startswith("❌"): abstract_text = api_abs
                        if not abstract_text and self.BeautifulSoup and not current_url_to_check.lower().endswith(".pdf"):
                             if not (("arxiv.org/abs/" in current_url_to_check or "arxiv.org/html/" in current_url_to_check) and 'arxiv' in item_chunk.lower()):
                                abstract_text = self._fetch_and_parse_html_for_abstract(current_url_to_check)
                    if not abstract_text and title_for_api:
                        # abstract_text = self._fetch_abstract_from_semantic_scholar(title_for_api, components["authors"])
                        # # if not abstract_text:
                        # #     abstract_text = self._fetch_abstract_from_openalex(title_for_api, components["authors"])
                        
                        abstract_text = self._fetch_abstract_from_openalex(title_for_api, components["authors"])
                        if not abstract_text:
                            abstract_text = self._fetch_abstract_from_semantic_scholar(title_for_api, components["authors"])
                    # (Fallback to general URL parsing from bibitem as in original)
                    if not abstract_text:
                        latex_url_match = re.search(r"\\url\{([^}]+)\}", item_chunk)
                        http_url_match = re.search(r"\b(https?://[^\s\"'<>()\[\]{},;]+[^\s\"'<>()\[\]{},;\.])", item_chunk)
                        potential_urls = []
                        if latex_url_match and isinstance(latex_url_match.group(1), str): potential_urls.append(latex_url_match.group(1).strip())
                        if http_url_match and isinstance(http_url_match.group(1), str):
                            url_candidate = http_url_match.group(1).strip()
                            if not (latex_url_match and isinstance(latex_url_match.group(1), str) and latex_url_match.group(1).strip() == url_candidate):
                                preceding_char_idx = http_url_match.start(1) - 1
                                if preceding_char_idx < 0 or item_chunk[preceding_char_idx] not in ['{']: potential_urls.append(url_candidate)
                        for p_url in potential_urls:
                            if not isinstance(p_url, str): continue
                            if components["is_url_title"] and components["url_if_title"] == p_url: continue
                            if ("arxiv.org/pdf/" in p_url or "arxiv.org/abs/" in p_url) and 'arxiv' in item_chunk.lower(): continue
                            # (Elsevier/Springer/HTML/PDF fallbacks for p_url as in original)
                            if "linkinghub.elsevier.com/retrieve/pii/" in p_url and self.elsevier_api_key:
                                api_abs = self._get_elsevier_abstract_from_linking_url(p_url, self.elsevier_api_key)
                                if api_abs and not api_abs.startswith("❌"): abstract_text = api_abs
                            elif "link.springer.com/" in p_url and self.springer_api_key:
                                api_abs = self._get_springer_abstract_from_url(p_url, self.springer_api_key)
                                if api_abs and not api_abs.startswith("❌"): abstract_text = api_abs
                            if not abstract_text and self.BeautifulSoup and not p_url.lower().endswith(".pdf"):
                                if not (("arxiv.org/abs/" in p_url or "arxiv.org/html/" in p_url) and 'arxiv' in item_chunk.lower()):
                                    html_abstract = self._fetch_and_parse_html_for_abstract(p_url)
                                    if html_abstract: abstract_text = html_abstract; self._log(f"Fallback: Abstract from HTML URL '{p_url}'.", "success")
                            if not abstract_text and self.PdfReader and (p_url.lower().endswith(".pdf") or "arxiv.org/pdf/" in p_url):
                                try:
                                    pdf_resp = requests.get(p_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/pdf, */*'}, allow_redirects=True)
                                    pdf_resp.raise_for_status()
                                    if 'application/pdf' in pdf_resp.headers.get('Content-Type','').lower():
                                        pdf_txt = self._extract_text_from_pdf_content(pdf_resp.content)
                                        if pdf_txt:
                                            pdf_abs_candidate = self._find_abstract_in_pdf_text(pdf_txt)
                                            if pdf_abs_candidate: abstract_text = pdf_abs_candidate; self._log(f"Fallback: Abstract from PDF URL '{p_url}'.", "success")
                                except Exception as fb_pdf_err: self._log(f"Fallback: PDF error for {p_url}: {fb_pdf_err}", "warn")
                            if abstract_text: break
                
                if abstract_text:
                    escaped_abstract = self._latex_escape_abstract(abstract_text)
                    current_item_reconstructed += f" \\newblock \\textbf{{Abstract:}} {escaped_abstract}"
                escaped_key_for_display = components['key'].replace('_', r'\_')
                current_item_reconstructed += f" \\newblock (@{escaped_key_for_display})"
                processed_bibitems_parts.append(current_item_reconstructed)
        
        final_bbl_items_joined = "\n\n".join(p for p in processed_bibitems_parts if p)
        final_bbl_string = ""
        if bbl_preamble: final_bbl_string += bbl_preamble + "\n"
        if final_bbl_items_joined or begin_thebibliography_match:
            final_bbl_string += bibliography_env_start_for_pandoc + "\n"
            if final_bbl_items_joined: final_bbl_string += final_bbl_items_joined + "\n"
            final_bbl_string += r"\end{thebibliography}"
        elif not begin_thebibliography_match and final_bbl_items_joined :
            self._log("Warning: Bibitems found but no \\begin{thebibliography} was detected. Outputting items directly.", "warn")
            final_bbl_string += final_bbl_items_joined
        return final_bbl_string.strip()

    def _simple_inline_processed_bbl(self, tex_content: str, processed_bbl_string: str) -> str:
        # (As in original)
        self._log("Inlining fully processed BBL content into TeX...", "debug")
        references_heading_tex = "\n\n\\section*{References}\n\n"
        content_to_inline = (references_heading_tex + processed_bbl_string) if processed_bbl_string.strip() else ""
        modified_tex_content = re.sub(r"^\s*\\bibliographystyle\{[^{}]*\}\s*$", "", tex_content, flags=re.MULTILINE)
        bibliography_command_pattern = r"^\s*\\bibliography\{[^{}]*(?:,[^{}]*)*\}\s*$"
        input_references_bbl_command_pattern = r"\\input{[^\}]*.bbl}"
        has_bibliography_cmd = re.search(bibliography_command_pattern, modified_tex_content, flags=re.MULTILINE)
        has_input_bbl_cmd = re.search(input_references_bbl_command_pattern, modified_tex_content, flags=re.MULTILINE)
        # self._log(modified_tex_content, "debug")
        if has_bibliography_cmd:
            modified_tex_content = re.sub(input_references_bbl_command_pattern, "", modified_tex_content, flags=re.MULTILINE) # Remove any \input{*.bbl}
        elif has_input_bbl_cmd: # If only \input{*.bbl} exists, replace it with a placeholder \bibliography
            self._log(f"FOUND ONLY only \\input{{*.bbl}} but not \\bibliography{{...}}. Replacing \\input{{*.bbl}} with a placeholder \\bibliography{{placeholder}} before replacement.", "debug")
            modified_tex_content = re.sub(input_references_bbl_command_pattern, r"\\bibliography{references_placeholder_for_l2m}", modified_tex_content, count=1, flags=re.MULTILINE)
        else: # No bibliography-related command found, add one
            appendix_match = re.search(r"^(.*?)(\\appendix)", modified_tex_content, flags=re.MULTILINE | re.DOTALL)
            end_doc_match = re.search(r"^(.*?)(\\end\{document\})", modified_tex_content, flags=re.MULTILINE | re.DOTALL)
            if appendix_match:
                insertion_point_text = appendix_match.group(1)
                modified_tex_content = f"{insertion_point_text}\n\\bibliography{{references_placeholder_for_l2m}}\n\n{appendix_match.group(2)}{modified_tex_content[appendix_match.end():]}"
                self._log("Added \\bibliography before \\appendix.", "debug")
            elif end_doc_match:
                insertion_point_text = end_doc_match.group(1)
                modified_tex_content = f"{insertion_point_text}\n\\bibliography{{references_placeholder_for_l2m}}\n\n{end_doc_match.group(2)}"
                self._log("Added \\bibliography before \\end{document}.", "debug")
            else:
                modified_tex_content += "\n\\bibliography{references_placeholder_for_l2m}\n"
                self._log("Appended \\bibliography as no standard insertion point found.", "warn")
        
        final_bibliography_pattern_for_replacement = r"^\s*\\bibliography\{[^{}]*(?:,[^{}]*)*\}\s*$"
        if content_to_inline:
            if re.search(final_bibliography_pattern_for_replacement, modified_tex_content, flags=re.MULTILINE):
                modified_tex_content = re.sub(final_bibliography_pattern_for_replacement, lambda m: content_to_inline + "\n", modified_tex_content, count=1, flags=re.MULTILINE)
            else: # Should not happen if we added a placeholder
                self._log("Could not find a \\bibliography command to replace for inlining, even after attempting to add one. Appending BBL content.", "warn")
                modified_tex_content += "\n" + content_to_inline
        else: # No BBL content, remove \bibliography command
            modified_tex_content = re.sub(final_bibliography_pattern_for_replacement, "", modified_tex_content, flags=re.MULTILINE)
        
        modified_tex_content = re.sub(r"^\s*\\nobibliography\{[^{}]*(?:,[^{}]*)*\}\s*$", "", modified_tex_content, flags=re.MULTILINE)
        return modified_tex_content

    def _find_and_copy_figures_from_markdown(self, markdown_content, output_figure_folder_path):
        # (As in original)
        copied_figure_details = []
        if not markdown_content: self._log("No markdown content to find figures from.", "debug"); return copied_figure_details
        patterns = [r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+[\"'][^\"']*[\"'])?\)", r"<img\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>", r"<embed\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>"]
        img_paths = []
        for p in patterns:
            for match in re.finditer(p, markdown_content, flags=re.IGNORECASE | re.DOTALL): img_paths.append(match.group(1))
        if not img_paths: self._log("No image paths found in markdown content.", "debug"); return copied_figure_details
        self._log(f"Found {len(img_paths)} potential image paths in markdown. Target figure folder: {output_figure_folder_path}", "debug")
        output_figure_folder_path.mkdir(parents=True, exist_ok=True)
        common_ext = ['.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg', '.gif', '.bmp', '.tiff']
        copied_count = 0; copied_source_abs_paths_set = set()
        for raw_path in img_paths:
            try: decoded_path_str = urllib.parse.unquote(raw_path)
            except Exception as e_dec: self._log(f"Could not decode path '{raw_path}': {e_dec}. Skipping.", "warn"); continue
            if urllib.parse.urlparse(decoded_path_str).scheme in ['http', 'https']: self._log(f"Skipping web URL: {decoded_path_str}", "debug"); continue
            src_abs_candidate = (self.folder_path / decoded_path_str).resolve(); found_abs_path = None
            if src_abs_candidate.is_file(): found_abs_path = src_abs_candidate
            elif not src_abs_candidate.suffix:
                for ext_ in common_ext:
                    path_with_ext = (self.folder_path / (decoded_path_str + ext_)).resolve()
                    if path_with_ext.is_file(): found_abs_path = path_with_ext; self._log(f"Found '{decoded_path_str}' as '{path_with_ext.name}' by adding extension.", "debug"); break
            if not found_abs_path and self.main_tex_path: # Try relative to main .tex file
                src_abs_candidate_rel_main = (self.main_tex_path.parent / decoded_path_str).resolve()
                if src_abs_candidate_rel_main.is_file(): found_abs_path = src_abs_candidate_rel_main
                elif not src_abs_candidate_rel_main.suffix:
                    for ext_ in common_ext:
                        path_with_ext = (self.main_tex_path.parent / (decoded_path_str + ext_)).resolve()
                        if path_with_ext.is_file(): found_abs_path = path_with_ext; self._log(f"Found '{decoded_path_str}' as '{path_with_ext.name}' (relative to main .tex) by adding extension.", "debug"); break
            if found_abs_path:
                dest_filename = found_abs_path.name; dest_abs_path = (output_figure_folder_path / dest_filename).resolve()
                if found_abs_path in copied_source_abs_paths_set and dest_abs_path.exists():
                    self._log(f"Figure source '{found_abs_path.name}' (from MD path '{raw_path}') already copied to '{dest_abs_path}'. Adding reference.", "debug")
                else:
                    try:
                        shutil.copy2(str(found_abs_path), str(dest_abs_path))
                        copied_source_abs_paths_set.add(found_abs_path); copied_count +=1
                        self._log(f"Copied '{found_abs_path.name}' (from MD path '{raw_path}') to '{dest_abs_path}'.", "debug")
                    except Exception as e_copy: self._log(f"Figure copy error for '{found_abs_path}' to '{dest_abs_path}': {e_copy}", "warn"); continue
                details = {"raw_markdown_path": raw_path, "original_filename_with_ext": found_abs_path.name, "source_abs_path": str(found_abs_path), "copied_dest_abs_path": str(dest_abs_path), "original_ext": found_abs_path.suffix.lower()}
                copied_figure_details.append(details)
            else: self._log(f"Could not find figure source for markdown path: '{raw_path}' (decoded: '{decoded_path_str}')", "warn")
        if copied_count > 0: self._log(f"Copied {copied_count} unique figure files to '{output_figure_folder_path}'.", "success")
        elif img_paths: self._log("Found image paths in markdown, but no new files were copied (e.g., all URLs, files not found, or already copied).", "info")
        return copied_figure_details

    def _convert_and_update_figure_paths_in_markdown(self, markdown_content, figure_details_list, output_base_path):
        # (As in original, with pdf2image and Pillow fallbacks)
        updated_markdown_content = markdown_content
        if figure_details_list:
            NO_CONVERSION_EXTENSIONS = ['.jpg', '.jpeg', '.png']
            for fig_info in figure_details_list:
                original_md_path_in_doc = fig_info["raw_markdown_path"]
                copied_file_abs_path = Path(fig_info["copied_dest_abs_path"])
                original_ext = fig_info["original_ext"]
                original_filename_stem = copied_file_abs_path.stem
                target_filename_in_figures_subdir = copied_file_abs_path.name
                conversion_done = False
                if self.convert_from_path and original_ext == '.pdf':
                    target_png_filename = f"{original_filename_stem}.png" # Ensure it's just filename.png
                    target_png_abs_path_in_figures_subdir = copied_file_abs_path.with_name(target_png_filename)
                    if target_png_abs_path_in_figures_subdir.exists() and copied_file_abs_path != target_png_abs_path_in_figures_subdir:
                        self._log(f"Converted PNG '{target_png_abs_path_in_figures_subdir.name}' (from PDF) already exists. Using it.", "debug"); conversion_done = True
                    else:
                        try:
                            self._log(f"Attempting PDF to PNG conversion for '{copied_file_abs_path.name}' using pdf2image...", "debug")
                            images = self.convert_from_path(copied_file_abs_path, poppler_path=self.poppler_path, first_page=1, last_page=1, fmt='png')
                            if images:
                                images[0].save(target_png_abs_path_in_figures_subdir, "PNG")
                                self._log(f"Successfully converted PDF '{copied_file_abs_path.name}' to '{target_png_abs_path_in_figures_subdir.name}' using pdf2image.", "success"); conversion_done = True
                            else: self._log(f"pdf2image returned no images for '{copied_file_abs_path.name}'.", "warn")
                        except Exception as e_pdf2img_generic: self._log(f"Error during pdf2image conversion for '{copied_file_abs_path.name}': {e_pdf2img_generic}", "warn")
                    if conversion_done:
                        target_filename_in_figures_subdir = target_png_filename
                        if copied_file_abs_path.exists() and copied_file_abs_path.suffix == '.pdf' and copied_file_abs_path != target_png_abs_path_in_figures_subdir:
                            try: copied_file_abs_path.unlink(); self._log(f"Deleted original PDF '{copied_file_abs_path.name}' from figures subdir after pdf2image conversion.", "debug")
                            except Exception as e_del: self._log(f"Failed to delete original PDF '{copied_file_abs_path.name}' from figures subdir: {e_del}", "warn")
                
                if not conversion_done and original_ext not in NO_CONVERSION_EXTENSIONS:
                    target_png_filename = f"{original_filename_stem}.png"
                    target_png_abs_path_in_figures_subdir = copied_file_abs_path.with_name(target_png_filename)
                    pillow_conversion_attempted = False
                    if target_png_abs_path_in_figures_subdir.exists() and copied_file_abs_path != target_png_abs_path_in_figures_subdir:
                        self._log(f"Converted PNG '{target_png_abs_path_in_figures_subdir.name}' (Pillow fallback) already exists. Using it.", "debug"); conversion_done = True
                    elif self.PILImage:
                        pillow_conversion_attempted = True
                        self._log(f"Attempting conversion to PNG for '{copied_file_abs_path.name}' (in figures subdir) using Pillow...", "debug")
                        try:
                            if original_ext in ['.pdf', '.eps'] and not shutil.which("gs"): self._log(f"Ghostscript (gs) command not found. Pillow may fail to convert '{copied_file_abs_path.name}'. Install Ghostscript for PDF/EPS conversion.", "warn")
                            if original_ext == '.svg': self._log(f"Pillow does not directly support SVG conversion. Skipping Pillow conversion for '{copied_file_abs_path.name}'.", "warn")
                            else:
                                img = self.PILImage.open(copied_file_abs_path)
                                if img.mode == 'P' and 'transparency' in img.info: img = img.convert("RGBA")
                                elif img.mode not in ['RGB', 'RGBA', 'L', 'LA']: img = img.convert("RGBA")
                                img.save(target_png_abs_path_in_figures_subdir, "PNG")
                                self._log(f"Successfully converted '{copied_file_abs_path.name}' to '{target_png_abs_path_in_figures_subdir.name}' (in figures subdir) using Pillow.", "debug"); conversion_done = True
                        except FileNotFoundError: self._log(f"Pillow conversion failed for '{copied_file_abs_path.name}': A dependent component (like Ghostscript for PDF/EPS) might be missing or not in PATH.", "warn")
                        except Exception as e_conv: self._log(f"Error converting '{copied_file_abs_path.name}' to PNG using Pillow: {e_conv}", "warn")
                    elif not pillow_conversion_attempted: self._log("Pillow (PIL) library not found. Cannot convert images using Pillow.", "warn")
                    if conversion_done:
                        target_filename_in_figures_subdir = target_png_filename
                        if copied_file_abs_path.exists() and copied_file_abs_path != target_png_abs_path_in_figures_subdir:
                            try: copied_file_abs_path.unlink(); self._log(f"Deleted original file '{copied_file_abs_path.name}' from figures subdir after Pillow conversion.", "debug")
                            except Exception as e_del: self._log(f"Failed to delete original file '{copied_file_abs_path.name}' from figures subdir: {e_del}", "warn")
                    elif pillow_conversion_attempted: self._log(f"Pillow conversion skipped or failed for '{copied_file_abs_path.name}'. Markdown will link to original in figures subdir.", "debug")

                # 3. Update Markdown paths to point to the (potentially converted) file in the figures subdirectory
                new_md_path_for_doc = f"./figures/{urllib.parse.quote(target_filename_in_figures_subdir)}" # URL encode the filename
                
                escaped_original_md_path = re.escape(original_md_path_in_doc) # original_md_path_in_doc is already URL-decoded from _find_and_copy...
                                                                                # but might contain characters special to regex.

                # Update Markdown image links: ![alt](original_path) -> ![alt](./figures/new_name)
                md_link_pattern = rf"(!\[(?:[^\]]*)\]\()({escaped_original_md_path})(\))"
                updated_markdown_content = re.sub(md_link_pattern, rf"\1{new_md_path_for_doc}\3", updated_markdown_content)
                
                # Update HTML img src: <img src="original_path"> -> <img src="./figures/new_name">
                # Ensure to match both single and double quotes for src attribute
                img_src_pattern = rf'(<img\s+[^>]*?src\s*=\s*)(["\'])({escaped_original_md_path})(\2[^>]*?>)'
                updated_markdown_content = re.sub(img_src_pattern, rf"\1\2{new_md_path_for_doc}\2\4", updated_markdown_content)
                
                # Update HTML embed src (before it's converted to img): <embed src="original_path"> -> <embed src="./figures/new_name">
                embed_src_pattern = rf'(<embed\s+[^>]*?src\s*=\s*)(["\'])({escaped_original_md_path})(\2[^>]*?>)'
                updated_markdown_content = re.sub(embed_src_pattern, rf"<embed src=\2{new_md_path_for_doc}\2\4", updated_markdown_content)
                updated_markdown_content = re.sub(r"<embed src=" , r"<img src=", updated_markdown_content)

                if original_md_path_in_doc != new_md_path_for_doc and original_md_path_in_doc in markdown_content: # A basic check if replacement likely happened
                    self._log(f"Updated Markdown path for '{original_md_path_in_doc}' to '{new_md_path_for_doc}'.", "debug")
                elif original_md_path_in_doc == new_md_path_for_doc and original_md_path_in_doc in updated_markdown_content: # No effective path change but check content
                     pass # Path was already correct, no log needed unless it's verbose
                # Consider logging if a path was expected to be updated but wasn't found in the content.
        
        # Convert any remaining <embed> tags within <figure> to <img> tags
        # This regex tries to be careful about matching attributes and ensuring it's within a figure.
        final_markdown_content = re.sub(
            r"(<figure\b[^>]*>[\s\S]*?)<embed(\s+[^>]*?src\s*=\s*[\"'][^\"']+[\"'](?:[^>]*?)?)>([\s\S]*?</figure>)",
            r"\1<img\2/>\3",  # Ensure self-closing img tag for broader compatibility
            updated_markdown_content,
            flags=re.DOTALL | re.IGNORECASE
        )
        if final_markdown_content != updated_markdown_content: 
            self._log("Converted <embed> tags (within <figure>) to <img> tags.", "debug")
        
        return final_markdown_content

    def _resolve_include_path(self, filename: str, current_dir: Path) -> Path | None:
        # (As in original)
        filename = filename.strip()
        potential_filenames = [filename]
        if filename.endswith(".tex"): potential_filenames.append(filename[:-4])
        else: potential_filenames.append(filename + ".tex")
        search_dirs = [current_dir, self.folder_path] # Search in current file's dir, then project root
        for fname_variant in potential_filenames:
            for search_dir in search_dirs:
                try_path = (search_dir / fname_variant).resolve()
                if try_path.is_file(): return try_path
        return None

    def _input_replacer(self, match: re.Match, current_dir: Path, visited_files: set) -> str:
        command = match.group(1)  # 'input' or 'include'
        filename_group = match.group(2).strip() # Content within {} e.g., "myfile.tex" or "myfile" or "main.bbl"

        # Check if the filename_group explicitly ends with .bbl (case-insensitive)
        if filename_group.lower().endswith(".bbl"):
            self._log(f"Skipping expansion of BBL file via TeX include: {match.group(0)}", "debug")
            return match.group(0) # Return the original command, e.g., \input{myfile.bbl}, to leave it untouched by this expansion

        # If not a .bbl file, proceed with normal expansion logic
        resolved_path = self._resolve_include_path(filename_group, current_dir)

        if resolved_path and resolved_path not in visited_files:
            self._log(f"Expanding {command}: {resolved_path.name} (from {current_dir})", "debug")
            visited_files.add(resolved_path)
            try:
                with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f_inc:
                    included_content = f_inc.read()
                # Recursively expand includes within the newly read content
                expanded_included_content = self._recursively_expand_tex_includes(
                    resolved_path, # Pass the path of the file being included
                    included_content,
                    visited_files
                )
                return f"\n% --- Start content from {resolved_path.name} ---\n{expanded_included_content}\n% --- End content from {resolved_path.name} ---\n"
            except Exception as e_inc:
                self._log(f"Could not read included file {resolved_path}: {e_inc}", "warn")
                return match.group(0) # Return original command on error
        elif resolved_path in visited_files:
            self._log(f"Skipping already visited file during expansion: {resolved_path.name}", "debug")
            return f"% Skipped re-inclusion of {resolved_path.name}\n" # Comment out to avoid re-inclusion
        else:
            self._log(f"Could not resolve include path for '{filename_group}' from dir '{current_dir}'. Command: {match.group(0)}", "warn")
            return match.group(0) # Return original command if path not resolved

    def _recursively_expand_tex_includes(self, current_file_path: Path, current_content: str, visited_files: set) -> str:
        # (As in original)
        include_pattern = re.compile(r"\\(input|include)\s*\{([^}]+)\}")
        current_dir = current_file_path.parent
        while True:
            new_content = include_pattern.sub(lambda m: self._input_replacer(m, current_dir, visited_files), current_content)
            if new_content == current_content: break
            current_content = new_content
        return current_content

    def _preprocess_latex_table_environments(self, initial_main_tex_content: str) -> str:

        # (As in original)
        self._log("Preprocessing LaTeX: Expanding includes and then converting table environments...", "debug")
        visited_files = {self.main_tex_path.resolve()} 
        expanded_content = self._recursively_expand_tex_includes(self.main_tex_path, initial_main_tex_content, visited_files)
        self._log(f"LaTeX content expanded to ~{len(expanded_content)//1024} KB before table conversion.", "debug")
        initial_table_content_for_log = expanded_content; processed_table_content = expanded_content
        
        def insert_title_before_document(latex_text: str) -> str:
            """
            Finds the \icmltitle in a LaTeX string and inserts a standard \title{}
            command before \begin{document}.

            Args:
                latex_text: A string containing the LaTeX document content.

            Returns:
                The modified LaTeX string with the \title command added,
                or the original string if \icmltitle is not found.
            """
            # 1. Define the regular expression to find \icmltitle and capture its content.
            #    - \\icmltitle{ : Matches the literal text "\icmltitle{"
            #    - (.*?)       : A non-greedy capture group to match any character until the first '}'
            #    - }           : Matches the closing brace.
            #    - re.DOTALL   : Allows '.' to match newline characters, in case the title spans multiple lines.
            title_pattern = r"\\icmltitle[a-z]*\{([^\}]*)\}"
            
            # 2. Search for the pattern in the input text.
            title_match = re.search(title_pattern, latex_text, re.DOTALL)
            
            # 3. If a match is found, proceed with the replacement.
            if title_match:
                # Extract the captured group (the title text itself).
                # group(0) would be the full match: \icmltitle{The Title}
                # group(1) is just the first captured group: The Title
                title = title_match.group(1)
                # 4. Define the text to be replaced and the new text.
                #    The \n creates a newline for better formatting.
                target_text = r"\\begin{document}"
                replacement_text = r"\\title{" + title + r"}\n\n\n\\begin{document}"
                
                # 5. Use re.sub() to replace the target with the new text.
                #    It replaces only the first occurrence of \begin{document}.
                modified_text = re.sub(target_text, replacement_text, latex_text, count=1)
                return modified_text
            else:
                # If no \icmltitle is found, return the original text without changes.
                return latex_text
            
        processed_table_content = insert_title_before_document(processed_table_content)
        
        # processed_table_content = re.sub(r"\\begin\{table\}.*?(\\begin\{tabular\}.*?\\end\{tabular\}).*?\\end\{table\}", r"\1", processed_table_content, flags=re.DOTALL)
        def replace_table_with_tabular(latex_string):
            """
            Processes 'table' environments in a LaTeX string.
            If a 'table' environment encapsulates a 'tabular' environment (optionally
            wrapped by a 'resizebox'), it reconstructs the 'table' environment to 
            only include:
            - The \begin{table}[options] command.
            - The \caption{...} line (if present).
            - The core \begin{tabular}...\end{tabular} content (with any 'resizebox' 
            wrapper removed).
            - The \label{...} line (if present).
            - The \end{table} command.
            
            Other commands within the table environment (e.g., \centering, \vspace)
            are removed. Table environments that do not contain such a core tabular
            structure are left unchanged.
            """

            # This pattern is used to find relevant 'table' environments and to extract 
            # the block that contains the tabular (which might be a resizebox wrapping it).
            # Group 1: The block containing the tabular content (either 
            #          \resizebox{...}{!} { \begin{tabular}...\end{tabular} } or 
            #          just \begin{tabular}...\end{tabular} )
            pattern_to_find_tables_and_extract_core_block = (
                r"\\begin\{table\}"  # Start of the table environment
                r".*?"              # Non-greedily matches any characters before the core content.
                r"("                # Start of Capturing Group 1 (the block containing tabular).
                    # Option A: A \resizebox command directly wrapping a tabular environment.
                    r"(?:\\resizebox\{[^{}]*\}\{[^{}]*\}\s*\{\s*(?:\\begin\{tabular\}.*?\\end\{tabular\}\s*)\s*\})"
                    r"|"                # OR operator for regex
                    # Option B: A tabular environment not wrapped by an immediate resizebox.
                    r"(?:\\begin\{tabular\}.*?\\end\{tabular\})"
                r")"                # End of Capturing Group 1
                r".*?"              # Non-greedily matches any characters after the core content.
                r"\\end\{table\}"    # End of the table environment
            )

            def custom_replacement_function(match_obj):
                # The entire matched \begin{table}...\end{table} block
                original_table_block = match_obj.group(0)
                # The block containing the tabular (could be resizebox{tabular} or just tabular)
                core_block_containing_tabular = match_obj.group(1).strip()

                # --- Extract the pure tabular content, removing resizebox if present ---
                # Try to match if core_block_containing_tabular is a resizebox wrapping a tabular
                resizebox_tabular_match = re.match(
                    r"\\resizebox\{[^{}]*\}\{[^{}]*\}\s*\{\s*(\\begin\{tabular\}.*?\\end\{tabular\}\s*)\s*\}",
                    core_block_containing_tabular,
                    re.DOTALL
                )
                
                if resizebox_tabular_match:
                    # If it was a resizebox, extract the inner tabular content (Group 1 of this sub-match)
                    core_tabular_content = resizebox_tabular_match.group(1).strip()
                else:
                    # Otherwise, the core_block_containing_tabular was already just the tabular content
                    core_tabular_content = core_block_containing_tabular
                # --- End of pure tabular content extraction ---

                # 1. Extract table options (e.g., [ht]) from the original block
                table_options_match = re.match(r"\\begin\{table\}(\[.*?\])?", original_table_block)
                table_options = table_options_match.group(1) if table_options_match and table_options_match.group(1) else ""

                # 2. Extract caption (if any) from the original block
                caption_match = re.search(r"\\caption\{.*?\}(?:\s*\\par)?", original_table_block, re.DOTALL)
                caption_line = caption_match.group(0).strip() if caption_match else ""
                
                # 3. Extract label (if any) from the original block
                label_match = re.search(r"\\label\{[^{}]*?\}", original_table_block)
                label_line = label_match.group(0).strip() if label_match else ""

                # 4. Reconstruct the new table environment
                new_table_parts = [f"\\begin{{table}}{table_options}"]
                if caption_line:
                    new_table_parts.append(caption_line)
                
                new_table_parts.append(core_tabular_content) # This is now guaranteed to be just the tabular
                
                if label_line:
                    new_table_parts.append(label_line)
                new_table_parts.append("\\end{table}")
                
                return "\n".join(new_table_parts)

            # Apply the substitution.
            modified_latex_string = re.sub(
                pattern_to_find_tables_and_extract_core_block, 
                custom_replacement_function, 
                latex_string, 
                flags=re.DOTALL
            )
            
            return modified_latex_string
        
        # processed_table_content = replace_table_with_tabular(processed_table_content)
        
        def find_matching_closing_brace(text: str, open_brace_index: int) -> int:
            """
            Finds the index of the matching closing brace '}' for an opening brace '{'
            at a given index, respecting nested braces.

            Args:
                text (str): The string to search within.
                open_brace_index (int): The index of the opening brace '{'.

            Returns:
                int: The index of the matching closing brace '}'.

            Raises:
                ValueError: If no matching closing brace is found or if the character
                            at open_brace_index is not an opening brace or index is out of bounds.
            """
            if not (0 <= open_brace_index < len(text) and text[open_brace_index] == '{'):
                raise ValueError(f"Character at index {open_brace_index} is not an opening brace or index is out of bounds.")

            balance = 0
            for i in range(open_brace_index, len(text)):
                if text[i] == '{':
                    balance += 1
                elif text[i] == '}':
                    balance -= 1
                    if balance == 0:
                        return i
            raise ValueError(f"No matching closing brace found for opening brace at index {open_brace_index}.")


        def remove_resizebox_wrapper(latex_string: str) -> str:
            """
            Removes all \resizebox{arg1}{arg2}{content} commands from a LaTeX string,
            keeping only the 'content'. This version uses a brace matching algorithm
            for robustness with nested structures. If a \resizebox command is malformed
            and cannot be parsed, it will be left as is in the output.

            Args:
                latex_string (str): The LaTeX string to process.

            Returns:
                str: The LaTeX string with \resizebox wrappers removed.
            """
            output_parts = []
            last_processed_end = 0 # Tracks the end of the last segment (processed or original)

            # Iterate over all occurrences of "\resizebox"
            for match in re.finditer(r"\\resizebox", latex_string):
                resizebox_start_index = match.start()
                
                # Append text before the current \resizebox
                output_parts.append(latex_string[last_processed_end:resizebox_start_index])
                
                current_parse_ptr = match.end() # Position right after "\resizebox" token

                try:
                    # Helper to skip whitespace and find the next non-whitespace char
                    def advance_to_char(text, char_to_find, start_ptr):
                        p = start_ptr
                        while p < len(text) and text[p].isspace():
                            p += 1
                        if p < len(text) and text[p] == char_to_find:
                            return p
                        # If char_to_find is not found, or end of string reached
                        raise ValueError(f"Expected '{char_to_find}' not found after index {start_ptr} (current: '{text[p]}' at {p})")

                    # Argument 1
                    open_arg1_idx = advance_to_char(latex_string, '{', current_parse_ptr)
                    close_arg1_idx = find_matching_closing_brace(latex_string, open_arg1_idx)
                    current_parse_ptr = close_arg1_idx + 1

                    # Argument 2
                    open_arg2_idx = advance_to_char(latex_string, '{', current_parse_ptr)
                    close_arg2_idx = find_matching_closing_brace(latex_string, open_arg2_idx)
                    current_parse_ptr = close_arg2_idx + 1
                    
                    # Argument 3 (content)
                    open_content_idx = advance_to_char(latex_string, '{', current_parse_ptr)
                    close_content_idx = find_matching_closing_brace(latex_string, open_content_idx)
                    
                    # Extract the content
                    content = latex_string[open_content_idx + 1 : close_content_idx]
                    output_parts.append(content)
                    
                    # Move last_processed_end to after this successfully processed \resizebox command
                    last_processed_end = close_content_idx + 1
                    
                except ValueError as e:
                    # Parsing failed for this \resizebox.
                    # It will be left as is in the output because last_processed_end
                    # is not updated beyond resizebox_start_index for this iteration.
                    # The original text for this segment will be picked up by the next iteration's
                    # output_parts.append(latex_string[last_processed_end:next_match_start])
                    # or by the final append.
                    print(f"Warning: Could not parse resizebox at index {resizebox_start_index}: {e}. Leaving it as is.")
                    # To ensure the failed \resizebox is included, we set last_processed_end
                    # to resizebox_start_index. The segment from last_processed_end (before this match)
                    # to resizebox_start_index has been added. The current \resizebox will be part of
                    # the segment from `last_processed_end` (which is now effectively `resizebox_start_index`
                    # in terms of what's *not yet added from original string*) to the start of the next match.
                    # This means if parsing fails, `last_processed_end` remains where it was before this match was attempted.
                    # The text from `last_processed_end` (value from previous iteration) to `resizebox_start_index` (current match) is added.
                    # The current `\resizebox` (from `resizebox_start_index` onwards) is then part of the next segment
                    # that will be appended if no more `\resizebox` commands are found or before the next one.
                    # This logic correctly leaves unparsable \resizebox commands in the output.
                    pass # last_processed_end is not updated, so original text will be kept

            # Append the rest of the string after the last \resizebox (or if none were found)
            output_parts.append(latex_string[last_processed_end:])
            return "".join(output_parts)
        
        processed_table_content = remove_resizebox_wrapper(processed_table_content)
        
        processed_table_content = re.sub(r'\\cr', r'\\\\', processed_table_content)
        processed_table_content = re.sub(r'\\centering', '', processed_table_content)
        processed_table_content = re.sub(r'\\vspace\{[^}]*\}', '', processed_table_content)
        processed_table_content = re.sub(r'\\setlength\{[^}]*\}\{[^}]*\}', '', processed_table_content)
        processed_table_content = re.sub(r'\\small', '', processed_table_content)
        processed_table_content = re.sub(r'\\hskip', '', processed_table_content)
        
        
        def modify_latex_algorithm_blocks(latex_content):
            """
            Modifies LaTeX content by:
            1. Inserting "Algorithm block\n" after each \begin{algorithmic}[options] line.
            2. Changing \State to State, and \State{arg} to State {arg}.
            3. Changing other specified commands like \If to If, etc., within each
            algorithmic block.
            """
            # List of command names (without backslash) to be processed.
            # 'State' is handled specially and is not needed in this list.
            ALGORITHMIC_COMMANDS_TO_PROCESS = [
                "If", "ElsIf", "Else", "EndIf",
                "For", "ForAll", "EndFor",
                "While", "EndWhile",
                "Repeat", "Until",
                "Loop", "EndLoop",
                "Require", "Ensure", 
                "Return",
                "Print",
                "Call",    
                "Procedure", "EndProcedure"
            ]

            def process_single_algorithmic_block(match_obj):
                """
                Callback function for re.sub. Processes one found algorithmic block.
                - match_obj.group(0) is the entire matched block.
                """
                entire_block_text = match_obj.group(0)

                # Regex to capture:
                # Group 1: \begin{algorithmic}[optional_args]
                # Group 2: The actual content of the algorithm
                # Group 3: \end{algorithmic}
                # This regex handles optional arguments and is non-greedy for content.
                parts_regex = r"(\\begin{algorithmic}(?:\[[^\]]*\])?)(.*?)(\\end{algorithmic})"
                
                block_parts_match = re.match(parts_regex, entire_block_text, re.DOTALL)
                
                if not block_parts_match:
                    # Should not happen if the outer regex matched correctly
                    return entire_block_text 

                start_tag = block_parts_match.group(1)
                inner_content = block_parts_match.group(2)
                end_tag = block_parts_match.group(3)

                modified_inner_content = inner_content
                
                modified_inner_content = re.sub(
                    r"\\State\s*({[^}]*})",  # Matches \State whitespace? {group1}
                    r"\\\\\1",             # \1 refers to the captured group ({[^}]*})
                    modified_inner_content
                )
                # 2. Handle remaining plain \State -> State
                #    This will catch any \State that wasn't followed by {...}
                modified_inner_content = re.sub(
                    r"\\State",             # Matches any remaining \State
                    r"\\\\",               # Replaces with "State"
                    modified_inner_content
                )
                
                modified_inner_content = re.sub(
                    r"\\Comment\s*({[^}]*})",  # Matches \State whitespace? {group1}
                    r" \# comment: \1",             # \1 refers to the captured group ({[^}]*})
                    modified_inner_content
                )

                # --- Handling for other commands: \CmdName -> CmdName ---
                for cmd_name in ALGORITHMIC_COMMANDS_TO_PROCESS:
                    # This regex finds "\CmdName" but ensures it's not part of a longer command
                    # (e.g., matches \For but not \Format if Format is not in our list).
                    # re.escape is used for safety if cmd_name had special characters.
                    
                    find_cmd_regex = r"\\" + re.escape(cmd_name) + r"\s*({[^}]*})"
                    replace_with = r"\\\\ \\textbf{" + re.escape(cmd_name) + r"} \1"
                    modified_inner_content = re.sub(find_cmd_regex, replace_with, modified_inner_content)
                    
                    find_cmd_regex = r"\\" + re.escape(cmd_name) + r"(?![a-zA-Z])"
                    replace_with = r"\\\\" + cmd_name # Replacement: remove the leading backslash
                    modified_inner_content = re.sub(find_cmd_regex, replace_with, modified_inner_content)
                
                # This is where your self._log would go if this were part of a class
                # For example:
                # self._log(f"Processed Block - Start: {start_tag}, End: {end_tag}", "debug")
                # print(f"DEBUG (Simulated Log): {start_tag}\nAlgorithm block\n{modified_inner_content[:200]}...\n{end_tag}")


                # Assemble the block with "Algorithm block\n" inserted after the start_tag
                return f"{start_tag}\nALGORITHM BLOCK (caption below)\n{modified_inner_content}{end_tag}"

            # Regex to find each complete \begin{algorithmic} ... \end{algorithmic} block.
            # - (?:\[[^\]]*\])? handles optional arguments like [1].
            # - .*? is a non-greedy match for the content within the block.
            # - re.DOTALL flag ensures '.' matches newline characters.
            algorithmic_block_finder_regex = r"\\begin{algorithmic}(?:\[[^\]]*\])?.*?\\end{algorithmic}"
            
            # Use re.sub with the callback function to process each found block
            final_modified_content = re.sub(
                algorithmic_block_finder_regex,
                process_single_algorithmic_block,
                latex_content,
                flags=re.DOTALL
            )
            
            return final_modified_content
        processed_table_content = modify_latex_algorithm_blocks(processed_table_content)
        
        def convert_custom_tabular_format(latex_string: str) -> str:
            """
            Converts specific custom tabular column formats like 'y{...}' to 'l'
            and 'x{...}' to 'c', and removes '|', using brace matching for robustness.
            Example: \begin{tabular}{y{53}x{40}|x{30}x{30}x{30}x{30}}
            becomes: \begin{tabular}{lccccc}

            Args:
                latex_string (str): The LaTeX string to process.

            Returns:
                str: The LaTeX string with custom tabular formats converted.
            """
            output_parts = []
            last_processed_end = 0

            for match in re.finditer(r"\\begin\{tabular\}\s*\*?\\?(\[[^\]]*\])?(\[[^\]]*\])?", latex_string):
                begin_tabular_start_index = match.start()
                
                # Append text before the current \begin{tabular}
                output_parts.append(latex_string[last_processed_end:begin_tabular_start_index])
                
                current_parse_ptr = match.end() # Position right after "\begin{tabular}" token

                # Helper to skip whitespace and find the next non-whitespace char
                # (reused from remove_resizebox_wrapper, could be a global helper)
                def advance_to_char(text, char_to_find, start_ptr):
                    p = start_ptr
                    while p < len(text) and text[p].isspace():
                        p += 1
                    if p < len(text) and text[p] == char_to_find:
                        num_backslashes = 0
                        k_adv = p - 1
                        while k_adv >= 0 and text[k_adv] == '\\':
                            num_backslashes += 1
                            k_adv -=1
                        if num_backslashes % 2 == 1: 
                            raise ValueError(f"Found '{char_to_find}' at index {p}, but it is escaped.")
                        return p
                    raise ValueError(f"Expected non-escaped '{char_to_find}' not found after index {start_ptr}. Found '{text[p] if p < len(text) else 'EOF'}' at {p}.")

                # Find the opening brace for column specifiers
                try:
                    open_spec_idx = advance_to_char(latex_string, '{', current_parse_ptr)
                except:
                    self._log("[!] Skipping conversion y{53}x{40}|x{30}x{30}x{30}x{30} for this paper", "warn")
                    output_parts.append(latex_string[last_processed_end:begin_tabular_start_index])
                    last_processed_end = begin_tabular_start_index + 1
                    continue # Move to the next \begin{tabular}
                close_spec_idx = find_matching_closing_brace(latex_string, open_spec_idx)
                
                original_specifiers_with_braces = latex_string[open_spec_idx : close_spec_idx + 1]
                specifiers_content = latex_string[open_spec_idx + 1 : close_spec_idx]

                # Process the specifiers_content
                modified_specifiers = re.sub(r"y\{[^{}]*\}", "l", specifiers_content)
                modified_specifiers = re.sub(r"x\{[^{}]*\}", "c", modified_specifiers)
                # modified_specifiers = modified_specifiers.replace("|", "")
                modified_specifiers = "".join(modified_specifiers.split())
                
                
                
                # Append the \begin{tabular} part, then the modified specifiers in braces
                output_parts.append(latex_string[begin_tabular_start_index : open_spec_idx + 1]) # Includes \begin{tabular}{
                output_parts.append(modified_specifiers)
                output_parts.append("}") # Add the closing brace for specifiers
                
                last_processed_end = close_spec_idx + 1
                    
                # except ValueError as e:
                #     # Parsing failed for this \begin{tabular}{...} specifier block
                #     print(f"Warning: Could not parse tabular specifiers starting near index {begin_tabular_start_index}: {e}. Leaving original segment.")
                #     # If parsing fails, the `last_processed_end` is NOT updated from its value *before* this iteration.
                #     # The current (failed) \begin{tabular} command will be included in the segment appended
                #     # in the next iteration or at the end of the loop.
                #     pass

            # Append the rest of the string after the last processed \begin{tabular}
            output_parts.append(latex_string[last_processed_end:])
            return "".join(output_parts)
        processed_table_content = convert_custom_tabular_format(processed_table_content)
        
        def _advance_to_char_for_parser(text: str, char_to_find: str, start_ptr: int) -> int:
            """Helper to skip whitespace and find the next non-escaped char_to_find."""
            p = start_ptr
            while p < len(text):
                if text[p].isspace():
                    p += 1
                    continue # Skip whitespace

                # At this point, text[p] is a non-whitespace character
                if text[p] == char_to_find:
                    num_backslashes = 0
                    k_adv = p - 1
                    while k_adv >= 0 and text[k_adv] == '\\':
                        num_backslashes += 1
                        k_adv -= 1
                    if num_backslashes % 2 == 1: # If char_to_find is escaped
                        p += 1 # Skip the escaped char and continue search
                        continue
                    return p # Found non-escaped char_to_find
                else:
                    # Found a non-whitespace character that is NOT char_to_find
                    raise ValueError(f"Expected non-escaped '{char_to_find}' after whitespace, but found '{text[p]}' at index {p} (start_ptr was {start_ptr}, at {text[start_ptr:start_ptr+200]}")
            raise ValueError(f"Expected non-escaped '{char_to_find}' not found after index {start_ptr} (end of string reached).")

        def preprocess_latex_for_pandoc(latex_string: str) -> str:
            """
            Preprocesses LaTeX text to simplify custom command definitions for Pandoc conversion.
            Specifically, modifies \newcommand{\cmd}[1]{...#1...} definitions to \newcommand{\cmd}[1]{#1},
            making them pass-through commands for their argument.
            """
            processed_string = latex_string

            # Simplify \newcommand{\cmd}[1]{...#1...} definitions to \newcommand{\cmd}[1]{#1}
            output_parts_newcommand_simplifier = []
            last_processed_end_newcommand_simplifier = 0
            
            # Iterate over all occurrences of "\newcommand"
            for match_newcommand_token in re.finditer(r"\\newcommand\s*\*?(\[[^\]]*\])", processed_string):
                self._log(str(match_newcommand_token) + f". Position starts after {latex_string[match_newcommand_token.start(): match_newcommand_token.end()+200]}", 'debug')
                newcommand_start_idx = match_newcommand_token.start()
                # Append text before the current \newcommand
                output_parts_newcommand_simplifier.append(
                    processed_string[last_processed_end_newcommand_simplifier:newcommand_start_idx]
                )
                current_parse_ptr = match_newcommand_token.end() # Position right after "\newcommand" token
                
                # 1. Parse command name: {\cmdname}
                open_cmd_name_idx = _advance_to_char_for_parser(processed_string, '{', current_parse_ptr)
                close_cmd_name_idx = find_matching_closing_brace(processed_string, open_cmd_name_idx)
                # cmd_name_with_braces = processed_string[open_cmd_name_idx : close_cmd_name_idx + 1] # e.g. {\mycmd}
                current_parse_ptr = close_cmd_name_idx + 1

                # 2. Parse number of arguments: looking for [1]
                # Skip whitespace before potential argument specifier
                temp_ptr = current_parse_ptr
                while temp_ptr < len(processed_string) and processed_string[temp_ptr].isspace():
                    temp_ptr += 1
                
                is_one_arg_cmd = False
                num_args_part_end_ptr = temp_ptr # End of the [1] part, or temp_ptr if no [1]

                if temp_ptr < len(processed_string) and processed_string[temp_ptr] == '[':
                    # Try to match exactly "[1]" with optional spaces
                    num_args_match = re.match(r"\[\s*1\s*\]", processed_string[temp_ptr:])
                    if num_args_match:
                        is_one_arg_cmd = True
                        num_args_part_end_ptr = temp_ptr + len(num_args_match.group(0))
                
                if not is_one_arg_cmd:
                    # This \newcommand is not of the form \newcommand{\cmd}[1]{...}
                    # Append the original \newcommand and its name part, and whatever was parsed as args (or lack thereof)
                    output_parts_newcommand_simplifier.append(
                        processed_string[newcommand_start_idx : num_args_part_end_ptr]
                    )
                    last_processed_end_newcommand_simplifier = num_args_part_end_ptr
                    continue # Move to the next \newcommand token

                current_parse_ptr = num_args_part_end_ptr # Position after [1]

                # 3. Parse definition body: {definition_body}
                open_def_body_idx = _advance_to_char_for_parser(processed_string, '{', current_parse_ptr)
                close_def_body_idx = find_matching_closing_brace(processed_string, open_def_body_idx)
                definition_body = processed_string[open_def_body_idx + 1 : close_def_body_idx]

                # Check if it's a candidate for simplification
                if "#1" in definition_body and definition_body.strip() != "#1":
                    # It's a single-argument command, its definition uses #1, and it's not already just {#1}
                    # Construct the simplified definition: \newcommand{\cmdname}[1]{#1}
                    
                    # Append the part from \newcommand up to and including the opening brace of the definition body
                    output_parts_newcommand_simplifier.append(
                        processed_string[newcommand_start_idx : open_def_body_idx + 1]
                    )
                    output_parts_newcommand_simplifier.append("#1") # The new, simplified body
                    output_parts_newcommand_simplifier.append("}")   # The closing brace for the new body
                    last_processed_end_newcommand_simplifier = close_def_body_idx + 1
                else:
                    # Not a candidate (e.g., no #1, or already just {#1}, or not a [1] arg command)
                    # Append the original, full \newcommand definition segment
                    output_parts_newcommand_simplifier.append(
                        processed_string[newcommand_start_idx : close_def_body_idx + 1]
                    )
                    last_processed_end_newcommand_simplifier = close_def_body_idx + 1

            # Append the rest of the string after the last processed \newcommand
            output_parts_newcommand_simplifier.append(processed_string[last_processed_end_newcommand_simplifier:])
            processed_string = "".join(output_parts_newcommand_simplifier)
            
            # Other simplification steps (like simple \def replacement or two-arg unwraps) are removed
            # as per the request to focus only on \newcommand[1]{...#1...} -> \newcommand[1]{#1}.
                
            return processed_string
        
        processed_table_content = preprocess_latex_for_pandoc(processed_table_content)
        
        def remove_pipes_from_multicolumn_spec(match):
            multicolumn_command = match.group(1); align_spec_with_braces = match.group(2); content = match.group(3)
            cleaned_align_spec_no_braces = align_spec_with_braces[1:-1].replace("|", "")
            return f"{multicolumn_command}{{{cleaned_align_spec_no_braces}}}{content}"
        processed_table_content = re.sub(r"(\\multicolumn\s*\{[^}]*\})(\{[^}|]*?\|[^}]*?\})(\{[^}]*\})", remove_pipes_from_multicolumn_spec, processed_table_content)
        
        
        def modify_multirow_asterisk(latex_string: str) -> str:
            """
            Modifies \multirow commands of the form \multirow{num}{*}{content}
            to \multirow{num}{c}{content}.

            Args:
                latex_string (str): The LaTeX string to process.

            Returns:
                str: The LaTeX string with specified \multirow commands modified.
            """
            # Pattern breakdown:
            # (\\multirow\s*\{[^}{]*\}): Captures Group 1: \multirow{num}
            #   - \\multirow: Matches the command.
            #   - \s*\{[^}{]*\}: Matches the first argument like {2} or {whatever_not_braces}.
            # \s*\*\s*: Matches the asterisk, surrounded by optional whitespace.
            # (\{.*?\}: Captures Group 2: The content argument like {\textit{Metrics}}.
            #   - \{: Matches the opening brace.
            #   - .*?: Non-greedily matches any characters (content).
            #   - \}: Matches the closing brace.
            # re.DOTALL allows .*? to match across newlines if the content spans lines.
            pattern = r"(\\multirow\s*\{[^}{]*\})\s*\*\s*(\{.*?\})"
            
            # Replacement string:
            # \1 refers to the first captured group (\multirow{num}).
            # {c} is the new second argument.
            # \2 refers to the second captured group ({content}).
            replacement = r"\1{c}\2"
            
            modified_string = re.sub(pattern, replacement, latex_string, flags=re.DOTALL)
            return modified_string

        processed_table_content = modify_multirow_asterisk(processed_table_content)
        
        def remove_pipes_from_tabular_spec(match):
            tabular_start = match.group(1); optional_align = match.group(2) if match.group(2) else ""; column_specs_with_braces = match.group(3)
            cleaned_specs_no_braces = column_specs_with_braces[1:-1].replace("|", "")
            return f"{tabular_start}{optional_align}{{{cleaned_specs_no_braces}}}"
        processed_table_content = re.sub(r"(\\begin{tabular})(\[[^\]]*\])?({[^}]*})", remove_pipes_from_tabular_spec, processed_table_content)
        processed_table_content = re.sub(r"\\scalebox{[^}]*}{\s*(\\begin{tabular}[\s\S]*?\\end{tabular})\s*}", r"\1", processed_table_content)
        processed_table_content = re.sub(r"\\scalebox{[^}]*}\[[^\]]*\]{\s*(\\begin{tabular}[\s\S]*?\\end{tabular})\s*}", r"\1", processed_table_content)
        processed_table_content = re.sub(r"\\begin{wraptable}(?:\[[^\]]*\])?\s*\{[^}]*\}\s*\{[^}]*\}", r"\\begin{table}[htb]", processed_table_content)
        processed_table_content = re.sub(r"\\end{wraptable}", r"\\end{table}", processed_table_content)
        processed_table_content = re.sub(r"\\begin{table\*}(?:\[[^\]]*\])?", r"\\begin{table}[htb]", processed_table_content)
        processed_table_content = re.sub(r"\\end{table\*}", r"\\end{table}", processed_table_content)
        processed_table_content = re.sub(r"\\begin{wrapfigure}(\[[^\]]*\])?({[^}]*})?({[^}]*})?({[^}]*})?", r"\\begin{figure}", processed_table_content)
        processed_table_content = re.sub(r"\\end{wrapfigure}", r"\\end{figure}", processed_table_content)
        processed_table_content = re.sub(r"\\begin{minipage}(\[[^\]]*\])?({[^}]*})?({[^}]*})?({[^}]*})?", "", processed_table_content)
        processed_table_content = re.sub(r"\\end{minipage}", "", processed_table_content)
        processed_table_content = re.sub(r"\\captionof{table}(\[[^\]]*\])?({[^}]*})", r"\2", processed_table_content)
        processed_table_content = re.sub(r"\\begin{algorithm}", r"\\begin{figure}", processed_table_content)
        processed_table_content = re.sub(r"\\end{algorithm}", r"\\end{figure}", processed_table_content)
        
        
        processed_table_content = re.sub(r"\\begin{tablenotes}\s*(?:\[.*?\])?([\s\S]*?)\\end{tablenotes}", r"\n% tablenotes content:\n\1\n", processed_table_content)
        processed_table_content = re.sub(r"\\begin{threeparttable}(?:\[[^\]]*\])?", "% threeparttable start removed\n", processed_table_content)
        processed_table_content = re.sub(r"\\end{threeparttable}", "% threeparttable end removed\n", processed_table_content)
        def replace_content_inside_shortstack(match):
            content = match.group(1); return re.sub(r"\\\\\s*", " ", content)
        processed_table_content = re.sub(r"\\shortstack\{((?:[^{}]*|\{[^{}]*\})*)\}", replace_content_inside_shortstack, processed_table_content)
        if processed_table_content != initial_table_content_for_log: self._log("Applied LaTeX table preprocessing to expanded content.", "info")
        else: self._log("No changes made during LaTeX table preprocessing of expanded content.", "debug")
        return processed_table_content

    def _add_paragraph_spacing_to_checklist_block(self, text_block):
        # (As in original)
        lines = text_block.splitlines(); result_lines = []
        if not lines: return ""
        for i, line in enumerate(lines):
            current_stripped = line.strip()
            if result_lines:
                prev_line_in_result_stripped = result_lines[-1].strip()
                current_is_qajg_transformed = current_stripped.startswith(r"{\bf Question:") or current_stripped.startswith(r"{\bf Answer:") or current_stripped.startswith(r"{\bf Justification:") or current_stripped.startswith(r"{\bf Guidelines:")
                prev_was_main_item = prev_line_in_result_stripped.startswith(r"\item") and not prev_line_in_result_stripped.startswith(r"{\bf")
                prev_was_qaj_transformed = prev_line_in_result_stripped.startswith(r"{\bf Question:") or prev_line_in_result_stripped.startswith(r"{\bf Answer:") or prev_line_in_result_stripped.startswith(r"{\bf Justification:")
                if current_is_qajg_transformed and prev_line_in_result_stripped:
                    if prev_was_main_item or prev_was_qaj_transformed:
                        if not (prev_line_in_result_stripped.startswith(r"{\bf Guidelines:") and current_stripped.startswith(r"\begin{itemize}")):
                            result_lines.append("")
            result_lines.append(line)
        final_lines = []; start_idx = 0
        if result_lines:
            while start_idx < len(result_lines) and not result_lines[start_idx].strip(): start_idx += 1
            for j in range(start_idx, len(result_lines)):
                if not (result_lines[j].strip() == "" and final_lines and not final_lines[-1].strip()):
                    final_lines.append(result_lines[j])
            while final_lines and not final_lines[-1].strip(): final_lines.pop()
        return "\n".join(final_lines)

    def _validate_checklist_major_item_structure(self, major_item_lines_list):
        # (As in original)
        if not major_item_lines_list or not re.match(r"^\s*\\item(?!\s*\[)", major_item_lines_list[0].strip()): return False
        cursor = 1
        def _get_next_significant_line_content(lines, current_cursor):
            idx = current_cursor
            while idx < len(lines) and not lines[idx].strip(): idx += 1
            if idx < len(lines): return lines[idx].strip(), idx + 1
            return None, idx
        expected_sequence_items = [r"\item\[\] Question:", r"\item\[\] Answer:", r"\item\[\] Justification:", r"\item\[\] Guidelines:"]
        for i, pattern_start in enumerate(expected_sequence_items):
            line_content, cursor = _get_next_significant_line_content(major_item_lines_list, cursor)
            if not (line_content and line_content.startswith(pattern_start)): return False
            if pattern_start == r"\item\[\] Guidelines:": # Check for itemize after Guidelines
                next_struct_line_content, _ = _get_next_significant_line_content(major_item_lines_list, cursor)
                if not (next_struct_line_content and next_struct_line_content.startswith(r"\begin{itemize}")): return False
        return True

    def _transform_checklist_enumerate_block_content(self, content):
        # (As in original)
        transformed_content = content
        transformed_content = re.sub(r"^(\s*)\\item\[\]\s*(Question|Answer|Justification):\s*(.*)$", r"\1{\\bf \2:} \3", transformed_content, flags=re.MULTILINE)
        transformed_content = re.sub(r"^(\s*)\\item\[\]\s*(Guidelines):\s*$", r"\1{\\bf \2:}", transformed_content, flags=re.MULTILINE)
        spaced_content = self._add_paragraph_spacing_to_checklist_block(transformed_content)
        return spaced_content

    def _preprocess_checklist_enumerations(self, latex_full_text: str) -> str:
        # (As in original)
        self._log("Preprocessing LaTeX checklist 'enumerate' environments...", "debug")
        initial_content = latex_full_text
        def replacement_validator_transformer(matchobj):
            begin_env, content, end_env = matchobj.group(1), matchobj.group(2), matchobj.group(3)
            content_lines = content.splitlines()
            main_item_indices = [i for i, line in enumerate(content_lines) if re.match(r"^\s*\\item(?!\s*\[)", line.strip())]
            if not main_item_indices: return matchobj.group(0)
            major_item_line_blocks = []
            for k in range(len(main_item_indices)):
                start_idx = main_item_indices[k]
                end_idx = main_item_indices[k+1] if k + 1 < len(main_item_indices) else len(content_lines)
                major_item_line_blocks.append(content_lines[start_idx:end_idx])
            if not major_item_line_blocks: return matchobj.group(0)
            all_sub_blocks_are_valid = all(self._validate_checklist_major_item_structure(item_block) for item_block in major_item_line_blocks)
            if all_sub_blocks_are_valid:
                self._log(f"Found a valid QAJG checklist block. Applying transformations.", "debug")
                return begin_env + self._transform_checklist_enumerate_block_content(content) + end_env
            else:
                self._log(f"Enumerate block did not match strict QAJG structure. Leaving unchanged.", "debug")
                return matchobj.group(0)
        fixed_text = re.sub(r"(\\begin{enumerate\})([\s\S]*?)(\\end{enumerate\})", replacement_validator_transformer, latex_full_text)
        if fixed_text != initial_content: self._log("Applied checklist enumeration preprocessing.", "info")
        else: self._log("No changes made during checklist enumeration preprocessing.", "debug")
        return fixed_text

    def _execute_single_pandoc_run(self, tex_content, pandoc_timeout, strategy_idx, retry_attempt_idx):
        debug_filename = f"l2m_debug_strategy_{strategy_idx}_initial.tex"
        if retry_attempt_idx > 0:
            debug_filename = f"l2m_debug_strategy_{strategy_idx}_retry_{retry_attempt_idx}.tex"
        self._write_debug_tex_if_needed(tex_content, debug_filename)

        final_md_path = self.final_output_folder_path / "paper.md"
        pandoc_local_out_basename = "_l2m_pandoc_temp_paper.md" # Basename for temp output
        
        tmp_tex_path_obj = None
        
        try:
            # Create temp .tex file in the project's root for Pandoc
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".tex", encoding="utf-8", dir=self.folder_path) as tmp_f:
                tmp_f.write(tex_content)
                tmp_tex_path_obj = Path(tmp_f.name)

            # Command uses relative paths for input/output, cwd will handle context
            cmd = ["pandoc", tmp_tex_path_obj.name, "-f", "latex", "-t", "markdown_mmd-tex_math_dollars", "--verbose", "--wrap=none", "-o", pandoc_local_out_basename]
            if self.template_path and Path(self.template_path).exists():
                cmd.extend(["--template", str(self.template_path)]) # Template path should be absolute or resolvable by Pandoc from cwd
            else:
                if self.template_path: # Only log if a template was specified but not found
                    self._log(f"Pandoc template '{self.template_path}' not found. Using Pandoc default.", "info")


            self._log(f"Running Pandoc (Strategy {strategy_idx}, Retry {retry_attempt_idx}) in '{self.folder_path}': {' '.join(cmd)}", "debug")
            
            # Execute Pandoc with cwd set to the project folder
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, 
                                  encoding='utf-8', errors='ignore', timeout=pandoc_timeout, 
                                  cwd=self.folder_path) # Key change: use cwd
            
            created_md_path_local = self.folder_path / pandoc_local_out_basename # Path to temp MD output in project folder

            if proc.returncode == 0 and created_md_path_local.exists():
                self._log("Pandoc conversion successful for this attempt.", "debug")
                self.final_output_folder_path.mkdir(parents=True, exist_ok=True)
                shutil.move(str(created_md_path_local), str(final_md_path)) # Move to final output
                
                if final_md_path.exists():
                    self._log(f"Markdown file generated at: {final_md_path}", "debug")
                    with open(final_md_path, "r", encoding="utf-8", errors="ignore") as md_f:
                        markdown_content_after_pandoc = md_f.read()
                    
                    output_figures_subdir = self.final_output_folder_path / "figures"
                    # output_figures_subdir.mkdir(parents=True, exist_ok=True) # Already done when moving MD
                    
                    copied_figure_details = self._find_and_copy_figures_from_markdown(
                        markdown_content_after_pandoc, output_figures_subdir
                    )
                    updated_markdown_content = self._convert_and_update_figure_paths_in_markdown(
                        markdown_content_after_pandoc, copied_figure_details, self.final_output_folder_path
                    )
                    with open(final_md_path, "w", encoding="utf-8") as md_f:
                        md_f.write(updated_markdown_content)
                    
                    if copied_figure_details: self._log("Markdown figure paths updated and relevant images processed.", "success")
                    else: self._log("No local figures found in Markdown to process.", "debug")
                    return True, proc.stdout, proc.stderr
                else:
                    self._log(f"Pandoc seemed to succeed but final markdown file '{final_md_path}' not found after move.", "error")
                    return False, proc.stdout, proc.stderr
            else:
                self._log(f"Pandoc conversion failed. RC: {proc.returncode}.", "error")
                if created_md_path_local.exists(): # Clean up temp MD if created despite error
                    try: created_md_path_local.unlink()
                    except Exception as e_del_tmp: self._log(f"Could not delete temp pandoc output '{created_md_path_local}': {e_del_tmp}", "warn")
                return False, proc.stdout, proc.stderr

        except subprocess.TimeoutExpired:
            self._log(f"Pandoc command timed out after {pandoc_timeout} seconds.", "error")
            return False, "", "Pandoc command timed out."
        except Exception as e:
            self._log(f"An exception occurred during this Pandoc run: {e}", "error")
            return False, "", str(e)
        finally:
            if tmp_tex_path_obj and tmp_tex_path_obj.exists():
                try: tmp_tex_path_obj.unlink()
                except Exception as e_unlink: self._log(f"Could not delete temporary TeX file '{tmp_tex_path_obj}': {e_unlink}", "warn")
            # Ensure temp MD in source dir is cleaned if it wasn't moved (e.g. on error before move)
            pandoc_temp_output_in_src = self.folder_path / pandoc_local_out_basename
            if pandoc_temp_output_in_src.exists():
                try: pandoc_temp_output_in_src.unlink()
                except Exception as e_del_tmp2: self._log(f"Could not delete temp pandoc output '{pandoc_temp_output_in_src}' from source folder: {e_del_tmp2}", "warn")

    def _run_pandoc_strategy_with_retries(self, tex_content_for_pandoc, pandoc_timeout, strategy_idx):
        # Manages the iterative commenting and retrying for a single Pandoc strategy.
        current_tex_content = tex_content_for_pandoc
        commented_line_indices = set() # Store 0-indexed line numbers that have been commented

        for retry_attempt in range(MAX_PANDOC_COMMENT_RETRIES + 1): # +1 for the initial attempt
            success, stdout, stderr = self._execute_single_pandoc_run(
                current_tex_content, pandoc_timeout, strategy_idx, retry_attempt
            )

            if success:
                return True # This strategy succeeded

            # Pandoc failed for this attempt
            self._log(f"Pandoc STDOUT (Strategy {strategy_idx}, Retry {retry_attempt}): {stdout[:500]}...", "debug")
            self._log(f"Pandoc STDERR (Strategy {strategy_idx}, Retry {retry_attempt}): {stderr[:1000]}...", "error")

            if retry_attempt < MAX_PANDOC_COMMENT_RETRIES:
                error_line_num_1_indexed = self._parse_pandoc_error_line_number(stderr)
                if error_line_num_1_indexed is not None:
                    error_line_idx_0_indexed = error_line_num_1_indexed - 1

                    if error_line_idx_0_indexed in commented_line_indices:
                        self._log(f"Line {error_line_num_1_indexed} was already commented in a previous retry for this strategy. Error likely persists or shifted. Stopping retries for this strategy.", "error")
                        return False # Fail this strategy

                    tex_lines = current_tex_content.splitlines()
                    if 0 <= error_line_idx_0_indexed < len(tex_lines):
                        original_line_content = tex_lines[error_line_idx_0_indexed]
                        # Check if line is already a L2M comment to prevent double-commenting by this script
                        if original_line_content.strip().startswith("% L2M_PANDOC_ERROR_COMMENT:"):
                             self._log(f"Line {error_line_num_1_indexed} seems to be already commented by L2M. Error might be elsewhere or unfixable by this method. Stopping retries for this strategy.", "warn")
                             return False


                        tex_lines[error_line_idx_0_indexed] = f"% L2M_PANDOC_ERROR_COMMENT (Strategy {strategy_idx}, Retry {retry_attempt+1}): Original: {original_line_content.strip()}"
                        current_tex_content = "\n".join(tex_lines)
                        commented_line_indices.add(error_line_idx_0_indexed)
                        self._log(f"Commenting out line {error_line_num_1_indexed} and retrying Pandoc.", "info")
                        # Continue to the next retry_attempt
                    else:
                        self._log(f"Pandoc reported error on line {error_line_num_1_indexed}, which is out of bounds for the TeX content ({len(tex_lines)} lines). Stopping retries for this strategy.", "error")
                        return False # Fail this strategy
                else:
                    self._log("Could not parse specific error line from Pandoc stderr. Cannot iteratively comment. Stopping retries for this strategy.", "error")
                    return False # Fail this strategy
            else: # Max retries for commenting reached
                self._log(f"Max Pandoc comment retries ({MAX_PANDOC_COMMENT_RETRIES}) reached for strategy {strategy_idx}.", "error")
                return False # Fail this strategy
        
        return False # Should be reached if all retries fail

    def convert_to_markdown(self, output_folder_path_str):
        overall_start_time = time.time()
        MAX_OVERALL_PROCESSING_TIME = 1200  # 20 minutes

        def check_overall_timeout(stage_name=""):
            if (time.time() - overall_start_time) > MAX_OVERALL_PROCESSING_TIME:
                self._log(f"Overall processing timed out after {MAX_OVERALL_PROCESSING_TIME} seconds (during {stage_name}).", "error")
                return True
            return False

        if check_overall_timeout("initialization"): return False
        if not self.find_main_tex_file(): return False
        if check_overall_timeout("find_main_tex_file"): return False
        
        self.final_output_folder_path = Path(output_folder_path_str).resolve()
        try:
            self.final_output_folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._log(f"Output dir error: {e}", "error"); return False
        if check_overall_timeout("output_folder_setup"): return False

        processed_bbl_with_abstracts_and_keys = self._generate_bbl_content()
        if check_overall_timeout("_generate_bbl_content"): return False
        if processed_bbl_with_abstracts_and_keys is None:
            self._log("Failed to generate or process BBL content. This might be due to an earlier timeout or other errors.", "error")
            self._cleanup_debug_files(); return False # Cleanup and exit

        # problematic_macros = r"""\providecommand{\linebreakand}{\par\noindent\ignorespaces} \providecommand{\email}[1]{\texttt{#1}} \providecommand{\IEEEauthorblockN}[1]{#1\par} \providecommand{\IEEEauthorblockA}[1]{#1\par} \providecommand{\and}{\par\noindent\ignorespaces} \providecommand{\And}{\par\noindent\ignorespaces} \providecommand{\AND}{\par\noindent\ignorespaces} \providecommand{\IEEEoverridecommandlockouts}{} \providecommand{\CLASSINPUTinnersidemargin}{} \providecommand{\CLASSINPUToutersidemargin}{} \providecommand{\CLASSINPUTtoptextmargin}{} \providecommand{\CLASSINPUTbottomtextmargin}{} \providecommand{\CLASSOPTIONcompsoc}{} \providecommand{\CLASSOPTIONconference}{} \providecommand{\@toptitlebar}{} \providecommand{\@bottomtitlebar}{} \providecommand{\@thanks}{} \providecommand{\@notice}{} \providecommand{\@noticestring}{} \providecommand{\acksection}{} \newenvironment{ack}{\par\textbf{Acknowledgments}\par}{\par} \providecommand{\answerYes}[1]{[Yes] ##1} \providecommand{\answerNo}[1]{[No] ##1} \providecommand{\answerNA}[1]{[NA] ##1} \providecommand{\answerTODO}[1]{[TODO] ##1} \providecommand{\justificationTODO}[1]{[TODO] ##1} \providecommand{\textasciitilde}{~} \providecommand{\textasciicircum}{^} \providecommand{\textbackslash}{\symbol{92}}"""
        problematic_macros = ""
        pandoc_attempts_config = [
            {"mode": "venue_only", "desc": "Venue-specific styles commented", "timeout_factor": 1.0}, # Base timeout
            # {"mode": "original", "desc": "Original TeX content (no script style commenting)", "timeout_factor": 1.0}, # Can be re-enabled
            {"mode": "all_project", "desc": "All project-specific styles commented", "timeout_factor": 1.2} # Slightly more time for potentially larger file
        ]
        
        initial_main_tex_content_for_processing = self.original_main_tex_content
        expanded_and_table_processed_tex = self._preprocess_latex_table_environments(initial_main_tex_content_for_processing)
        if check_overall_timeout("_preprocess_latex_table_environments"): self._cleanup_debug_files(); return False
        
        fully_preprocessed_tex = self._preprocess_checklist_enumerations(expanded_and_table_processed_tex)
        if check_overall_timeout("_preprocess_checklist_enumerations"): self._cleanup_debug_files(); return False

        base_pandoc_timeout_per_strategy = 60 # seconds for one strategy before iterative commenting kicks in

        for strategy_idx, attempt_config in enumerate(pandoc_attempts_config):
            if check_overall_timeout(f"before Pandoc strategy {strategy_idx}"): self._cleanup_debug_files(); return False
            
            self._log(f"Pandoc Conversion Strategy {strategy_idx + 1}/{len(pandoc_attempts_config)} ({attempt_config['desc']})...", "info")
            
            current_tex_base = fully_preprocessed_tex # Start with the fully preprocessed TeX for each strategy
            if attempt_config["mode"] == "venue_only":
                style_modified_tex = self._comment_out_style_packages(current_tex_base, mode="venue_only")
            elif attempt_config["mode"] == "all_project":
                style_modified_tex = self._comment_out_style_packages(current_tex_base, mode="all_project")
            else: # "original" or other modes
                style_modified_tex = current_tex_base
            if check_overall_timeout(f"style_commenting_for_strategy_{strategy_idx}"): self._cleanup_debug_files(); return False

            tex_with_final_bib = self._simple_inline_processed_bbl(style_modified_tex, processed_bbl_with_abstracts_and_keys)
            if check_overall_timeout(f"bbl_inlining_for_strategy_{strategy_idx}"): self._cleanup_debug_files(); return False
            
            final_tex_for_pandoc = problematic_macros + "\n" + tex_with_final_bib

            time_elapsed_so_far = time.time() - overall_start_time
            remaining_overall_time = MAX_OVERALL_PROCESSING_TIME - time_elapsed_so_far
            
            if remaining_overall_time <= 5: # Need at least a few seconds for Pandoc
                self._log(f"Overall processing time limit reached before Pandoc strategy {strategy_idx} could run effectively.", "error")
                self._cleanup_debug_files(); return False
            
            # Calculate timeout for this specific strategy attempt
            # This timeout is for each call to _execute_single_pandoc_run within the retry loop
            effective_pandoc_run_timeout = min(base_pandoc_timeout_per_strategy * attempt_config.get("timeout_factor", 1.0), remaining_overall_time - 2) # -2 for buffer
            effective_pandoc_run_timeout = max(5, effective_pandoc_run_timeout) # Minimum 5s per run

            self._log(f"Running Pandoc strategy {strategy_idx} with per-run timeout: {effective_pandoc_run_timeout:.2f}s", "debug")

            if self._run_pandoc_strategy_with_retries(final_tex_for_pandoc, effective_pandoc_run_timeout, strategy_idx):
                self._log(f"Pandoc conversion successful with strategy: {attempt_config['desc']}.", "success")
                self._cleanup_debug_files(); return True # Overall success

            if check_overall_timeout(f"after Pandoc strategy {strategy_idx}"): self._cleanup_debug_files(); return False

            if strategy_idx < len(pandoc_attempts_config) - 1:
                self._log(f"Pandoc strategy {strategy_idx + 1} ({attempt_config['desc']}) failed. Trying next strategy.", "warn")
            else:
                self._log("All Pandoc conversion strategies failed.", "error")
        
        self._cleanup_debug_files()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts LaTeX to Markdown with advanced abstract fetching and table processing.")
    parser.add_argument("project_folder", help="Path to LaTeX project folder.")
    parser.add_argument("-o", "--output_folder", default=None, help="Output folder. Default: '[project_name]_output'.")
    parser.add_argument("-q", "--quiet", action="count", default=0, help="Suppress info messages.")
    parser.add_argument("--template", default="template.md", help="Pandoc Markdown template. Default: 'template.md'.")
    parser.add_argument("--openalex-email", default=os.environ.get("OPENALEX_EMAIL", "your-email@example.com"), help="Your email for OpenAlex API. Can also be set via OPENALEX_EMAIL environment variable.")
    parser.add_argument("--openalex-api-key", default=os.environ.get("OPENALEX_API_KEY"), help="Your OpenAlex API key. Can also be set via OPENALEX_API_KEY environment variable.")
    parser.add_argument("--elsevier-api-key", default=os.environ.get("ELSEVIER_API_KEY"), help="Your Elsevier API key. Can also be set via ELSEVIER_API_KEY environment variable.")
    parser.add_argument("--springer-api-key", default=os.environ.get("SPRINGER_API_KEY"), help="Your Springer Nature API key. Can also be set via SPRINGER_API_KEY environment variable.")
    parser.add_argument("--semantic-scholar-api-key", default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"), help="Your SEMANTIC_SCHOLAR_API_KEY. Can also be set via SEMANTIC_SCHOLAR_API_KEY environment variable.")
    parser.add_argument("--poppler-path", default=os.environ.get("POPPLER_PATH"), help="Path to Poppler binaries directory for pdf2image. Can also be set via POPPLER_PATH environment variable.")
    parser.add_argument("--output-debug-tex", action="store_true", help="Keep the l2m_debug_*.tex file(s) in the output/project folder after conversion.")


    args = parser.parse_args()
    project_path = Path(args.project_folder).resolve()
    if not project_path.is_dir(): print(f"Error: Project folder '{args.project_folder}' not found.", file=sys.stderr); sys.exit(1)
    output_folder_path = Path(args.output_folder).resolve() if args.output_folder else project_path.parent / f"{project_path.name}_output"

    template_fpath_resolved = Path(args.template)
    final_template_path_str = None
    if template_fpath_resolved.is_file() and template_fpath_resolved.exists(): # Check absolute/relative path first
        final_template_path_str = str(template_fpath_resolved.resolve())
    else: # Check relative to script, then CWD
        script_dir_template = (Path(__file__).parent / args.template).resolve()
        if script_dir_template.is_file() and script_dir_template.exists():
            final_template_path_str = str(script_dir_template)
            print(f"[*] Info: Template '{args.template}' not found directly, using template from script directory: {final_template_path_str}")
        else:
            cwd_template = (Path.cwd() / args.template).resolve()
            if cwd_template.is_file() and cwd_template.exists():
                final_template_path_str = str(cwd_template)
                print(f"[*] Info: Template '{args.template}' not found directly or in script dir, using template from CWD: {final_template_path_str}")
            else:
                print(f"[!] Warning: Pandoc template '{args.template}' not found in standard locations. Pandoc will use its default.", file=sys.stderr)

    converter = LatexToMarkdownConverter(
        str(project_path),
        quiet_level=args.quiet,
        template_path=final_template_path_str,
        openalex_email=args.openalex_email,
        openalex_api_key=args.openalex_api_key,
        elsevier_api_key=args.elsevier_api_key,
        springer_api_key=args.springer_api_key,
        semantic_scholar_api_key=args.semantic_scholar_api_key,
        poppler_path=args.poppler_path,
        output_debug_tex=args.output_debug_tex # Pass the new flag
    )
    converter._log(f"Processing LaTeX project in: '{project_path}'", "info")

    if converter.convert_to_markdown(str(output_folder_path)):
        converter._log(f"Conversion successful. Output: '{output_folder_path / 'paper.md'}'", "success")
    else: converter._log("Conversion failed.", "error")

    converter._log("Dependencies: Pandoc, LaTeX (pdflatex, bibtex), requests.", "info")
    if converter.PdfReader is None: converter._log("Optional for PDF abstracts: PyPDF2 (pip install PyPDF2)", "info")
    if converter.BeautifulSoup is None: converter._log("Optional for HTML abstracts: beautifulsoup4 (pip install beautifulsoup4)", "info")
    if converter.PILImage is None: converter._log("Optional for image conversion: Pillow (pip install Pillow)", "info")
    if converter.convert_from_path is None: converter._log("Optional for PDF to PNG: pdf2image (pip install pdf2image) and Poppler.", "info")

