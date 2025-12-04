# LaTeX to Markdown Converter
This tool converts LaTeX projects to Markdown. It supports single-project conversion and batch processing of papers from OpenReview venues. It can enrich bibliographies by fetching abstracts from online sources.

## 1. Prerequisites
### System Dependencies
- Pandoc (v3.7.0.2 recommended)

- TeX Live (with pdflatex and bibtex)

- Poppler

### Python Dependencies
See requirements.txt.

## 2. Setup
### Install Python packages:
`pip install -r requirements.txt`
Create and configure the .env file:
Create a file named .env in the project root. Add the following keys as needed.
```bash
# Required for OpenReview batch mode
OPENREVIEW_USERNAME="your_user@email.com"
OPENREVIEW_PASSWORD="your_password"

# Required for bibliography abstract fetching
OPENALEX_EMAIL="your_user@email.com"
ELSEVIER_API_KEY="your_key"
SPRINGER_API_KEY="your_key"
SEMANTIC_SCHOLAR_API_KEY="your_key"

# Optional: If poppler is not in system PATH
# POPPLER_PATH="/path/to/poppler/bin"
```
## 3. Usage
The tool has two primary modes.
### Mode 1: Convert a Single Project
Use `latex.py` for converting a local LaTeX project.
**Command:**
`python3 latex.py /path/to/latex_project -o /path/to/output`
- `project_folder`: Path to the folder containing .tex source.
- `-o`: (Optional) Output directory. Defaults to `[project_name]_output`.

## Mode 2: Batch Convert from OpenReview

Use `run_multithreaded_latex.py` to download and convert papers from an OpenReview venue in parallel.**Command:**
`python3 run_multithreaded_latex.py --venue "ICLR.cc/2025/Conference" --max_workers 8`
- --`venue`: The ID for the OpenReview venue.
-  `--num_accepted`: (Optional) Number of accepted papers to process.
-  `--num_rejected`: (Optional) Number of rejected papers to process.
-  `--max_workers`: (Optional) Number of parallel threads to use.

Run any script with -h or --help to see all available options.