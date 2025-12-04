# This script will download the latest version instead of v1

# ICLR 2024
python run_multithreaded_latex.py --venue ICLR.cc/2024/Conference --output_dir parsed_arxiv/ICLR2024_latest --max_workers 8 --arxiv_version 0 

# ICLR 2025
python run_multithreaded_latex.py --venue ICLR.cc/2025/Conference --output_dir parsed_arxiv/ICLR2025_latest --max_workers 8 --arxiv_version 0 
