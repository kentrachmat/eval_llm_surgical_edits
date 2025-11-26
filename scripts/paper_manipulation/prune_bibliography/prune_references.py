#!/usr/bin/env python3
"""
Script to copy a folder structure and prune References sections from all markdown files.

This script:
1. Copies the entire folder structure from source to destination
2. For all markdown files (*.md), removes everything from "# References" tag onwards
"""

import os
import shutil
import argparse
import re
from pathlib import Path


def prune_references_from_markdown(content: str) -> str:
    """
    Remove everything from "# References" onwards in markdown content.
    
    Handles variations like:
    - # References
    - # References [references]
    - ## References
    - #References (no space)
    
    Args:
        content: The markdown file content
        
    Returns:
        The content with References section and everything after it removed
    """
    # Pattern to match "# References" or "## References" at the start of a line
    # Also handles variations like "# References [references]" or "#References"
    pattern = r'^#+\s*References\b.*$'
    
    lines = content.split('\n')
    pruned_lines = []
    
    for line in lines:
        # Check if this line matches the References header
        if re.match(pattern, line, re.IGNORECASE):
            # Stop adding lines once we hit References
            break
        pruned_lines.append(line)
    
    # Join the lines back together
    return '\n'.join(pruned_lines)


def copy_and_prune_folder(source_dir: Path, dest_dir: Path, verbose: bool = False):
    """
    Recursively copy folder structure and prune References from markdown files.
    
    Args:
        source_dir: Source directory path
        dest_dir: Destination directory path
        verbose: Whether to print progress messages
    """
    source_dir = Path(source_dir).resolve()
    dest_dir = Path(dest_dir).resolve()
    
    if not source_dir.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    if dest_dir.exists():
        raise ValueError(f"Destination directory already exists: {dest_dir}")
    
    # Counters for statistics
    total_files = 0
    markdown_files = 0
    pruned_files = 0
    
    # Walk through all files and directories
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path from source
        rel_path = Path(root).relative_to(source_dir)
        dest_root = dest_dir / rel_path
        
        # Create destination directory
        dest_root.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Processing directory: {rel_path}")
        
        # Process each file
        for file in files:
            source_file = Path(root) / file
            dest_file = dest_root / file
            
            total_files += 1
            
            # Check if it's a markdown file
            if file.lower().endswith('.md'):
                markdown_files += 1
                
                # Read the markdown file
                try:
                    with open(source_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Prune References section
                    pruned_content = prune_references_from_markdown(content)
                    
                    # Check if content was actually pruned
                    if len(pruned_content) < len(content):
                        pruned_files += 1
                        if verbose:
                            print(f"  Pruned References from: {rel_path / file}")
                    
                    # Write pruned content to destination
                    with open(dest_file, 'w', encoding='utf-8') as f:
                        f.write(pruned_content)
                        
                except Exception as e:
                    print(f"Error processing {source_file}: {e}")
                    # Copy original file if there's an error
                    shutil.copy2(source_file, dest_file)
            else:
                # For non-markdown files, just copy as-is
                shutil.copy2(source_file, dest_file)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Markdown files: {markdown_files}")
    print(f"  Files with References pruned: {pruned_files}")
    print(f"  Output directory: {dest_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Copy folder structure and prune References sections from markdown files'
    )
    parser.add_argument(
        'source_dir',
        type=str,
        help='Source directory path (e.g., path to NeurIPS2024 folder)'
    )
    parser.add_argument(
        'dest_dir',
        type=str,
        help='Destination directory path (will be created)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose progress messages'
    )
    
    args = parser.parse_args()
    
    try:
        copy_and_prune_folder(
            Path(args.source_dir),
            Path(args.dest_dir),
            verbose=args.verbose
        )
        print("\nDone!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

