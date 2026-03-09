"""
Generate or update README.md files with directory structure.

For a given root folder and all its subfolders, this script:
- Scans the directory tree
- Skips folders that have no subdirectories
- Creates or updates a README.md with a "Directory Structure" section

Usage:
    python generate_readme.py <target_directory> [options]

Options:
    --dry-run       Preview changes without writing files
    --ignore        Comma-separated list of directory names to ignore
    --max-depth     Maximum depth to recurse (default: unlimited)
"""
    
import os
import re
from datetime import datetime
from pathlib import Path


# Directories to always ignore
DEFAULT_IGNORE = {
    ".git", ".svn", ".hg",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".venv", "venv", "env",
    ".idea", ".vscode",
    ".tox", ".eggs", "dist", "build", '.zarr'
}

SECTION_HEADING = "# Directory structure"
MAX_SUBDIRS = 100


def get_subdirs(directory: Path, ignore: set[str]) -> list[Path]:
    """Return sorted list of visible subdirectories, excluding ignored names."""
    try:
        entries = sorted(directory.iterdir())
    except PermissionError:
        return []
    return [
        e for e in entries
        if e.is_dir() and not e.name.startswith(".") and e.name not in ignore
    ]

def build_tree_lines(
    directory: Path,
    ignore: set[str],
    prefix: str = "",
    max_depth: int | None = None,
    current_depth: int = 0,
) -> list[str]:
    """Build a list of strings representing the directory tree (directories only)."""
    if max_depth is not None and current_depth >= max_depth:
        return []

    # Only include directories, skip files entirely
    entries = [
        e for e in directory.iterdir()
        if e.is_dir()
        and not e.name.startswith(".")
        and e.name not in ignore
    ]

    # Truncate if too many subdirs: show first, ellipsis, last
    if len(entries) > MAX_SUBDIRS:
        entries = entries[:1] + [None] + entries[-1:]
    else:
        entries.sort(key=lambda e: e.name.lower())

    lines = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "

        # None is the ellipsis placeholder
        if entry is None:
            lines.append(f"{prefix}{connector}...")
            continue

        lines.append(f"{prefix}{connector}{entry.name}/")

        # Only recurse if the child itself is within the limit
        child_subdirs = get_subdirs(entry, ignore)
        if 0 < len(child_subdirs) <= MAX_SUBDIRS:
            extension = "    " if is_last else "│   "
            subtree = build_tree_lines(
                entry, ignore, prefix + extension, max_depth, current_depth + 1
            )
            lines.extend(subtree)

    return lines


def generate_code_block(directory: Path, ignore: set[str], max_depth: int | None) -> str:
    """Generate the ```text code block content for the directory structure."""
    tree_lines = build_tree_lines(directory, ignore, max_depth=max_depth)
    tree_str = "\n".join(tree_lines)

    return f"""```text
{directory.name}/
{tree_str}
```"""


def update_readme(readme_path: Path, code_block: str) -> str:
    """Update an existing README.md by replacing the code block under
    '# Directory Structure' (or '# Directory structure').

    If the heading exists with a fenced code block beneath it, only the code
    block is replaced — all other content in the section is preserved.
    If the heading exists but has no code block, the code block is inserted
    right after the heading.
    If the heading doesn't exist, the full section is appended at the end.
    Returns the new content."""

    today = datetime.now().strftime("%Y-%m-%d")
    full_section = f"{SECTION_HEADING}\n\n*Updated: {today}*\n\n{code_block}"

    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")

        # Find the heading (case-insensitive for "Structure" / "structure")
        heading_pattern = re.compile(
            r"^# [Dd]irectory [Ss]tructure\s*$",
            re.MULTILINE,
        )
        heading_match = heading_pattern.search(content)

        if heading_match:
            # Determine the boundary of this H1 section (up to next H1 or EOF)
            rest_after_heading = content[heading_match.end():]
            next_h1 = re.search(r"^# ", rest_after_heading, re.MULTILINE)
            if next_h1:
                section_end = heading_match.end() + next_h1.start()
            else:
                section_end = len(content)

            section_body = content[heading_match.end():section_end]

            # Find the fenced code block within this section
            code_block_pattern = re.compile(
                r"```(?:text)?\s*\n.*?```",
                re.DOTALL,
            )
            code_match = code_block_pattern.search(section_body)

            if code_match:
                # Replace only the code block
                abs_start = heading_match.end() + code_match.start()
                abs_end = heading_match.end() + code_match.end()
                new_content = content[:abs_start] + code_block + content[abs_end:]
            else:
                # No code block found — insert right after heading
                insert_pos = heading_match.end()
                new_content = content[:insert_pos] + "\n\n" + code_block + content[insert_pos:]

            return new_content
        else:
            # Heading not found — append the full section
            return content.rstrip("\n") + "\n\n" + full_section + "\n"
    else:
        # Brand new README
        return full_section + "\n"



def has_subdirectories(directory: Path, ignore: set[str]) -> bool:
    """Check if a directory contains at least one (but no more than 1000) subdirectories."""
    subdirs = get_subdirs(directory, ignore)
    return 0 < len(subdirs) <= MAX_SUBDIRS


def update_all_readmes(
    data_dir: str,
    ignore: set[str] = DEFAULT_IGNORE,
    dry_run: bool=False,
    max_depth: int | None=None,
    **kwargs,
) -> list[str]:
    """Process a single directory and all eligible subdirectories.
    Returns a list of paths where README.md was created/updated."""
    modified = []

    # Walk all directories (including root)
    data_dir = Path(data_dir).expanduser()
    for dirpath, dirnames, _filenames in os.walk(data_dir):
        current = Path(dirpath)

        # Prune ignored directories from the walk
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in ignore
        ]

        # Skip if no subdirectories
        if not has_subdirectories(current, ignore):
            continue

        readme_path = current / "README.md"
        print(readme_path)
        code_block = generate_code_block(current, ignore, max_depth)
        new_content = update_readme(readme_path, code_block)

        action = "Updated" if readme_path.exists() else "Created"

        if dry_run:
            print(f"[DRY RUN] Would {action.lower()}: {readme_path}")
            print("-" * 60)
            print(new_content)
            print("-" * 60)
        else:
            readme_path.write_text(new_content, encoding="utf-8")
            print(f"{action}: {readme_path}")

        modified.append(str(readme_path))

    return modified