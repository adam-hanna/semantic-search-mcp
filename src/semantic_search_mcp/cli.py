"""CLI commands for semantic-search-mcp."""

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_skills_source_dir() -> Path:
    """Get the path to the bundled skills directory."""
    return Path(__file__).parent / "skills"


def get_skills_target_dir() -> Path:
    """Get the user's Claude Code skills directory."""
    return Path.home() / ".claude" / "skills"


def _install_skills_core() -> tuple[list[str], list[str]]:
    """Core skill installation logic.

    Returns:
        Tuple of (installed_skills, updated_skills)

    Raises:
        FileNotFoundError: If skills source directory doesn't exist
        ValueError: If no skills found in package
    """
    source_dir = get_skills_source_dir()
    target_dir = get_skills_target_dir()

    if not source_dir.exists():
        raise FileNotFoundError(f"Skills source directory not found: {source_dir}")

    # Find all skill directories (contain SKILL.md)
    skill_dirs = [d for d in source_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]

    if not skill_dirs:
        raise ValueError("No skills found in package")

    # Create target directory if needed
    target_dir.mkdir(parents=True, exist_ok=True)

    installed = []
    updated = []

    for skill_dir in skill_dirs:
        skill_name = skill_dir.name
        target_skill_dir = target_dir / skill_name

        if target_skill_dir.exists():
            # Update existing
            shutil.rmtree(target_skill_dir)
            shutil.copytree(skill_dir, target_skill_dir)
            updated.append(skill_name)
        else:
            # Install new
            shutil.copytree(skill_dir, target_skill_dir)
            installed.append(skill_name)

    return installed, updated


def install_skills_silent() -> None:
    """Install skills silently. Used during server startup.

    Logs errors but doesn't print to stdout.
    """
    try:
        installed, updated = _install_skills_core()
        if installed:
            logger.debug(f"Installed {len(installed)} Claude Code skill(s)")
        if updated:
            logger.debug(f"Updated {len(updated)} Claude Code skill(s)")
    except Exception as e:
        logger.warning(f"Failed to install Claude Code skills: {e}")


def install_skills() -> None:
    """Install semantic-search skills to ~/.claude/skills/."""
    try:
        installed, updated = _install_skills_core()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    target_dir = get_skills_target_dir()

    # Report results
    if installed:
        print(f"Installed {len(installed)} skill(s):")
        for name in installed:
            print(f"  /{name}")

    if updated:
        print(f"Updated {len(updated)} skill(s):")
        for name in updated:
            print(f"  /{name}")

    print(f"\nSkills installed to: {target_dir}")
    print("Restart Claude Code to use them.")


def main() -> None:
    """Entry point for install-skills command."""
    install_skills()


if __name__ == "__main__":
    main()
