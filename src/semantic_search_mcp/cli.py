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


def get_commands_source_dir() -> Path:
    """Get the path to the bundled commands directory."""
    return Path(__file__).parent / "commands"


def get_commands_target_dir() -> Path:
    """Get the user's Claude Code commands directory."""
    return Path.home() / ".claude" / "commands"


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


def _install_commands_core() -> tuple[list[str], list[str]]:
    """Core command installation logic.

    Installs commands to ~/.claude/commands/ for global availability.
    Commands are named semantic-search-*.md for /semantic-search-* syntax.

    Returns:
        Tuple of (installed_commands, updated_commands)

    Raises:
        FileNotFoundError: If commands source directory doesn't exist
        ValueError: If no commands found in package
    """
    source_dir = get_commands_source_dir()
    target_dir = get_commands_target_dir()

    if not source_dir.exists():
        raise FileNotFoundError(f"Commands source directory not found: {source_dir}")

    # Find all command files (*.md)
    command_files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix == ".md"]

    if not command_files:
        raise ValueError("No commands found in package")

    # Create target directory if needed
    target_dir.mkdir(parents=True, exist_ok=True)

    installed = []
    updated = []

    for cmd_file in command_files:
        cmd_name = cmd_file.stem  # filename without .md
        target_file = target_dir / cmd_file.name

        if target_file.exists():
            shutil.copy2(cmd_file, target_file)
            updated.append(cmd_name)
        else:
            shutil.copy2(cmd_file, target_file)
            installed.append(cmd_name)

    return installed, updated


def install_skills_silent() -> None:
    """Install skills and commands silently. Used during server startup.

    Logs errors but doesn't print to stdout.
    """
    # Install skills
    try:
        installed, updated = _install_skills_core()
        if installed:
            logger.debug(f"Installed {len(installed)} Claude Code skill(s)")
        if updated:
            logger.debug(f"Updated {len(updated)} Claude Code skill(s)")
    except Exception as e:
        logger.warning(f"Failed to install Claude Code skills: {e}")

    # Install commands
    try:
        installed, updated = _install_commands_core()
        if installed:
            logger.debug(f"Installed {len(installed)} Claude Code command(s)")
        if updated:
            logger.debug(f"Updated {len(updated)} Claude Code command(s)")
    except Exception as e:
        logger.warning(f"Failed to install Claude Code commands: {e}")


def install_skills() -> None:
    """Install semantic-search skills and commands.

    Skills are installed globally to ~/.claude/skills/ for AI auto-discovery.
    Commands are installed globally to ~/.claude/commands/ for user invocation.
    """
    skills_installed = []
    skills_updated = []

    # Install skills
    try:
        skills_installed, skills_updated = _install_skills_core()
    except FileNotFoundError as e:
        print(f"Error (skills): {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error (skills): {e}", file=sys.stderr)
        sys.exit(1)

    # Install commands
    try:
        cmds_installed, cmds_updated = _install_commands_core()
    except FileNotFoundError as e:
        print(f"Error (commands): {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error (commands): {e}", file=sys.stderr)
        sys.exit(1)

    # Report skill results
    if skills_installed:
        print(f"Installed {len(skills_installed)} skill(s) (AI auto-discovery):")
        for name in skills_installed:
            print(f"  {name}")

    if skills_updated:
        print(f"Updated {len(skills_updated)} skill(s) (AI auto-discovery):")
        for name in skills_updated:
            print(f"  {name}")

    # Report command results
    if cmds_installed:
        print(f"\nInstalled {len(cmds_installed)} command(s):")
        for name in cmds_installed:
            print(f"  /{name}")

    if cmds_updated:
        print(f"\nUpdated {len(cmds_updated)} command(s):")
        for name in cmds_updated:
            print(f"  /{name}")

    print(f"\nSkills installed to: {get_skills_target_dir()}")
    print(f"Commands installed to: {get_commands_target_dir()}")
    print("Restart Claude Code to use them.")


def main() -> None:
    """Entry point for install-skills command."""
    install_skills()


if __name__ == "__main__":
    main()
