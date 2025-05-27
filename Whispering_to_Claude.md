# Git Subtree Update Script (us) Installation Guide

This script automates the process of updating a git subtree by removing the existing subtree, adding a new one from a specified branch/commit, and keeping only a specified subdirectory (defaults to `src`).

## Prerequisites

- Git installed and configured
- Shell environment (bash/zsh)
- Write access to the target repository

## Installation Steps

### 0. Create a NEW github repo
This will be a mirror of the codebase (or a subfolder of the codebase) that you want to focus Claude on.

https://github.com/new

### 1. Download the Script

Save the script as `us.sh` in your project directory or a temporary location:

```bash
#!/bin/bash

# User Configuration
PROJECT_DIR="/Users/delattre/tmp/codel-text"  # Your project directory (where you want to overwrite main)
DEFAULT_SUBDIRECTORY="src"                    # Default subdirectory to keep (if not specified)
SOURCE_REPO_NAME="source-repo"                # Name of the git remote for the source repo (where we pull FROM)
MAIN_BRANCH="main"                            # Your main branch name

# Script to update git subtree with a specific branch or commit and keep only a specified subdirectory
# Usage: us <branch-name-or-commit-hash> [subdirectory]

set -e  # Exit on any error

echo "📂 Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || {
    echo "❌ Error: Could not change to directory $PROJECT_DIR"
    exit 1
}

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <branch-name-or-commit-hash> [subdirectory]"
    echo "Example: $0 main"
    echo "Example: $0 main src"
    echo "Example: $0 dev tests"
    echo "Example: $0 abc123def docs"
    exit 1
fi

BRANCH_OR_COMMIT="$1"
SUBDIRECTORY="${2:-$DEFAULT_SUBDIRECTORY}"  # Use configured default if no subdirectory specified

echo "🔄 Starting subtree update with: $BRANCH_OR_COMMIT"
echo "📁 Will keep only: code/$SUBDIRECTORY"

# Step 1: Remove existing code subtree if it exists and prepare for clean pull
echo "📝 Step 1: Preparing for clean pull..."
if [ -d "code" ]; then
    git rm -rf code
    git commit -m "Remove code subtree to prepare for clean pull"
fi

# Add the subtree
git subtree add --prefix=code $SOURCE_REPO_NAME "$BRANCH_OR_COMMIT" --squash

# Step 2: Keep only the specified subdirectory and clean up
echo "📝 Step 2: Keeping only code/$SUBDIRECTORY directory..."

# Check if the specified subdirectory exists
if [ ! -d "./code/$SUBDIRECTORY" ]; then
    echo "❌ Error: Directory './code/$SUBDIRECTORY' does not exist in the subtree"
    echo "Available directories in ./code/:"
    ls -la ./code/
    exit 1
fi

mkdir -p temp_backup && \
cp -r "./code/$SUBDIRECTORY" temp_backup/ && \
rm -rf ./code/* && \
mkdir -p ./code && \
mv "temp_backup/$SUBDIRECTORY" ./code/ && \
rm -rf temp_backup && \
git add ./code && \
git commit -m "Keep only code/$SUBDIRECTORY directory" && \
git push --force origin $MAIN_BRANCH

echo "✅ Subtree update completed successfully!"
echo "📁 Only code/$SUBDIRECTORY directory has been retained from: $BRANCH_OR_COMMIT" 
```

### 2. Configure the Script for Your Environment

Edit the configuration variables at the top of the script:

```bash
# User Configuration
PROJECT_DIR="/path/to/your/new/gh-project"  # Your project directory (where you want to overwrite main)
DEFAULT_SUBDIRECTORY="src"                  # Default subdirectory to keep (if not specified)
SOURCE_REPO_NAME="source-repo"              # Name of the git remote for the source repo (where we pull FROM)
MAIN_BRANCH="main"                          # Your main branch name
```

Also update the default subdirectory you want to keep.  In our case it defaults to `src`, which is our codebase. I don't want Claude looking at our whole repo because it blows out the context window.

**Examples:**
- `PROJECT_DIR="/Users/john/Projects/my-app"`
- `PROJECT_DIR="/home/user/code/my-project"`
- `PROJECT_DIR="/Users/so/Hacking/codel-text-active"`

### 3. Set Up Git Remote

Navigate to your NEW project directory and add the source repository (the original codebase you're pulling from) as a remote:

```bash
cd /path/to/your/new/gh-project
git remote add source-repo https://github.com/username/repository-name.git
```

**Example:**
```bash
cd /Users/so/Hacking/codel-text-active
git remote add source-repo https://github.com/srosro/ltmm-cleanup-v2.git
```

**Verify the remote was added:**
```bash
git remote -v
```

You should see output similar to:
```
source-repo   https://github.com/username/repository-name.git (fetch)
source-repo   https://github.com/username/repository-name.git (push)
origin        https://github.com/username/your-new-repo.git (fetch)
origin        https://github.com/username/your-new-repo.git (push)
```

### 4. Install the Script Globally

1. Create a personal bin directory:
   ```bash
   mkdir -p ~/bin
   ```

2. Copy and rename the script:
   ```bash
   cp us.sh ~/bin/us
   chmod +x ~/bin/us
   ```

3. Add to your PATH (add to `~/.zshrc` or `~/.bash_profile`):
   ```bash
   echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
   ```

4. Reload your shell:
   ```bash
   source ~/.zshrc
   ```

## Usage

Once installed, you can run the script from anywhere:

```bash
# Keep default 'src' subdirectory
us main
us feature-branch
us abc123def456

# Keep a specific subdirectory
us main src
us main tests
us dev docs
us feature-branch components
us abc123def frontend
```

## Verification

Test that everything is working:

1. Check the script is in your PATH:
   ```bash
   which us
   ```

2. Verify git remote configuration:
   ```bash
   cd /path/to/your/new/gh-project
   git remote -v
   ```

3. Run a test (make sure you're ready to update your subtree):
   ```bash
   us main
   ```

## Troubleshooting

### Script not found
- Ensure the script is executable: `chmod +x ~/bin/us`
- Verify PATH includes `~/bin`: `echo $PATH`
- Restart your terminal or run `source ~/.zshrc`

### Authentication errors
- Set up SSH keys for GitHub authentication
- Or use a personal access token instead of password

### Git remote errors
- Verify the remote exists: `git remote -v`
- Check the repository URL is correct
- Ensure you have access to the source repository

### Permission errors
- Ensure you have write access to both repositories
- Check that you're not in a detached HEAD state

## Claude Integration

Now you should be able to go to https://claude.ai/projects, create a new project, and sync your "targeted" GitHub repo containing only the specific subdirectory you want Claude to focus on.