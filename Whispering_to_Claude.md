# Git Subtree Update Script (us) Installation Guide

This script automates the process of updating a git subtree by removing the existing subtree, adding a new one from a specified branch/commit, and keeping only the `src` directory.

## Prerequisites

- Git installed and configured
- Shell environment (bash/zsh)
- Write access to the target repository

## Installation Steps

### 1. Download the Script

Save the script as `us.sh` in your project directory or a temporary location:

```bash
#!/bin/bash

# Script to update git subtree with a specific branch or commit
# Usage: us <branch-name-or-commit-hash>

set -e  # Exit on any error

# Change to the project directory
PROJECT_DIR="/Users/so/Hacking/codel-text-active"
echo "📂 Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || {
    echo "❌ Error: Could not change to directory $PROJECT_DIR"
    exit 1
}

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <branch-name-or-commit-hash>"
    echo "Example: $0 main"
    echo "Example: $0 abc123def"
    exit 1
fi

BRANCH_OR_COMMIT="$1"

echo "🔄 Starting subtree update with: $BRANCH_OR_COMMIT"

# Step 1: Remove existing code subtree and prepare for clean pull
echo "📝 Step 1: Removing existing code subtree..."
git rm -rf code && \
git commit -m "Remove code subtree to prepare for clean pull" && \
git subtree add --prefix=code child-repo "$BRANCH_OR_COMMIT" --squash

# Step 2: Keep only the src directory and clean up
echo "📝 Step 2: Keeping only code/src directory..."
mkdir -p temp_backup && \
cp -r ./code/src temp_backup/ && \
rm -rf ./code/* && \
mkdir -p ./code && \
mv temp_backup/src ./code/ && \
rm -rf temp_backup && \
git add ./code && \
git commit -m "Keep only code/src directory" && \
git push origin main

echo "✅ Subtree update completed successfully!"
echo "📁 Only code/src directory has been retained from: $BRANCH_OR_COMMIT"
```

### 2. Configure the Script for Your Environment

#### Update PROJECT_DIR

Edit the `PROJECT_DIR` variable on line 9 to point to your project directory:

```bash
PROJECT_DIR="/path/to/your/project"
```

**Examples:**
- `PROJECT_DIR="/Users/john/Projects/my-app"`
- `PROJECT_DIR="/home/user/code/my-project"`
- `PROJECT_DIR="/Users/so/Hacking/codel-text-active"`

### 3. Set Up Git Remote

Navigate to your project directory and add the child repository as a remote:

```bash
cd /path/to/your/project
git remote add child-repo https://github.com/username/repository-name.git
```

**Example:**
```bash
cd /Users/so/Hacking/codel-text-active
git remote add child-repo https://github.com/srosro/ltmm-cleanup-v2.git
```

**Verify the remote was added:**
```bash
git remote -v
```

You should see output similar to:
```
child-repo    https://github.com/username/repository-name.git (fetch)
child-repo    https://github.com/username/repository-name.git (push)
origin        https://github.com/username/main-repo.git (fetch)
origin        https://github.com/username/main-repo.git (push)
```

### 4. Install the Script Globally

#### Option A: User bin directory (Recommended)

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

#### Option B: System-wide installation

```bash
sudo cp us.sh /usr/local/bin/us
sudo chmod +x /usr/local/bin/us
```

## Usage

Once installed, you can run the script from anywhere:

```bash
us main
us feature-branch
us abc123def456
```

## Verification

Test that everything is working:

1. Check the script is in your PATH:
   ```bash
   which us
   ```

2. Verify git remote configuration:
   ```bash
   cd /path/to/your/project
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
- Ensure you have access to the child repository

### Permission errors
- Ensure you have write access to both repositories
- Check that you're not in a detached HEAD state

## Customization

You can modify the script to:
- Change the target directory (currently `code`)
- Keep different subdirectories (currently `src`)
- Use different commit messages
- Add additional validation or logging