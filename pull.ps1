# Get the directory of the script itself
$projectDir = $PSScriptRoot

# Navigate to the project directory
cd $projectDir

# Fetch the latest changes
git fetch origin

# Pull the latest changes from the main branch
git pull origin main
