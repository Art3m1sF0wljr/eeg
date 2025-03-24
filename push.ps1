# Get the directory of the script itself
$projectDir = $PSScriptRoot

# Navigate to the project directory
cd $projectDir

# Get commit message from the user
$commitMessage = Read-Host "Enter a commit message"

# Stage all changes
git add .

# Commit the changes
git commit -m $commitMessage

# Push to the main branch
git push origin main
