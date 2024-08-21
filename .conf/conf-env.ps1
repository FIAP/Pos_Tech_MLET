# PowerShell script

# Function to execute a command and print its output
function ExecuteCommand($command) {
    Write-Host "Executing: $command"
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error executing command: $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# Configure the package to use local .venv
$command = "poetry config virtualenvs.in-project true"
ExecuteCommand $command

# Select Python 3.10 as the package version
$command = "poetry env use python3.11"
ExecuteCommand $command

# Select Python 3.10 as the package version
$command = "poetry install"
ExecuteCommand $command