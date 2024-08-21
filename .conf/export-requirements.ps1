# Navigate to the project directory
Set-Location -Path (git rev-parse --show-toplevel)

# Initialize the poetry shell
poetry shell

# Export the requirements to a requirements.txt file
poetry export -f requirements.txt --output requirements.txt --without-hashes
