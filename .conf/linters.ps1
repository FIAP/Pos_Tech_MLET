$rootDir = git rev-parse --show-toplevel
Get-ChildItem -Path $rootDir -Include *.py -Recurse | ForEach-Object {black --config pyproject.toml $_.FullName}
Get-ChildItem -Path $rootDir -Include *.py -Recurse | ForEach-Object {pylint --rcfile=pyproject.toml $_.FullName}