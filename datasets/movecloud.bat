setlocal enabledelayedexpansion
for /d %%v in (./cloud_data/cloud/*) do (
move .\cloud_data\cloud\%%v .\traindata\
)
ENDLOCAL
