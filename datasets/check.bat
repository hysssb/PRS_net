setlocal enabledelayedexpansion
for /d %%v in (./traindata/*) do (
copy .\traindata\%%v\model.pcd .\check\%%v.pcd
)
ENDLOCAL