setlocal enabledelayedexpansion
for /d %%v in (./voxel_res/*) do (
move .\voxel_res\%%v\model.nrrd .\testdata\%%v\
)
ENDLOCAL
