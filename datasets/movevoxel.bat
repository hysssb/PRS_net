setlocal enabledelayedexpansion
for /d %%v in (./voxel_data/voxel/datarst/*) do (
move .\voxel_data\voxel\datarst\%%v\model.nrrd .\traindata\%%v\
)
ENDLOCAL
