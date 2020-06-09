setlocal enabledelayedexpansion
set /a count = 0
for /d %%v in (./ShapeNetCore.v1/04554684/*) do (
set /a count += 1
md .\voxel\datarst\!count!
.\binvox.exe -d 32 -t nrrd .\ShapeNetCore.v1\04554684\%%v\model.obj 
move .\ShapeNetCore.v1\04554684\%%v\model.nrrd .\voxel\datarst\!count!
)
ENDLOCAL
