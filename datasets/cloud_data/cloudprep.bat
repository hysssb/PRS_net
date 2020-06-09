setlocal enabledelayedexpansion
set /a count = 0
for /d %%v in (./ShapeNetCore.v1/04554684/*) do (
set /a count += 1
md .\cloud\!count!
D:\Project1\x64\Release\Project1.exe .\ShapeNetCore.v1\04554684\%%v\model.obj  .\cloud\!count!\model.pcd -n_samples 1000 -no_vis_result
)
ENDLOCAL
