# Block-path audit (mandatory)

Label `sNbM` uses **stage_idx N** (0-indexed into [stage1..stage4]); the Flax module is `stage{N+1}`. So `s3b0` = stage_idx 3 = `stage4[0]`. The manuscript's selected block **(3,0) = s3b0**; the stale code default was **s3b1**. Both are in the grid; the report states which minimizes mean val NLL.

| label | stage_idx | block_idx | Flax conv1 | conv2 | in→out ch | downsample |
|---|---|---|---|---|---|---|
| s1b0 | 1 | 0 | stage2.0.conv1.kernel | stage2.0.conv2.kernel | 64→128 | True |
| s1b1 | 1 | 1 | stage2.1.conv1.kernel | stage2.1.conv2.kernel | 128→128 | False |
| s2b0 | 2 | 0 | stage3.0.conv1.kernel | stage3.0.conv2.kernel | 128→256 | True |
| s2b1 | 2 | 1 | stage3.1.conv1.kernel | stage3.1.conv2.kernel | 256→256 | False |
| s3b0 | 3 | 0 | stage4.0.conv1.kernel | stage4.0.conv2.kernel | 256→512 | True |
| s3b1 | 3 | 1 | stage4.1.conv1.kernel | stage4.1.conv2.kernel | 512→512 | False |
