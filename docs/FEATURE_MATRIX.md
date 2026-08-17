# v1.0 feature matrix

| World kind | Generate | Simulate | Visualize | Stable inversion | Score |
|---|---:|---:|---:|---:|---:|
| circle | yes | yes | yes | joint and staged grid search | yes |
| rectangle | yes | yes | yes | no | limited |
| layered | yes | yes | yes | no | limited |
| blobs | yes | yes | yes | no | limited |
| ellipse | yes | yes | yes | known-shape center search | yes |
| ring | yes | yes | yes | no | mask visualization only |
| two-circles | yes | yes | yes | no | limited |
| crack | yes | yes | yes | no | limited |
| circle-layered | yes | yes | yes | no dedicated baseline | limited |
| mask-blocks | yes | yes | yes | greedy coarse cell search | yes |

## Cross-cutting features

| Capability | Status |
|---|---|
| Devito CPU forward model | stable toy baseline |
| Simultaneous shots | supported |
| Sequential shots | supported |
| Simple sponge | supported, not PML |
| Noise and sensor perturbations | supported |
| Differential trace objective | supported |
| L2 and correlation metrics | supported |
| Candidate uncertainty summaries | supported, qualitative |
| Open challenges | supported |
| Blind challenges | supported |
| Challenge leaderboard | supported |
| Active sensing | supported with heuristic policies |
| Active leaderboard | supported |
| HTML experiment reports | supported |
| Artifact validation | supported |
| Standard release suite | supported |
| Master release audit | supported |

| Wheel/install release smoke | supported |
