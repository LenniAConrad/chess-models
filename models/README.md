# Models

## `leela_112planes-30blocksx384-policyhead80-valuehead32-policy4672-wdl3.bin` (LC0J format)

- Size: 317 MiB
- SHA-256: `c4dd6b62acd3c86be3d6199a32d6119d9144f508f84c823f69881ae0bae41034`
- Origin: LCZero run1 network `#610153` (classical policy+value net, converted to LC0J format)
- Architecture: residual CNN / ResNet-style classical LCZero network with squeeze-and-excitation (SE) blocks and dual policy/value heads
- Training-server SHA (download key): `09bc5be154e7401cce28e7d8f44b59548ffa0a6197d408ea872368738618d128`
- Approx. size: 30 blocks, 384 filters (~83.0M parameters)
- Architecture diagram: [`../assets/lc0-610153-architecture.dot`](../assets/lc0-610153-architecture.dot)
- Notes: Converted from `weights_run1_610153.pb.gz` to `leela_112planes-30blocksx384-policyhead80-valuehead32-policy4672-wdl3.bin`.

## `leela_112planes-10blocksx128-policyhead80-valuehead32-policy4672-wdl3.bin` (LC0J format)

- Size: 15 MiB
- SHA-256: `b99bec1aba97e96bf03ac8e016578527b983b6653f1adf040452f86c6f3ef348`
- Origin: LCZero network `#744706` (small classical net)
- Architecture: small residual CNN / ResNet-style classical LCZero network with squeeze-and-excitation (SE) blocks and dual policy/value heads
- Training-server SHA (download key): `0df5ca5b7485a043b88bea4f30d5b802033423f2f8ff0ca99b95e0646dd9325c`
- Approx. size: 10 blocks, 128 filters (~3.7M parameters)
- Architecture diagram: [`../assets/lc0-744706-architecture.dot`](../assets/lc0-744706-architecture.dot)

## `puzzle-classifier_21planes-6blocksx64-head32-logit1.bin` (classifier format)

- Size: 1.8 MiB
- SHA-256: `3b2e926aed216d8af8f4921805d429d737690cfb27dfc450abce99f04da58485`
- Origin: local classifier training run `training/classifier/runs/20260415-141952-cnn`
- Architecture: 21 input planes, 6 residual blocks x 64 channels, 32-channel classifier head, 1 output logit
- Training split: 50,000 positions at 1:4 puzzle/non-puzzle ratio
- Held-out test accuracy: `0.9112`
- Held-out test ROC-AUC: `0.9467`
- Held-out test average precision: `0.8267`
- Documentation: [`puzzle-classifier.md`](puzzle-classifier.md)
