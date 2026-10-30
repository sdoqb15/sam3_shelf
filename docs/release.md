# Release preparation / 发布准备

This is a clean, standalone source tree. It is not a rewrite of the private
research repository's Git history. Full experiment logs and private annotations
remain in the separate internal archive.

Before public publication:

1. The owner selects a code license before granting open-source permissions. [License status](../LICENSE_PENDING.md) is currently pending.
2. Review upstream SAM3 model access/terms; model weights are obtained separately.
3. Run tests and `python scripts/check_release.py` from this source tree.
4. Publish this repository's clean history, not an internal Git bundle or `.git` directory.

The release checker rejects model/data archives, runtime products, private
machine paths, non-loopback numeric addresses and common credential tokens.
It complements inspection and does not grant permission to release a dataset.

The owner has selected `https://github.com/sdoqb15/sam3_shelf` as the source
repository. The [associated HF checkpoints](pretrained_models.md) retain the SAM
License. The dataset is private and has separate access control. Publishing the
source does not select its pending code license. The internal intranet review
website and original checkpoints keep operating independently.
