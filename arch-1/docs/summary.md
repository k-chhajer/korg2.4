# Research Note

The repo now separates three things cleanly:

- implementation of Architecture 1 contracts
- evaluation tooling and benchmark assets
- paper-facing documentation

Most important change:

- Architecture 1 is now represented as a post-trained committee interface with explicit blocker checks when only a prompt scaffold is available

That means the code can support a paper-valid committee, while the docs prevent accidental overclaiming from local scaffold runs.
