# Third-party notices

## Emmy

This project depends on `emmy-ml==0.3.11`, licensed under Apache-2.0.

- Source: https://github.com/cloudrift-ai/emmy
- Package: https://pypi.org/project/emmy-ml/0.3.11/
- License: https://www.apache.org/licenses/LICENSE-2.0

Emmy is installed from PyPI and is not vendored in this repository. It is used only to
deserialize the Kernel IR (`emmy.compiler.graph.Graph`). The IR to PTX lowering is this
project's own code.

## Emmy Hackathon 2026 task materials

The packages under `tests/variants/` were built for this submission using the task repository
and its IR, `meta.json` and `expected.npz` conventions:

- Task repository: https://github.com/cloudrift-ai/emmy-hackathon-2026

The task repository does not carry a license file. The files under `tests/variants/` are
therefore **not** licensed under this repository's MIT grant; they are included only so that
reviewers can reproduce the submission's regression results. All rights in the task materials
remain with CloudRift and the organizers.

If you reuse this repository outside the hackathon, replace the variant packages with cases you
are entitled to redistribute.

## Tool-only dependencies

`numpy`, `requests` and `ruff` are used for local verification and benchmarking. They are not
required by `compile.py` and are not part of the compile path.

- numpy: BSD-3-Clause, https://numpy.org/
- requests: Apache-2.0, https://requests.readthedocs.io/
- ruff: MIT, https://github.com/astral-sh/ruff
