# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _probe_import(package_parent: Path) -> dict[str, object]:
    source = """
import json
from pathlib import Path
import sys

package_parent = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(package_parent))
path_registry = sys.path
path_snapshot = tuple(sys.path)

import nemo_gym

observed = {
    "module_path": str(Path(nemo_gym.__file__).resolve(strict=True)),
    "path_identity_preserved": sys.path is path_registry,
    "path_tuple_preserved": tuple(sys.path) == path_snapshot,
    "package_parent_count": sys.path.count(str(package_parent)),
}
sys.stdout.write(json.dumps(observed, sort_keys=True) + "\\n")
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-E",
            "-c",
            source,
            str(package_parent),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def test_import_nemo_gym_preserves_existing_sys_path_registry() -> None:
    assert _probe_import(REPO_ROOT) == {
        "module_path": str((REPO_ROOT / "nemo_gym/__init__.py").resolve()),
        "path_identity_preserved": True,
        "path_tuple_preserved": True,
        "package_parent_count": 1,
    }


def test_installed_layout_import_preserves_existing_sys_path_registry(
    tmp_path: Path,
) -> None:
    package_parent = tmp_path / "site-packages"
    installed_package = package_parent / "nemo_gym"
    installed_package.mkdir(parents=True)
    for name in ("__init__.py", "package_info.py"):
        shutil.copyfile(REPO_ROOT / "nemo_gym" / name, installed_package / name)

    assert _probe_import(package_parent) == {
        "module_path": str((installed_package / "__init__.py").resolve()),
        "path_identity_preserved": True,
        "path_tuple_preserved": True,
        "package_parent_count": 1,
    }
