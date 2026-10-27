# -*- coding: utf-8 -*-
# Copyright©2020-2026 Shanghai Biren Technology Co., Ltd. All rights reserved.

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
COVERAGE_SUPPORT_DIR = TESTS_ROOT / "_coverage_subproc"

# Paths start at the repository's ``tests`` directory. A directory entry covers
# every test below it and is used only when all tests in that directory are multi-card.
MULTI_CARD_CASES = {
    2: {
        "tests/unit_tests/distributed/test_dp_overlap_grad_reduce.py",
        "tests/unit_tests/fusions/test_masked_softmax_fusion.py",
        "tests/unit_tests/pipeline_parallel/test_schedules_overlap.py",
        "tests/unit_tests/pipeline_parallel/test_zb_runtime.py",
        "tests/unit_tests/tensor_parallel",
        "tests/unit_tests/test_hierarchical_context_parallel.py",
        "tests/unit_tests/transformer/moe/test_moe_parallel_folding.py",
        "tests/unit_tests/transformer/test_cp_comm_type.py",
        "tests/upstream_unit_tests/dist_checkpointing/test_async_save.py",
        "tests/upstream_unit_tests/dist_checkpointing/test_fully_parallel.py",
        "tests/upstream_unit_tests/dist_checkpointing/test_global_metadata_reuse.py",
        "tests/upstream_unit_tests/dist_checkpointing/test_msc.py",
        "tests/upstream_unit_tests/dist_checkpointing/test_nonpersistent.py",
        "tests/upstream_unit_tests/dist_checkpointing/test_optimizer.py",
        "tests/upstream_unit_tests/dist_checkpointing/test_torch_dist.py",
        "tests/upstream_unit_tests/tensor_parallel",
        "tests/upstream_unit_tests/transformer/moe/test_aux_loss.py",
        "tests/upstream_unit_tests/transformer/moe/test_moe_layer.py",
        "tests/upstream_unit_tests/transformer/moe/test_moe_layer_discrepancy.py",
        "tests/upstream_unit_tests/transformer/moe/test_routers.py",
        "tests/upstream_unit_tests/transformer/moe/test_sequential_mlp.py",
        "tests/upstream_unit_tests/transformer/moe/test_token_dispatcher.py",
        "tests/upstream_unit_tests/transformer/moe/test_upcycling.py",
    },
    # Enable these mappings when 4- or 8-card CI resources are available.
    # 4: set(),
    # 8: {
    #     "tests/upstream_unit_tests/dist_checkpointing/test_fp8.py",
    #     "tests/upstream_unit_tests/dist_checkpointing/test_serialization.py",
    # },
}


class TestCollection:
    def __init__(self):
        self.config = ""
        self.collected = []

    def pytest_collection_modifyitems(self, config, items):
        self.config = config.option.markexpr
        valid_items = []
        for item in items:
            casename = [str(item.path)]
            if casename not in self.collected:
                self.collected.append(casename)
            valid_items.append(item)

        items.clear()
        items.extend(valid_items)


class MyPlugin(TestCollection):
    def pytest_sessionfinish(self):
        print("\n Finish megatron-lm-supa pytest testing ")


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Run unit tests with torchrun.")
    parser.add_argument(
        "--data-file",
        type=Path,
        help="Coverage data file shared by torchrun subprocesses.",
    )
    return parser.parse_known_args(argv)


def _case_paths(pytest_args):
    """Use pytest's parser to separate test paths from pytest options."""
    from _pytest.config import get_config

    parsed_args = list(pytest_args)
    config = get_config()
    config.invocation_params = config.invocation_params.__class__(
        args=tuple(parsed_args),
        plugins=None,
        dir=Path.cwd(),
    )
    try:
        config.parse(parsed_args)
        return list(config.args)
    finally:
        config._ensure_unconfigure()


def _shared_pytest_args(pytest_args, case_paths):
    """Remove pytest's positional test selectors while preserving all options."""
    remaining_cases = Counter(case_paths)
    shared_args = []
    for arg in pytest_args:
        if remaining_cases[arg]:
            remaining_cases[arg] -= 1
        else:
            shared_args.append(arg)
    return shared_args


def _normalize_case_path(case):
    case_path = case.split("::", 1)[0]
    path = Path(case_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        relative_path = path.resolve(strict=False).relative_to(REPO_ROOT)
    except ValueError:
        return None
    return relative_path.as_posix()


def _case_card_count(case):
    case_path = _normalize_case_path(case)
    if case_path is None:
        return None

    matched_counts = []
    for card_count, configured_cases in MULTI_CARD_CASES.items():
        if any(
            case_path == configured_case or case_path.startswith(f"{configured_case}/")
            for configured_case in configured_cases
        ):
            matched_counts.append(card_count)

    if len(matched_counts) > 1:
        counts = ", ".join(map(str, sorted(matched_counts)))
        raise ValueError(f"Case matches different card counts ({counts}): {case_path}")
    return matched_counts[0] if matched_counts else None


def _has_configured_descendant(case):
    case_path = _normalize_case_path(case)
    if case_path is None:
        return False
    prefix = f"{case_path}/"
    return any(
        configured_case.startswith(prefix)
        for configured_cases in MULTI_CARD_CASES.values()
        for configured_case in configured_cases
    )


def _is_test_file(path):
    return path.is_file() and (path.name.startswith("test_") or path.name.endswith("_test.py"))


def _expand_case(case):
    """Split a directory at configured multi-card case boundaries."""
    case_path = Path(case.split("::", 1)[0])
    if not case_path.is_absolute():
        case_path = Path.cwd() / case_path
    case_path = case_path.resolve(strict=False)
    card_count = _case_card_count(case)
    if card_count is not None or not case_path.is_dir() or not _has_configured_descendant(case):
        return [(card_count, case)]

    expanded = []
    for child in sorted(case_path.iterdir()):
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        child_case = child.as_posix()
        child_count = _case_card_count(child_case)
        if child_count is not None:
            expanded.append((child_count, child_case))
        elif child.is_dir() and _has_configured_descendant(child_case):
            expanded.extend(_expand_case(child_case))
        elif child.is_dir():
            expanded.append((None, child_case))
        elif _is_test_file(child):
            expanded.append((None, child_case))
    return expanded or [(None, case)]


def _test_groups(pytest_args):
    case_paths = _case_paths(pytest_args)
    shared_args = _shared_pytest_args(pytest_args, case_paths)
    grouped_cases = defaultdict(list)
    for case in case_paths:
        for card_count, expanded_case in _expand_case(case):
            grouped_cases[card_count].append(expanded_case)
    if not grouped_cases:
        return shared_args, [(None, [])]
    ordered_counts = sorted(grouped_cases, key=lambda count: count or 0)
    return shared_args, [(card_count, grouped_cases[card_count]) for card_count in ordered_counts]


JUNIT_XML_OPTIONS = ("--junit-xml", "--junitxml")


def _junit_xml_arg(pytest_args):
    for index, arg in enumerate(pytest_args):
        for option in JUNIT_XML_OPTIONS:
            if arg == option:
                if index + 1 >= len(pytest_args):
                    raise ValueError(f"{option} requires a file path")
                return index + 1, None, Path(pytest_args[index + 1])
            if arg.startswith(f"{option}="):
                return index, option, Path(arg.split("=", 1)[1])
    return None


def _junit_xml_path(pytest_args):
    match = _junit_xml_arg(pytest_args)
    return match[2] if match else None


def _group_junit_args(pytest_args, output_path):
    rewritten = list(pytest_args)
    match = _junit_xml_arg(rewritten)
    if match:
        index, option, _ = match
        rewritten[index] = f"{option}={output_path}" if option else str(output_path)
    return rewritten


def _group_output_path(output_path, card_count):
    label = "single" if card_count is None else f"{card_count}card"
    suffix = output_path.suffix or ".xml"
    return output_path.with_name(f"{output_path.stem}.{label}{suffix}")


def _merge_junit_xml(output_path, group_paths):
    suites = []
    for group_path in group_paths:
        root = ET.parse(group_path).getroot()
        suites.extend(list(root) if root.tag == "testsuites" else [root])

    totals = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    total_time = 0.0
    for suite in suites:
        for name in totals:
            totals[name] += int(suite.attrib.get(name, 0))
        total_time += float(suite.attrib.get("time", 0))

    attributes = {name: str(value) for name, value in totals.items()}
    attributes["time"] = f"{total_time:g}"
    root = ET.Element("testsuites", attributes)
    root.extend(suites)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    for group_path in group_paths:
        group_path.unlink(missing_ok=True)


def _finalize_junit_xml(output_path, group_paths):
    if all(path.exists() for path in group_paths):
        _merge_junit_xml(output_path, group_paths)
        return
    missing = ", ".join(str(path) for path in group_paths if not path.exists())
    print(f"JUnit XML merge skipped; missing group reports: {missing}")


def _absolute_data_file(data_file):
    if not data_file:
        return None
    path = Path(data_file).expanduser()
    return str(path if path.is_absolute() else (Path.cwd() / path).resolve())


def _enable_subprocess_coverage(data_file=None):
    """Wire up coverage measurement for processes spawned by ``torchrun``."""
    coverage_requested = bool(data_file or os.environ.get("COVERAGE_RUN") or os.environ.get("COVERAGE_PROCESS_START"))
    if not coverage_requested:
        return

    os.environ.setdefault("COVERAGE_PROCESS_START", str(REPO_ROOT / ".coveragerc"))
    if data_file:
        os.environ["COVERAGE_FILE"] = _absolute_data_file(data_file)
    else:
        os.environ.setdefault("COVERAGE_FILE", str(REPO_ROOT / ".coverage"))

    COVERAGE_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    (COVERAGE_SUPPORT_DIR / "sitecustomize.py").write_text(
        "import os\n"
        "import sys\n"
        "if os.environ.get('COVERAGE_PROCESS_START'):\n"
        "    try:\n"
        "        import coverage\n"
        "        coverage.process_startup()\n"
        "    except Exception as exc:\n"
        "        print(f'coverage process startup failed: {exc}', file=sys.stderr)\n"
    )
    support_dir = str(COVERAGE_SUPPORT_DIR)
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = support_dir + (os.pathsep + pythonpath if pythonpath else "")


def _coverage_pytest_command(data_file, pytest_args):
    """Run pytest under coverage in a single process."""
    command = [sys.executable, "-m", "coverage", "run"]
    if data_file:
        command.extend(["--data-file", _absolute_data_file(data_file)])
    command.extend(["-m", "pytest", *pytest_args])
    return command


def _save_parent_coverage():
    """Write the parent shard before combining subprocess data."""
    if not os.environ.get("COVERAGE_RUN"):
        return
    try:
        import coverage

        active = coverage.Coverage.current()
        if active is not None:
            active.stop()
            active.save()
    except (ImportError, AttributeError):
        pass


def _combine_coverage_data(data_file=None):
    """Merge per-process coverage files into the requested data file."""
    if not (data_file or os.environ.get("COVERAGE_RUN")):
        return
    combine_env = dict(os.environ)
    combine_env.pop("COVERAGE_RUN", None)
    combine_env.pop("COVERAGE_PROCESS_START", None)
    pythonpath = [
        path
        for path in combine_env.get("PYTHONPATH", "").split(os.pathsep)
        if path and path != str(COVERAGE_SUPPORT_DIR)
    ]
    combine_env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    combine_file = _absolute_data_file(data_file) or combine_env.get("COVERAGE_FILE")
    command = [sys.executable, "-m", "coverage", "combine"]
    if combine_file:
        command[4:4] = ["--data-file", combine_file]
    try:
        ret = subprocess.call(command, cwd=str(REPO_ROOT), env=combine_env)
        if ret != 0:
            print(f"coverage combine exited with code {ret}")
    except Exception as exc:  # pragma: no cover - never block the test result
        print(f"coverage combine failed: {exc}")


def _pytest_command(card_count, data_file, pytest_args):
    if card_count is None:
        if data_file is not None and not os.environ.get("COVERAGE_RUN"):
            return _coverage_pytest_command(data_file, pytest_args)
        return [sys.executable, "-m", "pytest", *pytest_args]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={card_count}",
        "-m",
        "pytest",
        *pytest_args,
    ]


def main():
    args, pytest_args = _parse_args(sys.argv[1:])
    data_file = args.data_file
    shared_args, groups = _test_groups(pytest_args)
    junit_path = _junit_xml_path(shared_args)
    split_junit = junit_path is not None and len(groups) > 1
    group_junit_paths = []
    first_failure = 0

    if split_junit:
        junit_path.unlink(missing_ok=True)
    _enable_subprocess_coverage(data_file)
    run_env = os.environ.copy()
    for card_count, case_paths in groups:
        group_args = [*shared_args, *case_paths]
        if split_junit:
            group_path = _group_output_path(junit_path, card_count)
            group_path.unlink(missing_ok=True)
            group_junit_paths.append(group_path)
            group_args = _group_junit_args(group_args, group_path)

        cmd = _pytest_command(card_count, data_file, group_args)
        print("Run command:", " ".join(map(str, cmd)))
        ret = subprocess.call(cmd, cwd=str(Path.cwd()), env=run_env)
        if ret != 0 and first_failure == 0:
            first_failure = ret
        label = "single-card" if card_count is None else f"{card_count}-card"
        print(f" Exit {label} megatron-lm-supa pytest with code: {ret} ")

    if split_junit:
        _finalize_junit_xml(junit_path, group_junit_paths)

    _save_parent_coverage()
    _combine_coverage_data(data_file)
    sys.exit(first_failure)


if __name__ == "__main__":
    main()
