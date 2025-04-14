import sys
import os

static_patches = [
    (
        "Cargo.toml",
        [
            ('name = "mwpf"', 'name = "mwpf_$name$"', 3),
            ('default-run = "mwpf"', 'default-run = "mwpf_$name$"', 1),
        ],
    ),
    (
        "src/lib.rs",
        [
            ("fn mwpf(", "fn mwpf_$name$(", 1),
        ],
    ),
    (
        "src/main.rs",
        [
            ("use mwpf::cli::*;", "use mwpf_$name$::cli::*;", 1),
        ],
    ),
    (
        "src/python/mwpf/sinter_decoders.py",
        [
            ("import mwpf\n", "import mwpf_$name$\n", 1),
            ("from mwpf import", "from mwpf_$name$ import", 1),
            ("getattr(mwpf, decoder_type)", "getattr(mwpf_$name$, decoder_type)", 1),
            ("SinterMWPFDecoder", "SinterMWPF$Name$Decoder", 4),
            ("SinterHUFDecoder", "SinterHUF$Name$Decoder", 1),
            ("SinterSingleHairDecoder", "SinterSingleHair$Name$Decoder", 1),
            ("MwpfCompiledDecoder", "Mwpf$Name$CompiledDecoder", 3),
        ],
    ),
    (
        "src/python/mwpf/ref_circuit.py",
        [
            ("import mwpf\n", "import mwpf_$name$\n", 1),
            ("mwpf.", "mwpf_$name$.", None),
        ],
    ),
    (
        "src/python/mwpf/heralded_dem.py",
        [
            ("import mwpf\n", "import mwpf_$name$\n", 1),
            ("mwpf.", "mwpf_$name$.", None),
        ],
    ),
    (
        "src/python/mwpf/__init__.py",
        [
            ("from .mwpf import *", "from .mwpf_$name$ import *", 1),
            ("mwpf.", "mwpf_$name$.", 2),
            ("mwpf,", "mwpf_$name$,", 1),
        ],
    ),
    (
        "tests/python/test_sinter.py",
        [
            ("SinterMWPFDecoder", "SinterMWPF$Name$Decoder", None),
            ("SinterHUFDecoder", "SinterHUF$Name$Decoder", None),
        ],
    ),
    (
        "README.md",
        [
            ("pip install -U mwpf\n", "pip install -U mwpf_$name$\n", 1),
            (
                "pip install -U 'mwpf[stim]'\n",
                "pip install -U 'mwpf_$name$[stim]'\n",
                1,
            ),
            ('decoders = ["mwpf"],', 'decoders = ["mwpf_$name$"],', 1),
            (
                '"mwpf": SinterMWPFDecoder',
                '"mwpf_$name$": SinterMWPF$Name$Decoder',
                2,
            ),
            ("import SinterMWPFDecoder", "import SinterMWPF$Name$Decoder", 1),
            ("from mwpf import ", "from mwpf_$name$ import ", 2),
        ],
    ),
]

####### module name patches #######
pyclass_patch_files = [
    "src/dual_module.rs",
    "src/example_codes.rs",
    "src/html_export.rs",
    "src/mwpf_solver.rs",
    "src/util_py.rs",
    "src/util.rs",
    "src/visualize.rs",
    "src/matrix/interface.rs",
    "src/matrix/row.rs",
]
for filename in pyclass_patch_files:
    static_patches.append(
        (
            filename,
            [('pyclass(module = "mwpf"', 'pyclass(module = "mwpf_$name$"', None)],
        ),
    )


def patches_of(name: str) -> list:
    def patch(new: str):
        return (
            new.replace("$name$", name.lower())
            .replace("$Name$", name.capitalize())
            .replace("$NAME$", name.upper())
        )

    patches = [
        (name, [(old, patch(new), count) for old, new, count in lst])
        for name, lst in static_patches
    ]
    if name == "rational":
        patches.append(
            (
                "pyproject.toml",
                [
                    ('name = "mwpf"', 'name = "mwpf_rational"', 1),
                    ("f64_weight", "rational_weight", 2),
                ],
            )
        )
    elif name == "incr":
        patches.append(
            (
                "pyproject.toml",
                [
                    ('name = "mwpf"', 'name = "mwpf_incr"', 1),
                    (
                        "python_binding,f64_weight,embed_visualizer",
                        "python_binding,f64_weight,incr_lp,embed_visualizer",
                        1,
                    ),
                    ('"f64_weight",', '"f64_weight" ,"incr_lp",', 1),
                ],
            )
        )
    elif name == "fast":
        patches.append(
            (
                "pyproject.toml",
                [
                    ('name = "mwpf"', 'name = "mwpf_fast"', 1),
                    (
                        "python_binding,f64_weight,embed_visualizer",
                        "python_binding,f64_weight,fast_ds,embed_visualizer",
                        1,
                    ),
                    ('"f64_weight",', '"f64_weight" ,"fast_ds",', 1),
                ],
            )
        )
    return patches


# patch is strict
def patch(dry: bool, name: str):
    for filename, replacements in patches_of(name):
        with open(filename, "r") as f:
            content = f.read()
        # check occurrences first
        for old, new, occurrence in replacements:
            assert (
                occurrence is None or content.count(old) == occurrence
            ), f"count {filename} for '{old}': {content.count(old)} != {occurrence}"
            assert (
                content.count(new) == 0
            ), f"count {filename} for '{new}': {content.count(new)} != 0"
        # during application of the replacements, also check occurrence
        for old, new, occurrence in replacements:
            assert occurrence is None or content.count(old) == occurrence
            assert content.count(new) == 0
            old_content = content
            content = content.replace(old, new)
            assert (
                content != old_content
            ), f"Patch failed for {filename}: {old} -> {new}"
        # check occurrences last
        for old, new, occurrence in replacements:
            assert occurrence is None or content.count(new) == occurrence
            assert content.count(old) == 0
        if not dry:
            with open(filename, "w") as f:
                f.write(content)
    if not dry:
        # up to here, all files has been checked and updated, rename the src/python/mwpf folder
        os.rename("src/python/mwpf", f"src/python/mwpf_{name}")


# revert is best-practice
def revert(name: str):
    # first change the folder back
    os.rename(f"src/python/mwpf_{name}", "src/python/mwpf")
    for filename, replacements in patches_of(name):
        with open(filename, "r") as f:
            content = f.read()
        for old, new, occurrence in replacements:
            count = content.count(new)
            if occurrence is not None and count != occurrence:
                print(
                    f"[warning] reverting process counting error '{old}' '{new}' {occurrence} != {count}"
                )
            content = content.replace(new, old)
        with open(filename, "w") as f:
            f.write(content)


if __name__ == "__main__":
    assert (
        len(sys.argv) == 3
    ), "Usage: python pyproject-patch.py [rational|incr|fast] [dry|apply|revert]"
    assert sys.argv[1] in ["rational", "incr", "fast"]
    if sys.argv[2] == "dry":
        patch(dry=True, name=sys.argv[1])
    elif sys.argv[2] == "apply":
        patch(dry=True, name=sys.argv[1])
        patch(dry=False, name=sys.argv[1])
    elif sys.argv[2] == "revert":
        revert(name=sys.argv[1])
    else:
        raise ValueError("Invalid argument, should be dry|apply|revert")
