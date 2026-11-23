"""Build an isolated prototype using an existing CT2 build and oneDNN sources."""
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--build", type=Path, default=Path(".cache/cpu-plan/branch-build"))
parser.add_argument("--onednn-source", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("/tmp/ct2-attn-variants/planned"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
build = args.build.resolve()
output = args.output.resolve()
if 'set(PROJECT_VERSION "3.1.1")' not in (args.onednn_source / "CMakeLists.txt").read_text():
    parser.error("The prototype requires the matching oneDNN 3.1.1 source tree")
output.mkdir(parents=True, exist_ok=True)
flags = {}
for line in (build / "CMakeFiles/ctranslate2.dir/flags.make").read_text().splitlines():
    if " = " in line:
        key, value = line.split(" = ", 1)
        flags[key] = shlex.split(value)
common = flags["CXX_DEFINES"] + flags["CXX_INCLUDES"] + flags["CXX_FLAGS"]
common += ["-I" + str(args.onednn_source.resolve() / "src"),
           "-I" + str(root / "tools/onednn_attention_plan")]
source = (root / "src/ops/fused_attention.cc").read_text()
source = '#include "plan.h"\n' + source
needle = "      dim_t block_queries;"
assert source.count(needle) == 1
source = source.replace(needle, """      if (use_dnnl && !values_lengths
          && experimental::planned_attention(queries, keys, values, queries_scale, output))
        return true;
""" + needle)
(output / "fused_attention.cc").write_text(source)
for name, path in [("plan", root / "tools/onednn_attention_plan/plan.cc"),
                   ("fused_attention", output / "fused_attention.cc")]:
    subprocess.run(["c++", *common, "-c", str(path), "-o", str(output / (name + ".o"))], check=True)
link = shlex.split((build / "CMakeFiles/ctranslate2.dir/link.txt").read_text())
link[link.index("-o") + 1] = str(output / "libctranslate2.so.4.9.0")
idx = link.index("CMakeFiles/ctranslate2.dir/src/ops/fused_attention.cc.o")
link[idx:idx + 1] = [str(output / "fused_attention.o"), str(output / "plan.o")]
subprocess.run(link, cwd=build, check=True)
for name in ("libctranslate2.so.4", "libctranslate2.so"):
    path = output / name
    if path.is_symlink():
        path.unlink()
    path.symlink_to("libctranslate2.so.4.9.0")
print(output)
