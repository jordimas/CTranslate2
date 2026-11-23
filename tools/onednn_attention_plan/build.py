"""Build an isolated prototype using an existing CT2 build and oneDNN sources."""
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--build", type=Path, default=Path(".cache/cpu-plan/branch-build"))
parser.add_argument("--onednn-source", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("/tmp/ct2-attn-variants/planned"))
parser.add_argument("--int8-pack", action="store_true", help="Include the diagnostic hybrid INT8 weight adapter")
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
sources = [("plan", root / "tools/onednn_attention_plan/plan.cc"),
           ("fused_attention", output / "fused_attention.cc")]
if args.int8_pack:
    backend = (root / "src/cpu/backend.cc").read_text()
    needle = "    bool pack_gemm_weights(ComputeType compute_type) {"
    assert backend.count(needle) == 1
    backend = '#include "int8_pack.h"\n' + backend.replace(needle, needle + '''
      if ((compute_type == ComputeType::INT8 || compute_type == ComputeType::INT8_FLOAT32)
          && get_gemm_backend(compute_type) == GemmBackend::DNNL
          && experimental::int8_pack_enabled()) return true;''')
    primitives = (root / "src/cpu/primitives.cc").read_text()
    needle = "                                             int8_t* dest) {"
    assert primitives.count(needle) == 1
    primitives = '#include "int8_pack.h"\n' + primitives.replace(needle, needle + '''
    if (gemm_s8_backend == cpu::GemmBackend::DNNL && experimental::int8_pack_enabled())
      return experimental::pack_int8_weights(b, transpose_b, k, n, 1.f, dest);''')
    needle = """    case cpu::GemmBackend::DNNL: {
      const char transa = transpose_a ? 'T' : 'N';"""
    assert primitives.count(needle) == 1
    primitives = primitives.replace(needle, """    case cpu::GemmBackend::DNNL: {
      if (a_is_packed) throw std::runtime_error("experimental adapter does not pack activations");
      if (b_is_packed) {
        experimental::packed_int8_gemm(transpose_a, transpose_b, m, n, k, alpha,
          a, lda, b, ldb, beta, c, ldc, a_shift_compensation);
        break;
      }
      const char transa = transpose_a ? 'T' : 'N';""")
    for name, text in (("backend", backend), ("primitives", primitives)):
        path = output / (name + ".cc")
        path.write_text(text)
        sources.append((name, path))
    model = (root / "src/models/model.cc").read_text()
    start = model.index("    void Model::process_linear_weights() {")
    end = model.index("    static DataType get_dtype_from_item_size", start)
    block = model[start:end]
    needle = '''      const auto variable_index = _variable_index;
      for (const auto& pair : variable_index) {
        const std::string& name = pair.first;'''
    assert block.count(needle) == 1
    block = block.replace(needle, '''      // Snapshot names, not shared_ptrs: release each original when replaced.
      std::vector<std::string> variable_names;
      variable_names.reserve(_variable_index.size());
      for (const auto& pair : _variable_index) variable_names.push_back(pair.first);
      for (const auto& name : variable_names) {''')
    block = block.replace("const StorageView& weight = *pair.second;",
                          "const StorageView& weight = get_variable(name);")
    needle = "if (pack_weights && is_packable(name) && weight.rank() == 2) {"
    assert block.count(needle) == 1
    block = block.replace(needle, '''if (pack_weights && is_packable(name) && weight.rank() == 2
            && (!experimental::int8_pack_enabled()
                || experimental::pack_int8_weight_name(name))) {''')
    model = '#include "int8_pack.h"\n' + model[:start] + block + model[end:]
    path = output / "model.cc"
    path.write_text(model)
    sources.append(("model", path))
    sources.append(("int8_pack", root / "tools/onednn_attention_plan/int8_pack.cc"))
for name, path in sources:
    subprocess.run(["c++", *common, "-I" + str(root / "src/cpu"), "-c", str(path),
                    "-o", str(output / (name + ".o"))], check=True)
link = shlex.split((build / "CMakeFiles/ctranslate2.dir/link.txt").read_text())
link[link.index("-o") + 1] = str(output / "libctranslate2.so.4.9.0")
idx = link.index("CMakeFiles/ctranslate2.dir/src/ops/fused_attention.cc.o")
link[idx:idx + 1] = [str(output / "fused_attention.o"), str(output / "plan.o")]
if args.int8_pack:
    for name in ("backend", "primitives"):
        idx = link.index(f"CMakeFiles/ctranslate2.dir/src/cpu/{name}.cc.o")
        link[idx] = str(output / (name + ".o"))
    link.insert(link.index(str(output / "plan.o")) + 1, str(output / "int8_pack.o"))
    idx = link.index("CMakeFiles/ctranslate2.dir/src/models/model.cc.o")
    link[idx] = str(output / "model.o")
subprocess.run(link, cwd=build, check=True)
for name in ("libctranslate2.so.4", "libctranslate2.so"):
    path = output / name
    if path.is_symlink():
        path.unlink()
    path.symlink_to("libctranslate2.so.4.9.0")
print(output)
