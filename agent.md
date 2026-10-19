# Build instructions

Use `./build_amd.sh` from the CTranslate2 repository root to build and install
CTranslate2 for AMD/HIP:

```bash
./build_amd.sh
```

This script uses the production Linux ROCm CI build configuration by default.
Run `./build_amd.sh --help` for dependency requirements and build options.

Use `./build_all.sh` when you also need to build Python wheels and update the
consumer projects; it calls `build_amd.sh` for the native build.

# Code changes

Always use as little code as possible to solve the task correctly. Prefer
simple changes and reuse existing code; avoid unnecessary abstractions,
dependencies, and unrelated changes.

# Code style

Follow the existing style in nearby repository code. Use
`src/models/whisper.cc` and `include/ctranslate2/models/whisper.h` as C++
references. Apply the same conventions to C++, CUDA, headers, bindings, and tests.

- Use two spaces for indentation, including each namespace level. Put nested
  namespaces on separate lines, with blank lines around their contents.
- Put opening braces on the same line as functions, classes, and control
  statements. For constructors with multiline initializer lists, follow nearby
  constructors: one initializer per line, leading commas, and the body brace
  on its own line.
- Expand function bodies and control statements over multiple lines. Omit braces
  for simple single-statement control bodies when consistent with nearby code.
- Use spaces around operators and after commas. Attach pointer and reference
  symbols to the type, for example `const Model& model` and `StorageView* output`.
- Separate functions and logical steps with blank lines. Wrap long expressions
  and parameter lists readably, aligning continuation arguments with nearby code.
  Existing code is the reference; do not impose a strict new line-length limit.
- In headers, use `#pragma once`, followed by a blank line before includes.
  Align `public:`, `protected:`, and `private:` with the class declaration and
  indent members two spaces further. Expand short inline functions over multiple
  lines as in `whisper.h`.
- Follow existing naming: `snake_case` for functions and variables, `PascalCase`
  for classes, and a leading underscore for private data members.
- Keep includes grouped like nearby files: the implementation's own header,
  standard-library headers, external headers, then project/internal headers,
  separated by blank lines as applicable.
- For Python, use the Black and isort versions pinned in `.github/workflows/ci.yml`
  and the configuration in `python/setup.cfg`.
- There is no project-wide C++ formatter configuration. Do not assume an
  unmodified LLVM or Google preset matches the repository. Review formatter
  output against existing code.
- Keep formatting edits within the task's scope. When changing existing files,
  avoid reformatting unrelated code. Formatting-only edits must preserve behavior.
