import abc
import argparse
import json
import os
import shutil
import tempfile

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from ctranslate2.specs.model_spec import ACCEPTED_MODEL_TYPES, ModelSpec


def publish_directory(staging, destination):
    """Atomically publish a staged directory on the same filesystem."""
    os.rename(staging, destination)


@contextmanager
def _output_directory(output_dir, force=False, atomic=False):
    """Stage atomic conversions; keep the existing flow for other converters."""
    output = Path(output_dir)
    staging = (
        Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
        if atomic
        else output
    )
    try:
        if not atomic:
            if output.exists():
                shutil.rmtree(output)
            output.mkdir(parents=True)
        yield staging
        if atomic:
            if force and os.path.lexists(output):
                if output.is_symlink() or not output.is_dir():
                    raise ValueError("--force requires an existing regular directory")
                shutil.rmtree(output)
            publish_directory(staging, output)
    finally:
        if atomic:
            shutil.rmtree(staging, ignore_errors=True)


class Converter(abc.ABC):
    """Base class for model converters."""

    @staticmethod
    def declare_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds common conversion options to the command line parser.

        Arguments:
          parser: Command line argument parser.
        """
        parser.add_argument(
            "--output_dir", required=True, help="Output model directory."
        )
        parser.add_argument(
            "--vocab_mapping", default=None, help="Vocabulary mapping file (optional)."
        )
        parser.add_argument(
            "--quantization",
            default=None,
            choices=ACCEPTED_MODEL_TYPES,
            help="Weight quantization type.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force conversion even if the output directory already exists.",
        )
        return parser

    def convert_from_args(self, args: argparse.Namespace) -> str:
        """Helper function to call :meth:`ctranslate2.converters.Converter.convert`
        with the parsed command line options.

        Arguments:
          args: Namespace containing parsed arguments.

        Returns:
          Path to the output directory.
        """
        return self.convert(
            args.output_dir,
            vmap=args.vocab_mapping,
            quantization=args.quantization,
            force=args.force,
        )

    def convert(
        self,
        output_dir: str,
        vmap: Optional[str] = None,
        quantization: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Converts the model to the CTranslate2 format.

        Arguments:
          output_dir: Output directory where the CTranslate2 model is saved.
          vmap: Optional path to a vocabulary mapping file that will be included
            in the converted model directory.
          quantization: Weight quantization scheme (possible values are: int8, int8_float32,
            int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
          force: Override the output directory if it already exists.

        Returns:
          Path to the output directory.

        Raises:
          RuntimeError: If the output directory already exists and :obj:`force`
            is not set.
          NotImplementedError: If the converter cannot convert this model to the
            CTranslate2 format.
        """
        atomic = getattr(self, "_atomic_conversion", False)
        if atomic:
            Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(output_dir) and not force:
            if atomic:
                raise FileExistsError(
                    f"{output_dir} already exists; use --force to overwrite"
                )
            raise RuntimeError(
                "output directory %s already exists, use --force to override"
                % output_dir
            )

        model_spec = self._load()
        if model_spec is None:
            raise NotImplementedError(
                "This model is not supported by CTranslate2 or this converter"
            )
        if vmap is not None:
            model_spec.register_vocabulary_mapping(vmap)

        model_spec.validate()
        model_spec.optimize(quantization=quantization)

        with _output_directory(output_dir, force=force, atomic=atomic) as staging:
            model_spec.save(str(staging))
            report = getattr(self, "conversion_report", None)
            if report is not None:
                report.update(
                    output=str(output_dir),
                    output_bytes=(staging / "model.bin").stat().st_size,
                )
                self.write_report(staging / "conversion.json")
        return output_dir

    def write_report(self, path):
        """Write the current inspection or conversion report as JSON."""
        Path(path).write_text(json.dumps(self.conversion_report, indent=2) + "\n")

    @abc.abstractmethod
    def _load(self):
        raise NotImplementedError()
