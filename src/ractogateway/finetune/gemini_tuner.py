"""Google Gemini fine-tuning adapter for RactoGateway.

Workflow
--------
1. Build a text-only :class:`~ractogateway.finetune.dataset.RactoDataset`
   of single-turn pairs (each example serialises to ``text_input`` / ``output``).
2. Call :meth:`GeminiFineTuner.run_pipeline` for a one-shot run,
   **or** call the lower-level methods:

   a. :meth:`create_job`          → tuning operation
   b. :meth:`wait_for_completion` → ``tuned_model_name``

Supported base models (tuning-enabled, as of 2025)
----------------------------------------------------
- ``models/gemini-1.5-flash-001-tuning``  (recommended)
- ``models/gemini-1.0-pro-001``

Notes
-----
* The ``google-generativeai`` SDK supports **text-only, single-turn**
  fine-tuning via the Generative Language Tuning API.
* **Multimodal** or **multi-turn** training requires Google Vertex AI
  (``google-cloud-aiplatform``).  Use the ``to_gemini_dict()`` method on
  each example to get the ``contents`` format for Vertex AI.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ractogateway.finetune.dataset import RactoDataset


def _require_genai() -> Any:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "The 'google-generativeai' package is required for GeminiFineTuner. "
            "Install it with:  pip install ractogateway[google]"
        ) from exc
    return genai


class GeminiFineTuner:
    """Fine-tune Google Gemini models using the Generative AI tuning API.

    Parameters
    ----------
    api_key : str | None
        Google AI API key.  Falls back to the ``GEMINI_API_KEY`` environment
        variable when not supplied.

    Examples
    --------
    End-to-end pipeline::

        from ractogateway.finetune import RactoDataset, GeminiFineTuner

        ds = RactoDataset.from_pairs(
            [("capital of France?", "Paris"), ("capital of Japan?", "Tokyo")],
        )
        tuner = GeminiFineTuner()
        model_name = tuner.run_pipeline(
            ds,
            base_model="models/gemini-1.5-flash-001-tuning",
            display_name="geography-tutor",
        )
        print(model_name)  # "tunedModels/geography-tutor-abc123"
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def _configure(self) -> Any:
        genai = _require_genai()
        if self.api_key:
            genai.configure(api_key=self.api_key)
        return genai

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def create_job(
        self,
        dataset: RactoDataset,
        base_model: str = "models/gemini-1.5-flash-001-tuning",
        *,
        display_name: str = "",
        epoch_count: int = 5,
        batch_size: int = 4,
        learning_rate: float | None = None,
    ) -> Any:
        """Start a Gemini supervised fine-tuning job.

        Parameters
        ----------
        dataset : RactoDataset
            Training examples.  Each example **must** be a single-turn
            text pair (``text_input`` / ``output``).  Examples with
            attachments or multi-turn conversations are not supported
            by this adapter — use Vertex AI for those.
        base_model : str
            Tuning-enabled Gemini model identifier.
        display_name : str
            Human-readable label for the tuned model.
        epoch_count : int
            Number of training epochs.
        batch_size : int
            Training batch size.
        learning_rate : float | None
            Learning rate.  ``None`` uses the provider default.

        Returns
        -------
        google.generativeai.types.TunedModel  (operation-like object)
            Pass to :meth:`wait_for_completion`.

        Raises
        ------
        ValueError
            If the dataset fails validation, or if any examples are
            multimodal / multi-turn (unsupported by this adapter).
        """
        genai = self._configure()

        errors = dataset.validate("gemini")
        if errors:
            raise ValueError("Dataset validation failed:\n" + "\n".join(errors))

        training_data: list[dict[str, Any]] = []
        for ex in dataset:
            record = ex.to_gemini_dict()
            if "text_input" not in record:
                raise ValueError(
                    "GeminiFineTuner only supports single-turn text-pair examples. "
                    "Multimodal or multi-turn examples require Vertex AI.  "
                    "Found a 'contents' record — remove attachments or multi-turn "
                    "turns from this example."
                )
            training_data.append(record)

        kwargs: dict[str, Any] = {
            "source_model": base_model,
            "training_data": training_data,
            "epoch_count": epoch_count,
            "batch_size": batch_size,
        }
        if display_name:
            kwargs["display_name"] = display_name
        if learning_rate is not None:
            kwargs["learning_rate"] = learning_rate

        return genai.create_tuned_model(**kwargs)

    def get_model(self, tuned_model_name: str) -> dict[str, Any]:
        """Retrieve metadata for a tuned model.

        Parameters
        ----------
        tuned_model_name : str
            Full tuned model name, e.g. ``"tunedModels/my-model-abc123"``.

        Returns
        -------
        dict
            Keys: ``name``, ``display_name``, ``state``, ``base_model``.
        """
        genai = self._configure()
        model = genai.get_tuned_model(tuned_model_name)
        return {
            "name": model.name,
            "display_name": getattr(model, "display_name", ""),
            "state": str(model.state),
            "base_model": getattr(model, "base_model", ""),
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List all tuned models in this project."""
        genai = self._configure()
        return [
            {
                "name": m.name,
                "display_name": getattr(m, "display_name", ""),
                "state": str(m.state),
                "base_model": getattr(m, "base_model", ""),
            }
            for m in genai.list_tuned_models()
        ]

    def delete_model(self, tuned_model_name: str) -> None:
        """Permanently delete a tuned model from your project."""
        genai = self._configure()
        genai.delete_tuned_model(tuned_model_name)

    def wait_for_completion(
        self,
        operation: Any,
        *,
        poll_interval: int = 60,
        verbose: bool = True,
    ) -> str:
        """Block until a tuning operation finishes.

        Parameters
        ----------
        operation : TuningOperation
            The object returned by :meth:`create_job`.
        poll_interval : int
            Seconds between metadata checks.
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        str
            Tuned model name (e.g. ``"tunedModels/my-model-abc123"``).
            Pass directly to ``GoogleDeveloperKit(model=...)``.

        Raises
        ------
        RuntimeError
            If the tuning job ends in a failed state.
        """
        while True:
            metadata = operation.metadata
            state = str(getattr(metadata, "state", "UNKNOWN"))
            if verbose:
                completed = getattr(metadata, "completed_percent", None)
                pct = f" ({completed:.0f}%)" if completed is not None else ""
                print(f"[GeminiFineTuner] State: {state}{pct}")

            if "ACTIVE" in state or "FAILED" in state:
                break

            try:
                operation.result(timeout=poll_interval)
                break
            except Exception:
                time.sleep(poll_interval)

        if "FAILED" in state:
            raise RuntimeError(f"Gemini tuning job failed.  Final state: {state}")
        result = operation.result()
        model_name = getattr(result, "name", None)
        if not isinstance(model_name, str) or not model_name:
            raise RuntimeError("Gemini tuning completed but no tuned model name was returned.")
        return model_name

    # ------------------------------------------------------------------
    # High-level pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        dataset: RactoDataset,
        base_model: str = "models/gemini-1.5-flash-001-tuning",
        *,
        display_name: str = "",
        epoch_count: int = 5,
        batch_size: int = 4,
        learning_rate: float | None = None,
        poll_interval: int = 60,
        verbose: bool = True,
    ) -> str:
        """Validate → create → wait in a single call.

        Parameters
        ----------
        dataset : RactoDataset
            Text-pair training examples.
        base_model : str
            Tuning-enabled Gemini model.
        display_name : str
            Human-readable label for the tuned model.
        epoch_count : int
            Training epochs.
        batch_size : int
            Training batch size.
        learning_rate : float | None
            Learning rate override.
        poll_interval : int
            Seconds between status polls.
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        str
            Tuned model name — pass to ``GoogleDeveloperKit(model=...)``.
        """
        if verbose:
            stats = dataset.summary()
            print(f"[GeminiFineTuner] Starting tuning with {stats['examples']} examples…")
        operation = self.create_job(
            dataset,
            base_model,
            display_name=display_name,
            epoch_count=epoch_count,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        tuned_model = self.wait_for_completion(
            operation, poll_interval=poll_interval, verbose=verbose
        )
        if verbose:
            print(f"[GeminiFineTuner] Done!  Tuned model: {tuned_model}")
        return tuned_model
