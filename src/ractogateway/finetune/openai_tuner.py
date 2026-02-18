"""OpenAI fine-tuning adapter for RactoGateway.

Workflow
--------
1. Build a :class:`~ractogateway.finetune.dataset.RactoDataset`.
2. Call :meth:`OpenAIFineTuner.run_pipeline` for a one-shot end-to-end run,
   **or** call the lower-level methods individually:

   a. :meth:`upload_dataset`   → ``file_id``
   b. :meth:`create_job`       → ``job_id``
   c. :meth:`wait_for_completion` → ``fine_tuned_model``

Supported base models (as of 2025)
------------------------------------
- ``gpt-4o-mini-2024-07-18``  — recommended; cost-effective
- ``gpt-4o-2024-08-06``       — multimodal vision fine-tuning
- ``gpt-3.5-turbo-0125``      — legacy option
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

from ractogateway.finetune.dataset import RactoDataset


def _require_openai() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for OpenAIFineTuner. "
            "Install it with:  pip install ractogateway[openai]"
        ) from exc
    return openai


class OpenAIFineTuner:
    """Fine-tune OpenAI models using the fine-tuning API.

    Parameters
    ----------
    api_key : str | None
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when not supplied.
    base_url : str | None
        Optional custom base URL (Azure OpenAI, proxy, etc.).

    Examples
    --------
    End-to-end pipeline (simplest usage)::

        from ractogateway.finetune import RactoDataset, OpenAIFineTuner

        ds = RactoDataset.from_pairs(
            [("What is Python?", "A high-level programming language.")],
            system="You are a Python tutor.",
        )
        tuner = OpenAIFineTuner()
        model = tuner.run_pipeline(ds, model="gpt-4o-mini-2024-07-18")
        print(model)   # "ft:gpt-4o-mini-2024-07-18:org::abc123"
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url

    def _client(self) -> Any:
        openai = _require_openai()
        params: dict[str, Any] = {}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["base_url"] = self.base_url
        return openai.OpenAI(**params)

    # ------------------------------------------------------------------
    # Dataset upload
    # ------------------------------------------------------------------

    def upload_dataset(self, dataset: RactoDataset) -> str:
        """Upload *dataset* as an OpenAI training file.

        Parameters
        ----------
        dataset : RactoDataset
            The training examples to upload.

        Returns
        -------
        str
            The OpenAI file ID (e.g. ``"file-abc123"``).
        """
        client = self._client()
        jsonl_bytes = dataset.to_jsonl_string("openai").encode("utf-8")
        buf = io.BytesIO(jsonl_bytes)
        buf.name = "training_data.jsonl"
        response = client.files.create(file=buf, purpose="fine-tune")
        return response.id

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def create_job(
        self,
        training_file: str,
        model: str = "gpt-4o-mini-2024-07-18",
        *,
        validation_file: str | None = None,
        n_epochs: int | str = "auto",
        batch_size: int | str = "auto",
        learning_rate_multiplier: float | str = "auto",
        suffix: str | None = None,
    ) -> str:
        """Submit a fine-tuning job.

        Parameters
        ----------
        training_file : str
            File ID returned by :meth:`upload_dataset`.
        model : str
            Base model to fine-tune.
        validation_file : str | None
            Optional validation file ID (also produced by
            :meth:`upload_dataset`).
        n_epochs : int | "auto"
            Training epochs.
        batch_size : int | "auto"
            Per-device batch size.
        learning_rate_multiplier : float | "auto"
            Scales the default learning rate.
        suffix : str | None
            Custom label appended to the fine-tuned model name.

        Returns
        -------
        str
            The fine-tuning job ID (e.g. ``"ftjob-abc123"``).
        """
        client = self._client()
        hyperparams: dict[str, Any] = {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate_multiplier": learning_rate_multiplier,
        }
        kwargs: dict[str, Any] = {
            "model": model,
            "training_file": training_file,
            "hyperparameters": hyperparams,
        }
        if validation_file:
            kwargs["validation_file"] = validation_file
        if suffix:
            kwargs["suffix"] = suffix
        job = client.fine_tuning.jobs.create(**kwargs)
        return job.id

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Retrieve the current status of a fine-tuning job.

        Returns
        -------
        dict
            Keys: ``id``, ``status``, ``model``, ``fine_tuned_model``,
            ``created_at``, ``finished_at``, ``trained_tokens``, ``error``.
        """
        client = self._client()
        job = client.fine_tuning.jobs.retrieve(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "created_at": job.created_at,
            "finished_at": getattr(job, "finished_at", None),
            "trained_tokens": getattr(job, "trained_tokens", None),
            "error": getattr(job, "error", None),
        }

    def list_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent fine-tuning jobs (newest first)."""
        client = self._client()
        page = client.fine_tuning.jobs.list(limit=limit)
        return [self.get_status(job.id) for job in page.data]

    def list_events(
        self, job_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return recent training log events for a job."""
        client = self._client()
        events = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id, limit=limit
        )
        return [
            {
                "message": e.message,
                "level": e.level,
                "created_at": e.created_at,
            }
            for e in events.data
        ]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a running fine-tuning job."""
        client = self._client()
        job = client.fine_tuning.jobs.cancel(job_id)
        return {"id": job.id, "status": job.status}

    def wait_for_completion(
        self,
        job_id: str,
        *,
        poll_interval: int = 30,
        verbose: bool = True,
    ) -> str:
        """Block until a fine-tuning job finishes.

        Parameters
        ----------
        job_id : str
            The job ID returned by :meth:`create_job`.
        poll_interval : int
            Seconds between status-check API calls.
        verbose : bool
            Print status lines to stdout.

        Returns
        -------
        str
            The fine-tuned model name ready for use in :class:`OpenAILLMKit`.

        Raises
        ------
        RuntimeError
            If the job ends in ``"failed"`` or ``"cancelled"`` state.
        """
        terminal = {"succeeded", "failed", "cancelled"}
        status: dict[str, Any] = {}
        while True:
            status = self.get_status(job_id)
            state: str = status["status"]
            if verbose:
                print(f"[OpenAIFineTuner] Job {job_id} → {state}")
            if state in terminal:
                break
            time.sleep(poll_interval)

        if status["status"] != "succeeded":
            error = status.get("error") or "Unknown error"
            raise RuntimeError(
                f"Fine-tuning job {job_id} ended with status "
                f"'{status['status']}': {error}"
            )
        return status["fine_tuned_model"]

    # ------------------------------------------------------------------
    # High-level pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        dataset: RactoDataset,
        model: str = "gpt-4o-mini-2024-07-18",
        *,
        validation_dataset: RactoDataset | None = None,
        n_epochs: int | str = "auto",
        batch_size: int | str = "auto",
        learning_rate_multiplier: float | str = "auto",
        suffix: str | None = None,
        poll_interval: int = 30,
        verbose: bool = True,
    ) -> str:
        """Validate → upload → train → wait in a single call.

        This is the recommended entry-point for most use cases.

        Parameters
        ----------
        dataset : RactoDataset
            Training examples.
        model : str
            Base model to fine-tune.
        validation_dataset : RactoDataset | None
            Optional held-out validation set (uploaded separately).
        n_epochs, batch_size, learning_rate_multiplier : int | float | "auto"
            Training hyperparameters.  Pass ``"auto"`` to let OpenAI decide.
        suffix : str | None
            Short label appended to the fine-tuned model name.
        poll_interval : int
            Seconds between status polls while waiting.
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        str
            Fine-tuned model identifier — pass directly to
            ``OpenAIDeveloperKit(model=...)``::

                kit = opd.OpenAIDeveloperKit(model=fine_tuned_model)

        Raises
        ------
        ValueError
            If dataset validation fails.
        RuntimeError
            If the fine-tuning job fails remotely.
        """
        errors = dataset.validate("openai")
        if errors:
            raise ValueError(
                "Dataset validation failed:\n" + "\n".join(errors)
            )

        if verbose:
            stats = dataset.summary()
            print(
                f"[OpenAIFineTuner] Uploading {stats['examples']} training examples "
                f"({stats['multimodal_examples']} multimodal)…"
            )
        training_file = self.upload_dataset(dataset)
        if verbose:
            print(f"[OpenAIFineTuner] Training file: {training_file}")

        validation_file: str | None = None
        if validation_dataset:
            if verbose:
                print(
                    f"[OpenAIFineTuner] Uploading "
                    f"{len(validation_dataset)} validation examples…"
                )
            validation_file = self.upload_dataset(validation_dataset)
            if verbose:
                print(f"[OpenAIFineTuner] Validation file: {validation_file}")

        job_id = self.create_job(
            training_file,
            model,
            validation_file=validation_file,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
            suffix=suffix,
        )
        if verbose:
            print(f"[OpenAIFineTuner] Job created: {job_id}")

        fine_tuned_model = self.wait_for_completion(
            job_id, poll_interval=poll_interval, verbose=verbose
        )
        if verbose:
            print(f"[OpenAIFineTuner] Done!  Fine-tuned model: {fine_tuned_model}")
        return fine_tuned_model
