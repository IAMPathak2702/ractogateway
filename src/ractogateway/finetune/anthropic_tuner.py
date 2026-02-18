"""Anthropic Claude fine-tuning adapter for RactoGateway.

Workflow
--------
1. Build a :class:`~ractogateway.finetune.dataset.RactoDataset`.
2. Call :meth:`AnthropicFineTuner.run_pipeline` for a one-shot end-to-end run,
   **or** call the lower-level methods:

   a. :meth:`upload_dataset`      → ``file_id``
   b. :meth:`create_job`          → ``job_id``
   c. :meth:`wait_for_completion` → ``fine_tuned_model``

Supported base models (as of 2025)
------------------------------------
- ``claude-3-haiku-20240307``  — primary fine-tuning target

Notes
-----
Anthropic fine-tuning requires access approval.  Verify the exact API
surface against the latest Anthropic documentation before using this adapter,
as the fine-tuning API may evolve.

JSONL training record format (one line per example)::

    {
        "system": "Optional system prompt",
        "messages": [
            {"role": "user",      "content": "…"},
            {"role": "assistant", "content": "…"}
        ]
    }
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

from ractogateway.finetune.dataset import RactoDataset


def _require_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for AnthropicFineTuner. "
            "Install it with:  pip install ractogateway[anthropic]"
        ) from exc
    return anthropic


class AnthropicFineTuner:
    """Fine-tune Anthropic Claude models using the fine-tuning API.

    Parameters
    ----------
    api_key : str | None
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` environment
        variable when not supplied.

    Examples
    --------
    End-to-end pipeline::

        from ractogateway.finetune import RactoDataset, AnthropicFineTuner

        ds = RactoDataset.from_pairs(
            [("Summarise this: …", "The text discusses…")],
            system="You are a concise summariser.",
        )
        tuner = AnthropicFineTuner()
        model = tuner.run_pipeline(ds, model="claude-3-haiku-20240307")
        print(model)   # "claude-3-haiku-20240307:ft:org-xxx:suffix:abc123"
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def _client(self) -> Any:
        anthropic = _require_anthropic()
        params: dict[str, Any] = {}
        if self.api_key:
            params["api_key"] = self.api_key
        return anthropic.Anthropic(**params)

    # ------------------------------------------------------------------
    # Dataset upload
    # ------------------------------------------------------------------

    def upload_dataset(self, dataset: RactoDataset) -> str:
        """Upload *dataset* as an Anthropic training file.

        Parameters
        ----------
        dataset : RactoDataset
            The training examples to upload.

        Returns
        -------
        str
            The Anthropic file ID used in :meth:`create_job`.
        """
        client = self._client()
        jsonl_bytes = dataset.to_jsonl_string("anthropic").encode("utf-8")
        buf = io.BytesIO(jsonl_bytes)
        buf.name = "training_data.jsonl"
        response = client.beta.files.upload(
            file=("training_data.jsonl", buf, "application/jsonl"),
        )
        return response.id

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def create_job(
        self,
        training_file: str,
        model: str = "claude-3-haiku-20240307",
        *,
        validation_file: str | None = None,
        suffix: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> str:
        """Submit a fine-tuning job.

        Parameters
        ----------
        training_file : str
            File ID returned by :meth:`upload_dataset`.
        model : str
            Base Claude model to fine-tune.
        validation_file : str | None
            Optional validation file ID.
        suffix : str | None
            Short label appended to the fine-tuned model name.
        hyperparameters : dict | None
            Optional overrides, e.g. ``{"n_epochs": 3}``.

        Returns
        -------
        str
            The fine-tuning job ID.
        """
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": model,
            "training_file": training_file,
        }
        if validation_file:
            kwargs["validation_file"] = validation_file
        if suffix:
            kwargs["suffix"] = suffix
        if hyperparameters:
            kwargs["hyperparameters"] = hyperparameters
        job = client.fine_tuning.jobs.create(**kwargs)
        return job.id

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Retrieve the current status of a fine-tuning job.

        Returns
        -------
        dict
            Keys: ``id``, ``status``, ``model``, ``fine_tuned_model``,
            ``created_at``, ``finished_at``, ``error``.
        """
        client = self._client()
        job = client.fine_tuning.jobs.retrieve(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "created_at": getattr(job, "created_at", None),
            "finished_at": getattr(job, "finished_at", None),
            "error": getattr(job, "error", None),
        }

    def list_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent fine-tuning jobs (newest first)."""
        client = self._client()
        page = client.fine_tuning.jobs.list(limit=limit)
        return [self.get_status(job.id) for job in page.data]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a running fine-tuning job."""
        client = self._client()
        job = client.fine_tuning.jobs.cancel(job_id)
        return {"id": job.id, "status": job.status}

    def wait_for_completion(
        self,
        job_id: str,
        *,
        poll_interval: int = 60,
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
            Fine-tuned model name — pass directly to
            ``AnthropicDeveloperKit(model=...)``.

        Raises
        ------
        RuntimeError
            If the job ends in ``"failed"`` or ``"cancelled"`` state.
        """
        terminal = {"succeeded", "failed", "cancelled", "completed"}
        status: dict[str, Any] = {}
        while True:
            status = self.get_status(job_id)
            state: str = status["status"]
            if verbose:
                print(f"[AnthropicFineTuner] Job {job_id} → {state}")
            if state in terminal:
                break
            time.sleep(poll_interval)

        if status["status"] not in ("succeeded", "completed"):
            error = status.get("error") or "Unknown error"
            raise RuntimeError(
                f"Fine-tuning job {job_id} ended with status "
                f"'{status['status']}': {error}"
            )
        return status["fine_tuned_model"] or job_id

    # ------------------------------------------------------------------
    # High-level pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        dataset: RactoDataset,
        model: str = "claude-3-haiku-20240307",
        *,
        validation_dataset: RactoDataset | None = None,
        suffix: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        poll_interval: int = 60,
        verbose: bool = True,
    ) -> str:
        """Validate → upload → train → wait in a single call.

        Parameters
        ----------
        dataset : RactoDataset
            Training examples.
        model : str
            Base Claude model to fine-tune.
        validation_dataset : RactoDataset | None
            Optional held-out validation set.
        suffix : str | None
            Short label appended to the fine-tuned model name.
        hyperparameters : dict | None
            Optional overrides, e.g. ``{"n_epochs": 3}``.
        poll_interval : int
            Seconds between status polls.
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        str
            Fine-tuned model identifier — pass directly to
            ``AnthropicDeveloperKit(model=...)``.

        Raises
        ------
        ValueError
            If dataset validation fails.
        RuntimeError
            If the fine-tuning job fails remotely.
        """
        errors = dataset.validate("anthropic")
        if errors:
            raise ValueError(
                "Dataset validation failed:\n" + "\n".join(errors)
            )

        if verbose:
            stats = dataset.summary()
            print(
                f"[AnthropicFineTuner] Uploading {stats['examples']} training examples "
                f"({stats['multimodal_examples']} multimodal)…"
            )
        training_file = self.upload_dataset(dataset)
        if verbose:
            print(f"[AnthropicFineTuner] Training file: {training_file}")

        validation_file: str | None = None
        if validation_dataset:
            if verbose:
                print(
                    f"[AnthropicFineTuner] Uploading "
                    f"{len(validation_dataset)} validation examples…"
                )
            validation_file = self.upload_dataset(validation_dataset)
            if verbose:
                print(f"[AnthropicFineTuner] Validation file: {validation_file}")

        job_id = self.create_job(
            training_file,
            model,
            validation_file=validation_file,
            suffix=suffix,
            hyperparameters=hyperparameters,
        )
        if verbose:
            print(f"[AnthropicFineTuner] Job created: {job_id}")

        fine_tuned_model = self.wait_for_completion(
            job_id, poll_interval=poll_interval, verbose=verbose
        )
        if verbose:
            print(
                f"[AnthropicFineTuner] Done!  Fine-tuned model: {fine_tuned_model}"
            )
        return fine_tuned_model
