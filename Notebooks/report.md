# Video Processing Report

**Source:** `C:\Users\vpved\Documents\GitHub\RactoGateway\Notebooks\data\Screen Recording 2026-01-23 160616.mp4`

**Frames extracted:** 12  **Kept:** 11  **Discarded:** 1

## Errors

### `transcribe` — FileNotFoundError

```
[WinError 2] The system cannot find the file specified
```

<details><summary>Traceback</summary>

```
Traceback (most recent call last):
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\src\ractogateway\pipelines\video_processor\pipeline.py", line 441, in _run_pipeline
    transcript = fut.result()
                 ^^^^^^^^^^^^
  File "C:\Users\vpved\AppData\Local\Programs\Python\Python311\Lib\concurrent\futures\_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vpved\AppData\Local\Programs\Python\Python311\Lib\concurrent\futures\_base.py", line 401, in __get_result
    raise self._exception
  File "C:\Users\vpved\AppData\Local\Programs\Python\Python311\Lib\concurrent\futures\thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\src\ractogateway\pipelines\video_processor\pipeline.py", line 543, in _run_transcription
    audio_path = extract_audio(video_path)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\src\ractogateway\pipelines\video_processor\_transcriber.py", line 170, in extract_audio
    .run(quiet=True)
     ^^^^^^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\venv\Lib\site-packages\ffmpeg\_run.py", line 313, in run
    process = run_async(
              ^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\venv\Lib\site-packages\ffmpeg\_run.py", line 284, in run_async
    return subprocess.Popen(
           ^^^^^^^^^^^^^^^^^
  File "C:\Users\vpved\AppData\Local\Programs\Python\Python311\Lib\subprocess.py", line 1026, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
  File "C:\Users\vpved\AppData\Local\Programs\Python\Python311\Lib\subprocess.py", line 1538, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [WinError 2] The system cannot find the file specified
```
</details>

### `analyze` — ValidationError

```
1 validation error for RactoPrompt
tone
  Field required [type=missing, input_value={'role': 'vision analysis...with labelled sections'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
```

<details><summary>Traceback</summary>

```
Traceback (most recent call last):
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\src\ractogateway\pipelines\video_processor\pipeline.py", line 454, in _run_pipeline
    frames = analyze_frames_sync(
             ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\src\ractogateway\pipelines\video_processor\_analyzer.py", line 258, in analyze_frames_sync
    text, usg = fut.result()
                ^^^^^^^^^^^^
  File "C:\Users\vpved\AppData\Local\Programs\Python\Python311\Lib\concurrent\futures\_base.py", line 449, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vpved\AppData\Local\Programs\Python\Python311\Lib\concurrent\futures\_base.py", line 401, in __get_result
    raise self._exception
  File "C:\Users\vpved\AppData\Local\Programs\Python\Python311\Lib\concurrent\futures\thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\src\ractogateway\pipelines\video_processor\_analyzer.py", line 145, in _analyze_single_frame_sync
    prompt = RactoPrompt(
             ^^^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\venv\Lib\site-packages\pydantic\main.py", line 250, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for RactoPrompt
tone
  Field required [type=missing, input_value={'role': 'vision analysis...with labelled sections'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
```
</details>

### `summarize` — ValidationError

```
1 validation error for RactoPrompt
tone
  Field required [type=missing, input_value={'role': 'expert lecture ...s]\n\n[10.0s -10.0s]\n'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
```

<details><summary>Traceback</summary>

```
Traceback (most recent call last):
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\src\ractogateway\pipelines\video_processor\pipeline.py", line 487, in _run_pipeline
    summary = generate_summary_sync(
              ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\src\ractogateway\pipelines\video_processor\_summarizer.py", line 88, in generate_summary_sync
    prompt = RactoPrompt(
             ^^^^^^^^^^^^
  File "C:\Users\vpved\Documents\GitHub\RactoGateway\venv\Lib\site-packages\pydantic\main.py", line 250, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for RactoPrompt
tone
  Field required [type=missing, input_value={'role': 'expert lecture ...s]\n\n[10.0s -10.0s]\n'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing
```
</details>

## Sections

### 3.0s - 3.0s

### 1.0s - 1.0s

### 0.0s - 0.0s

### 2.0s - 2.0s

### 7.0s - 7.0s

### 4.0s - 4.0s

### 6.0s - 6.0s

### 11.0s - 11.0s

### 9.0s - 9.0s

### 8.0s - 8.0s

### 10.0s - 10.0s
