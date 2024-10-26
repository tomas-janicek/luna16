Contains traces and other metadata for profiled models.

## Profiling with `profilehooks`

```python
from profilehooks import profile

@profile(
    filename=settings.PROFILING_DIR
    / f"<profile_name>_{common_utils.get_datetime_string(datetime.datetime.now())}.prof",
    stdout=False,
    dirs=True,
)  # type: ignore
```

### Visualizing `.prof` file

```bash
uv run snakeviz profiling/<profile_name>_<datetime>.prof
```