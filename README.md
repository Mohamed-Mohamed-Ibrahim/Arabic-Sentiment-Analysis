# Arabic-Sentiment-Analysis

## Testing

### Pytest

```bash
python3 -m pytest -v tests/
```

### Coverage

```bash
coverage run -m pytest tests/
coverage html
coverage report -m
```

### Docker

```bash
docker build -t mlops-final-project:v1
```