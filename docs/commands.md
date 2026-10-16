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

## Docker

```bash
docker build -t mlops-final-project:v1 .
```

## DVC

### Reproduce 

```
uv run dvc repro -f
```

## BentoML

```
bentoml serve 
```

## Locust (does not work with venv using a different one -> conflicts)

- you Should run the server before these commands. 

### main

#### Fastapi

```
locust -f locustfile.py FastAPIUser --host=http://localhost:8000
```

#### Headless Fastapi
```
locust -f locustfile.py FastAPIUser --host=http://localhost:8000 --headless -u 10 -r 2 --run-time 1m
```

#### BentoML
```
locust -f locustfile.py BentoMLUser --host=http://localhost:8000
```

#### Headless BentoML
```
locust -f locustfile.py BentoMLUser --host=http://localhost:8000 --headless -u 20 -r 4 --run-time 1m
```

#### Kubernetes
```
kubectl get all -n production

```