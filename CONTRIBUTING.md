# Contributing to Cross-Backend AI Comparison

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/sudheerdevu/Cross-Backend-AI-Comparison.git
cd Cross-Backend-AI-Comparison

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install pytest black isort
```

## Running Tests

```bash
pytest tests/ -v
```

## Adding New Backends

1. Create `src/backends/<backend_name>/__init__.py`
2. Implement `BackendRunner` interface
3. Add configuration in `configs/`
4. Add tests
5. Update documentation

## Code Style

```bash
black src/ tests/
isort src/ tests/
```

## Pull Request Process

1. Fork repository
2. Create feature branch
3. Add tests for new backends
4. Submit PR

## License

Contributions are licensed under MIT License.
