run $experiment:
    echo "Running '$experiment'"
    uv run "./experiments/$experiment/main.py"

lint:
    uv run ruff check .

format:
    uv run ruff format .
