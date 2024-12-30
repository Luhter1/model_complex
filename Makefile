autoformatting:
	pre-commit install

remove_cache:
	find . -name "__pycache__" -type d -exec rm -rf {} +

isort:
	isort ./model_complex/*.py ./model_complex/*/*.py ./model_complex/*/*/*.py

black:
	black ./model_complex/*.py ./model_complex/*/*.py ./model_complex/*/*/*.py

install:
	poetry install --with dev

add:
	poetry add ${NAME}

add_dev:
	poetry add --group dev ${NAME}
