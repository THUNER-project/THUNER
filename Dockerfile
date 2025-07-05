FROM eshort0401/thuner-base:latest
COPY pyproject.toml .
RUN pip install -v .
COPY . .
CMD ["/bin/bash"]