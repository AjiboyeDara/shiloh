FROM python:3.12-slim

# Hugging Face Spaces (and most managed container hosts) run the image as a
# non-root user, so everything the app touches at runtime has to belong to
# that user: Chroma opens its sqlite read-write, and the embedding model is
# read out of the cache setup.py fills. Building as root would leave both
# unreadable and the container would die on the first question.
# The workdir is created here rather than left to WORKDIR, which makes it
# root-owned on some Docker versions even after USER — and a root-owned
# workdir is exactly the failure this whole block exists to avoid.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/shiloh \
    && chown app:app /home/app/shiloh
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    HF_HOME=/home/app/.cache/huggingface
WORKDIR /home/app/shiloh

COPY --chown=app:app requirements.txt .

# sentence-transformers pulls in torch, and pip's default Linux wheel drags
# ~2 GB of CUDA libraries along with it. Retrieval runs all-MiniLM-L6-v2 on
# CPU and never touches a GPU, so install the CPU-only wheel first — the
# requirements install below then finds torch already satisfied. Order
# matters: reverse it and pip fetches the CUDA build before this runs.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app:app . .

# Fetch texts + resources and build the vector indexes at image build
# time so the container starts up ready to serve.
RUN python scripts/setup.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
