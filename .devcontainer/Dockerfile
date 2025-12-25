FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends sudo

ARG CONTAINER_WORKDIR
ENV CONTAINER_WORKDIR=$CONTAINER_WORKDIR
WORKDIR $CONTAINER_WORKDIR

ARG CONTAINER_USER
ENV CONTAINER_USER=$CONTAINER_USER
RUN useradd -ms /bin/bash $CONTAINER_USER
RUN echo $CONTAINER_USER:password | chpasswd
RUN echo "$CONTAINER_USER   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ユーザID等をホストに合わせる
COPY --chown=$CONTAINER_USER --chmod=755 .devcontainer/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

RUN apt update && apt install -y --no-install-recommends  \
    x11-apps \
    python3-tk \
    fonts-ipafont \
    git

RUN pip install --upgrade pip

RUN pip install numpy torch torchaudio torchvision matplotlib tqdm torchdiffeq ipython numba seaborn gymnasium gTTS librosa