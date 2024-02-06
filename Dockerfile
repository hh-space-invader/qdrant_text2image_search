# Stage 1: builder/compiler (will build everything and copy to second stage, will reduce container size)
# I will use same as my OS (debian 12) with python 3.11 for reproducible output
FROM python:3.11-slim-bookworm AS builder

# install system dependencies and clean the cache
RUN apt-get update \
&& pip3 install --upgrade pip wheel \
&& python3 -m venv /opt/venv

# activate the venv
ENV PATH="/opt/venv/bin:$PATH"

# copy python dependencies file
COPY requirements.txt /app/

# install python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt


# Stage 2: runtime (will drop the previous stage, and run the app)
# Use the slim-build image as the final image
FROM python:3.11-slim-bookworm AS runtime

# create a user
RUN useradd --create-home hossam

# copy less frequent changed files (python dependencies)
COPY --from=builder /opt/venv /opt/venv

# actiavte venv
ENV PATH="/opt/venv/bin:$PATH"

# switch user
USER hossam

# copy more frequent changed files (the rest of the code files)
COPY --chown=hossam . /app

# set working directory
WORKDIR /app

# exposed used ports
EXPOSE 8501

# run the app
ENTRYPOINT ["streamlit", "run", "src/app.py"]

