FROM tensorflow/tensorflow:latest
# add user 'app'
RUN adduser --disabled-password --gecos "" app && \
    echo "app ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "app:app" | chpasswd && \
    chown -R app:app /home/app

COPY requirements.txt /home/app
COPY tensorflow_tasks /home/app/tensorflow_tasks

RUN pip install -r /home/app/requirements.txt

USER app
WORKDIR "/home/app"

CMD ["python"]