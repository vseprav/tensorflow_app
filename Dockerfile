FROM tensorflow/tensorflow:latest
# add user 'app'
RUN adduser --disabled-password --gecos "" app && \
    echo "app ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "app:app" | chpasswd && \
    chown -R app:app /home/app

COPY tensorflow_tasks /home/app/tensorflow_tasks

USER app
WORKDIR "/home/app"

ENTRYPOINT ["./tensorflow_tasks/runtests.sh"]

CMD ["python"]