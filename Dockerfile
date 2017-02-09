FROM tensorflow/tensorflow:latest
COPY tensorflow_tasks /tensorflow_tasks
WORKDIR "/tensorflow_tasks"

ENTRYPOINT ["./runtests.sh"]

CMD ["python"]