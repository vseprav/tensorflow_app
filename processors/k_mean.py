import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/data.csv"

points_n = 200
clusters_n = 2
iteration_n = 100

#Training inputs
data = [
  [39750,3],
  [5000,1],
  [68069,5],
  [10125,3],
  [33000,3],
  [250,1],
  [100,1]
]

# def create_file_reader_ops(filename):
#   reader = tf.TextLineReader()
#   _, csv_row = reader.read(filename)
#   record_defaults = [[0], [0]]
#   funding_total, funding_rounds = tf.decode_csv(csv_row, record_defaults=record_defaults)

#   return funding_total, funding_rounds

# filename = tf.train.string_input_producer(["data.csv"])
# data = create_file_reader_ops(filename)

points = tf.constant(data)
centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0,0], [clusters_n, -1]))

points_expanded = tf.expand_dims(points, 0)
centroids_expanded = tf.expand_dims(centroids, 1)

distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
assignments = tf.argmin(distances, 0)

means = []
for c in xrange(clusters_n):
  means.append(tf.reduce_mean(
    tf.gather(points,
      tf.reshape(
        tf.where(
          tf.equal(assignments, c)
        ),[1,-1])
     ),reduction_indices=[1]))

new_centroids = tf.concat(means, 0)
update_centroids = tf.assign(centroids, new_centroids)

init = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init)

  for step in xrange(iteration_n):
    [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])

print centroid_values
print '\n'
print points_values
print '\n'
print assignment_values

plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.6)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()
