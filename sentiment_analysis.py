import data_utils
import tensorflow as tf

TENSORBOARD_LOGDIR = "logdir"

# Clear the old log files
data_utils.delete_directory(TENSORBOARD_LOGDIR)

data_utils.download_data()

embedding_size = 50  # downloaded embeddings are 50, 100, 200, 300

vocabulary, all_embeddings = data_utils.get_word_embeddings(embedding_size)

word_ids_placeholder = tf.placeholder(tf.int32, [None, data_utils.FIXED_STRING_LENGTH], name="word_ids_placeholder")
word_embeddings_tensor = tf.constant(all_embeddings)

# DONE build model
embeddings_tensor = tf.nn.embedding_lookup(word_embeddings_tensor, word_ids_placeholder)

# create 3 LSTM cells
rnn_layers = [tf.nn.rnn_cell.GRUCell(128),
              tf.nn.rnn_cell.LSTMCell(256),
              tf.nn.rnn_cell.GRUCell(40)]

# create an RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

outputs, state = tf.nn.dynamic_rnn(cell = multi_rnn_cell,
                                   inputs=embeddings_tensor,
                                   dtype=tf.float32)

# Logit layer
logits = tf.layers.dense(outputs[:, -1, :], 2, name="logits")

# label placeholder
label_placeholder = tf.placeholder(tf.uint8, shape=[None, 2])
# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_placeholder, logits=logits))
# backpropagation algorithm
train = tf.train.AdamOptimizer().minimize(loss)

accuracy = data_utils.accuracy(logits, label_placeholder)

# summaries
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)

tf.summary.histogram("logits", logits)
tf.summary.histogram("labels", label_placeholder)
summary_tensor = tf.summary.merge_all()

saver = tf.train.Saver()

# Make tensorflow session
with tf.Session() as sess:
    training_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/training", sess.graph)
    test_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/test" , sess.graph)

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    step_count = 0
    while True:
        step_count += 1

        batch_training_data, batch_training_labels, batch_training_string = data_utils.get_sentence_batch(
            vocabulary=vocabulary,
            batch_size=42,
            is_eval=False)

        # for i in range(50):
        #     print(len(batch_training_data[i]), batch_training_data[i])
        #     print(len(batch_training_labels[i]), batch_training_labels[i])

        print(len(batch_training_data), batch_training_data[0])
        print(len(batch_training_labels), batch_training_labels[0])

        print("neg", batch_training_string[0])
        print("pos", batch_training_string[-1])

        # train network
        training_accuracy, training_loss, summary,  _ = sess.run([accuracy, loss, summary_tensor, train],
                                                                 feed_dict={word_ids_placeholder: batch_training_data,
                                                                         label_placeholder: batch_training_labels})

        # write data to tensorboard
        training_summary_writer.add_summary(summary, step_count)

        # every 10 steps check accuracy
        if step_count % 10 ==  0:
            batch_test_data, batch_test_labels, batch_test_string = data_utils.get_sentence_batch(
                vocabulary=vocabulary,
                batch_size=50,
                is_eval=True)
            print("neg", batch_test_string[0])
            print("pos", batch_test_string[-1])
            test_accuracy, test_loss, summary = sess.run([accuracy, loss, summary_tensor],
                                                         feed_dict={
                                                             word_ids_placeholder: batch_test_data,
                                                             label_placeholder: batch_test_labels})

            # write data to tensorboard
            test_summary_writer.add_summary(summary, step_count)

            print("Step Count:{}".format(step_count))
            print("Training accuracy: {:.6f} loss: {:.6f}".format(training_accuracy, training_loss))
            print("Test accuracy: {:.6f} loss: {:.6f}".format(test_accuracy, test_loss))


        if step_count % 100 == 0:
            save_path = saver.save(sess, "model/model.ckpt")

        # stop training after 1,000 steps
        if step_count > 10000:
            break
