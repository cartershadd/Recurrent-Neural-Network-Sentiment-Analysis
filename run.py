import tensorflow as tf
import data_utils

vocabulary, embeddings = data_utils.get_word_embeddings(50)

# add op to save and restore all variables.
saver = tf.train.import_meta_graph('model/model.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt")

    graph = tf.get_default_graph()

    for node in graph.as_graph_def().node:
        print("found node:", node.name)

    word_ids_placeholder = graph.get_tensor_by_name("word_ids_placeholder:0")
    logits = graph.get_tensor_by_name("logits/BiasAdd:0")

    while True:
        input_sentence = input("Type a sentence to test:")

        word_ids, converted_string = data_utils.convert_sentence(vocabulary, input_sentence)
        print(converted_string)
        print(word_ids)

        output = sess.run(logits, feed_dict={word_ids_placeholder: [word_ids]})

        print(output)
