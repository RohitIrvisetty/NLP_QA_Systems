import tensorflow as tf
import json
import math
from augment.utils import (input_fn_builder)
from models.cnn import CNNConfig, create_cnn_model
from models.contextualized_cnn import create_contextualized_cnn_model, ContextualizedCNNConfig
import time
from augment.train import (FLAGS, bert_config, OUTPUT_DIR, N_TRAIN_EXAMPLES,
                           EVAL_FILE_NAME, N_TOTAL_SQUAD_EXAMPLES, model_fn_builder, INIT_CHECKPOINT, TRAIN_FILE_NAME)


def config_and_model(filename):
    with tf.gfile.GFile(filename, 'r') as json_data:
        parsed = json.load(json_data)
        parsed['max_seq_length'] = FLAGS.max_seq_length
        parsed['bert_config'] = bert_config.to_dict()

        with tf.gfile.GFile('%s/config.json' % OUTPUT_DIR, 'w') as f:
            json.dump(parsed, f)
        parsed['bert_config'] = bert_config

        create_model = None
        config_class = None

        if parsed['model'] == 'cnn':
            config_class = CNNConfig
            create_model = create_cnn_model

        elif parsed['model'] == 'contextualized_cnn':
            config_class = ContextualizedCNNConfig
            create_model = create_contextualized_cnn_model

        else:
            raise ValueError('Unsupported model %s' % parsed['model'])

        return (config_class(**parsed), create_model)


def main(_):
    tf.gfile.MakeDirs(OUTPUT_DIR)

    tf.logging.set_verbosity(tf.logging.INFO)

    (config, create_model) = config_and_model(FLAGS.config)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_config = tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop,
                                          num_shards=FLAGS.num_tpu_cores,
                                          per_host_input_for_training=is_per_host)
    run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver,
                                          master=FLAGS.master,
                                          log_step_count_steps=1,
                                          save_summary_steps=2,
                                          model_dir=OUTPUT_DIR,
                                          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                          keep_checkpoint_max=2,
                                          tpu_config=tpu_config)

    if FLAGS.do_train:
        num_train_examples = N_TRAIN_EXAMPLES
        if num_train_examples is None:
            num_train_examples = math.ceil(
                N_TOTAL_SQUAD_EXAMPLES * (1. - FLAGS.eval_percent))
        num_train_steps = int(num_train_examples /
                              FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        print("Total training steps = %d" % num_train_steps)
        time.sleep(1)

        model_fn = model_fn_builder(bert_config=bert_config,
                                    init_checkpoint=INIT_CHECKPOINT,
                                    learning_rate=FLAGS.learning_rate,
                                    num_train_steps=num_train_steps,
                                    num_warmup_steps=num_warmup_steps,
                                    config=config,
                                    use_tpu=FLAGS.use_tpu,
                                    create_model_fn=create_model,
                                    fine_tune=FLAGS.fine_tune)

        estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu,
                                                model_fn=model_fn,
                                                config=run_config,
                                                train_batch_size=FLAGS.train_batch_size,
                                                eval_batch_size=FLAGS.eval_batch_size)

        train_input_fn = input_fn_builder(input_file=TRAIN_FILE_NAME,
                                          seq_length=FLAGS.max_seq_length,
                                          is_training=True,
                                          bert_config=bert_config,
                                          drop_remainder=True,
                                          fine_tune=FLAGS.fine_tune)
        eval_input_fn = input_fn_builder(
            input_file=EVAL_FILE_NAME,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            bert_config=bert_config,
            drop_remainder=True,
            fine_tune=FLAGS.fine_tune)

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            throttle_secs=FLAGS.eval_throttle_secs,
            steps=FLAGS.eval_steps,
        )

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.app.run()
