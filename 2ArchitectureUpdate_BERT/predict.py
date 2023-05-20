import tensorflow as tf
import os
import pickle
import json
from augment.utils import (input_fn_builder)
from train import model_fn_builder, FLAGS
from models.cnn import CNNConfig, create_cnn_model
from models.contextualized_cnn import create_contextualized_cnn_model, ContextualizedCNNConfig
from augment.predict import (FLAGS, bert_config, OUTPUT_DIR, DEV_FILENAME, write_predictions, RawResult,
                             model_fn_builder, INIT_CHECKPOINT)


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


(config, create_model) = config_and_model(
    '%s/config.json' % (FLAGS.output_dir))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_config = tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop,
                                          num_shards=FLAGS.num_tpu_cores,
                                          per_host_input_for_training=is_per_host)

    model_fn = model_fn_builder(bert_config=bert_config,
                                init_checkpoint=INIT_CHECKPOINT,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=1,
                                num_warmup_steps=0,
                                config=config,
                                use_tpu=FLAGS.use_tpu,
                                create_model_fn=create_model,
                                fine_tune=FLAGS.fine_tune)

    run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver,
                                          master=FLAGS.master,
                                          log_step_count_steps=1,
                                          save_summary_steps=2,
                                          model_dir=FLAGS.output_dir,
                                          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                          keep_checkpoint_max=2,
                                          tpu_config=tpu_config)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu,
                                            model_fn=model_fn,
                                            config=run_config,
                                            predict_batch_size=FLAGS.predict_batch_size)

    suffix = ''
    if FLAGS.fine_tune:
        suffix = '_fine_tune'

    eval_examples = None
    with tf.gfile.GFile('%s/dev_examples%s.pickle' % (FLAGS.features_dir, suffix),
                        'rb') as out_file:
        eval_examples = pickle.load(out_file)
    eval_features = None
    with tf.gfile.GFile('%s/dev_features%s.pickle' % (FLAGS.features_dir, suffix),
                        'rb') as out_file:
        eval_features = pickle.load(out_file)

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = input_fn_builder(input_file=DEV_FILENAME,
                                        seq_length=FLAGS.max_seq_length,
                                        bert_config=bert_config,
                                        is_training=False,
                                        drop_remainder=False,
                                        fine_tune=FLAGS.fine_tune)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(predict_input_fn, yield_single_examples=False):
        if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))

        if hasattr(result["unique_ids"], 'shape'):
            for i, unique_id_s in enumerate(result['unique_ids']):
                unique_id = int(unique_id_s)
                start_logits = [float(x)
                                for x in result["start_logits"][i].flat]
                end_logits = [float(x) for x in result["end_logits"][i].flat]
                all_results.append(
                    RawResult(unique_id=unique_id, start_logits=start_logits,
                              end_logits=end_logits))
        else:
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

    output_prediction_file = os.path.join(FLAGS.output_dir, FLAGS.predictions_output_directory,
                                          "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, FLAGS.predictions_output_directory,
                                     "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, FLAGS.predictions_output_directory,
                                             "null_odds.json")

    write_predictions(eval_examples, eval_features, all_results, FLAGS.n_best_size,
                      FLAGS.max_answer_length, FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)


if __name__ == "__main__":
    tf.app.run()
