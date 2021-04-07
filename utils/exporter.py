import tensorflow as tf
import types


def exporter_builder(models, flags):
    """
    build a exporter
    :param models: models that need to exported
    :param flags: FLAGS
    """
    exporters = []
    for name, m in models.items():
        exporters.append(exporter_wrapper(get_exporter(name, m, flags), name))
    return exporters


def exporter_wrapper(exporter, name):
    def export_wrapper(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        estimator._params['export_model'] = name
        self.export_orig(estimator, export_path, checkpoint_path, eval_result, is_the_final_export)

    setattr(exporter, 'export_orig', getattr(exporter, 'export'))
    setattr(exporter, 'export', types.MethodType(export_wrapper, exporter))
    return exporter


def get_exporter(name, model, flags):
    if flags.estimator_exporter_type == "BestExporter":
        return tf.estimator.BestExporter(
            name=name,
            serving_input_receiver_fn=model.serving_input_function,
            exports_to_keep=flags.exports_to_keep)
    else:
        return tf.estimator.LatestExporter(
            name=name,
            serving_input_receiver_fn=model.serving_input_function,
            exports_to_keep=flags.exports_to_keep)