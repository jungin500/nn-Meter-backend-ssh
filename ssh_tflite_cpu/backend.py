import logging
from nn_meter.builder.backends import BaseBackend
from nn_meter.builder.backends.tflite.cpu import TFLiteCPULatencyParser
from profiler import SSHTFLiteProfiler
from paramiko import SSHClient, AutoAddPolicy
from key_manager import parse_keyfile
from nn_meter.utils.path import get_filename_without_ext
import os
import tensorflow as tf

class SSHTFLiteBackend(BaseBackend):
    parser_class = TFLiteCPULatencyParser
    profiler_class = SSHTFLiteProfiler
    
    def __init__(self, configs):
        super(SSHTFLiteBackend, self).__init__(configs)
        
    def update_configs(self):
        """update the config parameters for TFLite platform
        """
        super().update_configs()
        
        self.cli = SSHClient()
        self.cli.set_missing_host_key_policy(AutoAddPolicy)
        
        pkey = parse_keyfile(
            filename=self.configs['SSH_PRIVATE_KEY_PATH'],
            password=self.configs['SSH_PRIVATE_KEY_PASSWORD']
        )
        if not pkey:
            raise RuntimeError("Invalid key or password: " + self.configs['SSH_PRIVATE_KEY_PATH'])
        
        self.cli.connect(
            hostname=self.configs['SSH_REMOTE_ADDR'],
            port=self.configs['SSH_REMOTE_PORT'],
            username=self.configs['SSH_REMOTE_USER'],
            pkey=pkey
        )
        
        # Create required path "REMOTE_MODEL_DIR"
        self.cli.exec_command(f'mkdir -p {self.configs["REMOTE_MODEL_DIR"]}')

        self.profiler_kwargs.update({
            'num_threads': self.configs['NUM_THREADS'],
            'dst_kernel_path': self.configs['REMOTE_KERNEL_PATH'],
            'benchmark_model_path': self.configs['REMOTE_BENCHMARK_MODEL_PATH'],
            'dst_graph_path': self.configs['REMOTE_MODEL_DIR'],
            'ssh_cli': self.cli,
        })

    def profile(self, converted_model, metrics = ['latency'], input_shape = None):
        max_cpuid = int(self.configs['NUM_THREADS']) - 1
        profiler_result = self.profiler.profile(
            graph_path=converted_model,
            preserve=False, clean=True, taskset_args=f'--cpu-list 0-{max_cpuid}'
        )
        parsed_results = self.parser.parse(profiler_result)
        result = parsed_results.results.get(metrics)
        print("Result metric:", result['latency'])
        return result

    def convert_model(self, model_path, save_path, input_shape=None):
        """convert the Keras model instance to ``.tflite`` and return model path
        """
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        converted_model = os.path.join(save_path, model_name + '.tflite')
        open(converted_model, 'wb').write(tflite_model)
        return converted_model

    def test_connection(self):
        """check the status of backend interface connection, ideally including open/close/check_healthy...
        """
        # from ppadb.client import Client as AdbClient
        # client = AdbClient(host="127.0.0.1", port=5037)
        stdin, stdout, stderr = self.cli.exec_command('echo Hello World!')
        print(''.join(stdout.readlines()))
            