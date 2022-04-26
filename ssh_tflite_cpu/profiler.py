import os
from nn_meter.builder.backends.interface import BaseProfiler


class SSHTFLiteProfiler(BaseProfiler):
    use_gpu = None

    def __init__(self, dst_kernel_path, benchmark_model_path, dst_graph_path='', ssh_cli=None, num_threads=1, num_runs=50, warm_ups=10):
        """
        @params:
        graph_path: graph file. path on host server
        dst_graph_path: graph file. path on android device
        kernel_path: dest kernel output file. path on android device
        benchmark_model_path: path to benchmark_model on android device
        """
        self._dst_graph_path = dst_graph_path
        self._dst_kernel_path = dst_kernel_path
        self._benchmark_model_path = benchmark_model_path
        self._num_threads = num_threads
        self._num_runs = num_runs
        self._warm_ups = warm_ups

        self.cli = ssh_cli
        if not self.cli:
            raise RuntimeError("SSHTFLiteProfiler: SSH client is None")


    def profile(self, graph_path, preserve=False, clean=True, taskset_args=None):
        """
        @params:
        preserve: tflite file exists in remote dir. No need to push it again.
        clean: remove tflite file after running.
        """
        model_name = os.path.basename(graph_path)
        remote_graph_path = os.path.join(self._dst_graph_path, model_name)
        
        print(f"Profiling {model_name} ... ")
        taskset_cmd = f'taskset {taskset_args}' if taskset_args else '' 

        if not preserve:
            # device.push(graph_path, remote_graph_path)
            sftp_client = self.cli.open_sftp()
            sftp_client.put(graph_path, remote_graph_path)
            sftp_client.close()
        try:
            kernel_cmd = f'--kernel_path={self._dst_kernel_path}' if self._dst_kernel_path else ''
            cmd = f' {taskset_cmd} {self._benchmark_model_path} {kernel_cmd}' \
                f' --num_threads={self._num_threads}' \
                f' --num_runs={self._num_runs}' \
                f' --warmup_runs={self._warm_ups}' \
                f' --graph={remote_graph_path}' \
                f' --enable_op_profiling=true' \
                f' --use_gpu={"true" if self.use_gpu else "false"}'
                
            # print(f"Executing command on target SSH: {cmd}")
            stdin, stdout,stderr = self.cli.exec_command(cmd)
            res = ''.join(stdout.readlines())
        except:
            raise
        finally:
            if clean:
                stdin, stdout, stderr = self.cli.exec_command(f"rm {remote_graph_path}")
        
        # print("SSH Intermediate Result: \n" + res)
        # print("<< END SSH OUTPUT")
        return res
