from argparse import Namespace
from subprocess import call
from typing import Callable, Optional
import datetime
import os
import signal
import sys
import traceback


class ClusterSubmit(object):
    """
    Submits a job to a cluster manager.
    """
    def __init__(
            self,
            fnc: Callable,
            fnc_kwargs: Namespace,
            save_dir: str,
            log_err: bool = True,
            log_out: bool = True,
            manager: str = 'slurm',
            time_limit: str = '02:00:00',
            begin: str = 'now',
            num_gpus: int = 0,
            num_nodes: int = 1,
            num_workers: int = 1,
            memory: str = '16GB',
            no_srun: bool = False,
            python_cmd: str = 'python3',
            entrypoint: Optional[str] = None,
            resubmit: bool = True,
    ):
        """
        Argument/s:
            fnc - function for the job.
            fnc_kwargs - keyword arguments for the function.
            save_dir -  directory where the cluster manager script, stdout, and stderr are saved.
            log_err - log standard error.
            log_out - log standard output.
            manager - 'slurm' (needs to be modified for other cluster managers, e.g., PBS).
            time_limit - time limit for job.
            begin - when to begin the job.
            num_gpus - number of gpus.
            num_nodes - number of nodes.
            num_workers - number of workers per GPU.
            memory - minimum memory amount.
            no_srun - don't use 'srun'.
            python_cmd - python command name.
            entrypoint - entrypoint to use instead of a script.
            resubmit - automatically resubmit job just before timout.
        """
        self.fnc = fnc
        self.fnc_kwargs = fnc_kwargs
        self.exp_dir = save_dir
        self.log_err = log_err
        self.log_out = log_out
        self.manager = manager
        self.time_limit = time_limit
        self.begin = begin
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.memory = memory
        self.no_srun = no_srun
        self.python_cmd = python_cmd
        self.entrypoint = entrypoint
        self.resubmit = resubmit

        self.script_name = os.path.realpath(sys.argv[0])
        self.manager_commands = []
        self.commands = []

        self.run_cmd = {
            'slurm': 'sbatch',
        }

        self.is_from_manager_object = bool(vars(fnc_kwargs)["slurm_cmd_path"])

    def add_manager_cmd(self, cmd=None, value=None):
        self.manager_commands.append((cmd, value))

    def add_command(self, cmd):
        self.commands.append(cmd)

    def notify_job_status(self, email, on_done, on_fail):
        self.email = email
        self.notify_on_end = on_done
        self.notify_on_fail = on_fail

    def submit(self, job_display_name=None):
        self.job_display_name = job_display_name
        self.log_dir()
        if self.is_from_manager_object:
            self.run()
        else:
            scripts_path = os.path.join(self.exp_dir, 'manager_scripts')
            self.schedule_experiment(self.get_max_session_number(scripts_path))

    def run(self):
        if self.resubmit:
            print('Setting signal to automatically requeue the job before timeout.')
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)
        else:
            print("Automatic requeuing has not been set. The job will not be requeued after timeout.")
        try:
            self.fnc(self.fnc_kwargs)

        except Exception as e:
            print('Caught exception in worker thread', e)
            traceback.print_exc()
            raise SystemExit

    def schedule_experiment(self, session):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamp = 'session_{}_{}'.format(session, timestamp)

        # Generate and save cluster manager script
        manager_cmd_script_path = os.path.join(self.manager_files_log_path, '{}.sh'.format(timestamp))
        if self.manager == 'slurm':
            manager_cmd = self.build_slurm_command(manager_cmd_script_path, timestamp, session)
        else:
            raise ValueError(f'{self.manager} is not a valid manager.')
        self.save_manager_cmd(manager_cmd, manager_cmd_script_path)

        # Run script to launch job
        print('\nLaunching experiment...')
        result = call('{} {}'.format(self.run_cmd[self.manager], manager_cmd_script_path), shell=True)
        if result == 0:
            print(f'Launched experiment {manager_cmd_script_path}.')
        else:
            print('Launch failed...')

    def call_resume(self):
        job_id = os.environ['SLURM_JOB_ID']
        cmd = 'scontrol requeue {}'.format(job_id)
        print(f'\nRequeing job {job_id}...')
        result = call(cmd, shell=True)
        if result == 0:
            print(f'Requeued job {job_id}.')
        else:
            print('Requeue failed...')
        os._exit(0)

    def sig_handler(self, signum, frame):
        print(f"Caught signal: {signum}")
        self.call_resume()

    def term_handler(self, signum, frame):
        print("Bypassing sigterm.")

    def save_manager_cmd(self, manager_cmd, manager_cmd_script_path):
        with open(manager_cmd_script_path, mode='w') as file:
            file.write(manager_cmd)

    def get_max_session_number(self, path):
        files = os.listdir(path)
        session_files = [f for f in files if 'session_' in f]
        if len(session_files) > 0:
            sessions = [int(f_name.split('_')[1]) for f_name in session_files]
            max_session = max(sessions)
            return max_session + 1
        else:
            return 0

    def log_dir(self):
        out_path = os.path.join(self.exp_dir)
        self.exp_dir = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.manager_files_log_path = os.path.join(out_path, 'manager_scripts')
        if not os.path.exists(self.manager_files_log_path):
            os.makedirs(self.manager_files_log_path)
        if self.log_err:
            err_path = os.path.join(out_path, 'error_logs')
            if not os.path.exists(err_path):
                os.makedirs(err_path)
            self.err_log_path = err_path
        if self.log_out:
            out_path = os.path.join(out_path, 'out_logs')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            self.out_log_path = out_path

    def build_slurm_command(self, manager_cmd_script_path, timestamp, session):

        sub_commands = ['#!/bin/bash -l']
        sub_commands.append('#SBATCH --job-name={}'.format('{}session_{}'.format(self.job_display_name, session)))

        if self.log_out:
            out_path = os.path.join(self.out_log_path, '{}_%j.out'.format(timestamp))
            sub_commands.append('#SBATCH --output={}'.format(out_path))

        if self.log_err:
            err_path = os.path.join(self.err_log_path, '{}_%j.err'.format(timestamp))
            sub_commands.append('#SBATCH --error={}'.format(err_path))
        sub_commands.append(f'#SBATCH --time={self.time_limit:s}')

        if self.begin != "now":
            sub_commands.append('#SBATCH --begin={}'.format(self.begin))

        if self.num_gpus:
            sub_commands.append('#SBATCH --gres=gpu:{}'.format(self.num_gpus))

        if self.num_workers > 0:
            sub_commands.append('#SBATCH --cpus-per-task={}'.format(self.num_workers))

        sub_commands.append('#SBATCH --nodes={}'.format(self.num_nodes))
        sub_commands.append('#SBATCH --mem={}'.format(self.memory))
        sub_commands.append(f'#SBATCH --signal=USR1@{6 * 60}')

        mail_type = []
        if self.notify_on_end:
            mail_type.append('END')
        if self.notify_on_fail:
            mail_type.append('FAIL')
        if len(mail_type) > 0 and self.email is not None:
            sub_commands.append('#SBATCH --mail-type={}'.format(','.join(mail_type)))
            sub_commands.append('#SBATCH --mail-user={}'.format(self.email))

        for (cmd, value) in self.manager_commands:
            if value:
                sub_commands.append('#SBATCH --{}={}'.format(cmd, value))
            else:
                sub_commands.append('#SBATCH --{}'.format(cmd))

        sub_commands = [x.lstrip() for x in sub_commands]

        for cmd in self.commands:
            sub_commands.append(cmd)

        args_string = self.args_to_string(self.fnc_kwargs)
        args_string = '{} --{} {}'.format(args_string, "slurm_cmd_path", manager_cmd_script_path)

        if self.entrypoint:
            cmd = f'{self.entrypoint} {args_string}'
        else:
            cmd = '{} {} {}'.format(self.python_cmd, self.script_name, args_string)
        if not self.no_srun:
            cmd = 'srun ' + cmd
        sub_commands.append(cmd)

        return '\n'.join(sub_commands)

    def args_to_string(self, args):
        params = []
        for k, v in vars(args).items():
            if v is not None:
                if self.escape(v):
                    cmd = '--{} \"{}\"'.format(k, v)
                else:
                    cmd = '--{} {}'.format(k, v)
                params.append(cmd)
        return ' '.join(params)

    def escape(self, v):
        v = str(v)
        return '[' in v or ';' in v or ' ' in v
    