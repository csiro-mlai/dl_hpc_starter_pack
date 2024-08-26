import datetime
import math
import os
import re
import signal
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from subprocess import call
from typing import Any, Callable, Optional

import yaml
from munch import DefaultMunch
from omegaconf import OmegaConf


class ClusterSubmit(object):
    """
    Submits a job to a cluster manager.
    """
    def __init__(
            self,
            fnc: Callable,
            args: Namespace,
            save_dir: str,
            cmd_line_args: Any = {},
            log_err: bool = True,
            log_out: bool = True,
            manager: str = 'slurm',
            time_limit: Optional[str] = None,
            begin: Optional[str] = None,
            num_gpus: Optional[int] = None,
            num_nodes: Optional[int] = None,
            num_workers: Optional[int] = None,
            memory: Optional[str] = None,
            no_srun: bool = False,
            no_cpus_per_task: bool = False,
            no_gpus_per_node: bool = False,
            no_ntasks_per_node: bool = False,
            python_cmd: str = 'python',
            entrypoint: Optional[str] = None,
            srun_options: str = '',
            email: Optional[str] = None,
            email_on_complete: bool = True,
            email_on_fail: bool = True,
            auto_resubmit: bool = True,
            auto_resubmit_method: str = 'signal',
    ):
        """
        Argument/s:
            fnc - function for the job.
            args - arguments for the job.
            cmd_line_args - command line arguments for the cluster manager script.
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
            no_cpus_per_task - prevent the --cpus-per-task option from being placed in the Slurm script. 
            no_gpus_per_node - prevent the --gpus-per-node option from being placed in the Slurm script. 
            no_ntasks_per_node - prevent the --ntasks-per-node option from being placed in the Slurm script. 
            python_cmd - python command name.
            entrypoint - entrypoint to use instead of a script.
            srun_options - options for srun.
            email - email address for job notifications.
            email_on_complete - send notification via email on job completion.
            email_on_fail - send notification via email on job fail.
            auto_resubmit - flag for automatically resubmitting to the cluster.
            auto_resubmit_method - method of automatically resubmitting job to queue.
        """
        self.fnc = fnc
        self.args = args
        self.cmd_line_args = cmd_line_args if isinstance(cmd_line_args, dict) else vars(cmd_line_args)
        self.save_dir = save_dir
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
        self.no_cpus_per_task = no_cpus_per_task
        self.no_gpus_per_node = no_gpus_per_node
        self.no_ntasks_per_node = no_ntasks_per_node
        self.python_cmd = python_cmd
        self.entrypoint = entrypoint
        self.srun_options = srun_options
        self.email = email
        self.email_on_complete = email_on_complete
        self.email_on_fail = email_on_fail
        self.auto_resubmit = auto_resubmit
        self.auto_resubmit_method = auto_resubmit_method

        self.auto_resubmit_method = self.auto_resubmit_method if self.auto_resubmit else None 

        self.script_name = os.path.realpath(sys.argv[0])
        self.manager_options = []
        self.commands = []
        self.clean_up_commands = []

        self.run_cmd = {'slurm': 'sbatch'}

        self.manager_script_path = self.args.manager_script_path
        self.is_from_manager_object = bool(self.manager_script_path)

    def add_manager_option(self, option=None, value=None):
        self.manager_options.append((option, value))

    def add_command(self, cmd):
        self.commands.append(cmd)

    def add_clean_up_command(self, cmd):
        self.clean_up_commands.append(cmd)

    def submit(self, job_display_name=None):
        self.job_display_name = job_display_name

        manager_script_dir = os.path.join(self.save_dir, 'manager_scripts')
        args_dir = os.path.join(self.save_dir, 'arguments')

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(manager_script_dir).mkdir(parents=True, exist_ok=True)
        Path(args_dir).mkdir(parents=True, exist_ok=True)
        if self.log_err:
            self.err_log_path = os.path.join(self.save_dir, 'error_logs')
            Path(self.err_log_path).mkdir(parents=True, exist_ok=True)
        if self.log_out:
            self.out_log_path = os.path.join(self.save_dir, 'out_logs')
            Path(self.out_log_path).mkdir(parents=True, exist_ok=True)

        if self.is_from_manager_object:

            try:
                if self.auto_resubmit_method == 'signal':
                    print('Setting signal to automatically requeue the job before timeout.')
                    signal.signal(signal.SIGUSR1, self.sig_handler)
                    # signal.signal(signal.SIGALRM, self.sig_handler)  # For one epoch only jobs.
                    signal.signal(signal.SIGTERM, self.term_handler)

                # Load arguments for session:
                session = re.search(r'\/session_(\d+)', self.manager_script_path).group(1)
                args_path = os.path.join(self.save_dir, 'arguments', f'session_{session}.yaml')
                with open(args_path, 'r') as f:
                    args = yaml.safe_load(f)
                    args = DefaultMunch.fromDict(args)
                
                self.fnc(args)

            except Exception as e:
                print('Caught exception in worker thread', e)
                traceback.print_exc()
                raise SystemExit

        else:
            scripts_path = os.path.join(self.save_dir, 'manager_scripts')

            # Get max session number:
            files = os.listdir(scripts_path)
            session_files = [f for f in files if 'session_' in f]
            session = 0
            if len(session_files) > 0:
                sessions = [int(f_name.split('_')[1]) for f_name in session_files]
                max_session = max(sessions)
                session = max_session + 1

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            timestamp = f'session_{session}_{timestamp}'

            # Generate and save cluster manager script:
            manager_script_path = os.path.join(manager_script_dir, f'{timestamp}.sh')
            if self.manager == 'slurm':
                self.create_slurm_script(manager_script_path, timestamp, session)
            else:
                raise ValueError(f'{self.manager} is not a valid manager.')

            # Save arguments for session:
            args_path = os.path.join(args_dir, f'session_{session}.yaml')
            with open(args_path, 'w') as f:
                OmegaConf.save(vars(self.args), f)

            # Run script to launch job
            print('\nLaunching experiment...')
            result = call(f'{self.run_cmd[self.manager]} {manager_script_path}', shell=True)
            if result == 0:
                print(f'Launched experiment {manager_script_path}.')
            else:
                print('Launch failed...')

    @staticmethod
    def sig_handler(signum, frame):

        print(f'Caught signal: {signum}')

        job_id = os.environ['SLURM_JOB_ID']
        cmd = f'scontrol requeue {job_id}'

        print(f'\nRequeing job {job_id}...')
        result = call(cmd, shell=True)
        
        if result == 0:
            print(f'Requeued job {job_id}.')
        else:
            print('Requeue failed...')
        # os._exit(result)
        sys.exit(result)

    def term_handler(self, signum, frame):
        print('Bypassing sigterm.')

    def create_slurm_script(self, manager_script_path, timestamp, session):

        script = ['#!/bin/bash -l']
        script.append(f'#SBATCH --job-name={self.job_display_name}session_{session}')

        if self.log_out:
            out_path = os.path.join(self.out_log_path, f'{timestamp}_%j.out')
            script.append(f'#SBATCH --output={out_path}')

        if self.log_err:
            err_path = os.path.join(self.err_log_path, f'{timestamp}_%j.err')
            script.append(f'#SBATCH --error={err_path}')

        if self.time_limit:
            script.append(f'#SBATCH --time={self.time_limit:s}')

        if self.begin:
            script.append(f'#SBATCH --begin={self.begin}')

        if not self.no_gpus_per_node and self.num_gpus:
            script.append(f'#SBATCH --gpus-per-node={self.num_gpus}')

        if not self.no_ntasks_per_node and self.num_gpus:
            script.append(f'#SBATCH --ntasks-per-node={self.num_gpus}')

        if not self.no_cpus_per_task and self.num_workers > 1:
            script.append(f'#SBATCH --cpus-per-task={self.num_workers}')

        if self.num_nodes:
            script.append(f'#SBATCH --nodes={self.num_nodes}')

        if self.memory:
            script.append(f'#SBATCH --mem={self.memory}')

        if self.auto_resubmit_method == 'signal':
            script.append(f'#SBATCH --signal=USR1@{2 * 60}')
            
        script.append(f'#SBATCH --open-mode=append')

        mail_type = []
        if self.email_on_complete:
            mail_type.append('END')
        if self.email_on_fail:
            mail_type.append('FAIL')
        if len(mail_type) > 0 and self.email is not None:
            script.append(f'#SBATCH --mail-type={",".join(mail_type)}')
            script.append(f'#SBATCH --mail-user={self.email}')

        for (option, value) in self.manager_options:
            if value:
                script.append(f'#SBATCH --{option}={value}')
            else:
                script.append(f'#SBATCH --{option}')

        script = [x.lstrip() for x in script]

        for cmd in self.commands:
            script.append(cmd)


        cmd_line_args = []
        cmd_line_args.append(f'--task {self.cmd_line_args["task"]}')
        cmd_line_args.append(f'--config {self.cmd_line_args["config"]}')
        cmd_line_args.append(f'--trial {self.cmd_line_args["trial"]}')
        cmd_line_args.append('--submit')
        cmd_line_args = ' '.join(cmd_line_args)

        cmd_line_args = f'{cmd_line_args} --manager_script_path {manager_script_path}'

        if self.entrypoint:
            cmd = f'{self.entrypoint} {cmd_line_args}'
        else:
            cmd = f'{self.python_cmd} {self.script_name} {cmd_line_args}'
        if not self.no_srun:
            cmd = self.srun_options + ' ' + cmd if self.srun_options else cmd
            cmd = 'srun ' + cmd

        if self.auto_resubmit_method == 'timeout':
            
            self.time_limit = '1-00:00:00' if self.time_limit == '24:00:00' else self.time_limit
            
            if self.time_limit.count(':') == 1:
                format = '%M:%S'
            elif self.time_limit.count(':') == 2:
                format = '%H:%M:%S'               
            if '-' in self.time_limit:
                format = '%d-' + format
            t = datetime.datetime.strptime(self.time_limit, format)
            
            if '-' in self.time_limit:
                delta = datetime.timedelta(days=t.day, hours=t.hour, minutes=t.minute, seconds=t.second)
            else:
                delta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
            
            minutes = math.floor((delta.total_seconds() / 60) - 2)
            cmd = f'timeout {minutes}m ' + cmd

        script.append(cmd)

        if self.auto_resubmit_method == 'timeout':
            script.append('if [[ $? -eq 124 ]]; then')
            script.append('    echo Job incomplete, submitting again...')
            script.append(f'    sbatch {manager_script_path}')
            script.append(f'    scontrol update jobid=$SLURM_JOB_ID mailtype=NONE')
            script.append('fi')

        for cmd in self.clean_up_commands:
            script.append(cmd)

        with open(manager_script_path, mode='w') as f:
            f.write('\n'.join(script))
    