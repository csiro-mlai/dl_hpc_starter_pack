from argparse import Namespace
import os
from pytest import fixture
from dlhpcstarter.main import submit
from dlhpcstarter.utils import importer, load_config_and_update_args

test_dir = os.path.dirname(__file__)

@fixture(scope="module", autouse=True, params=[test_dir])
def in_settings_folder(request):
    prevdir = os.getcwd()
    os.chdir(request.param)
    yield
    os.chdir(prevdir)


def test_sinusoid():
    
    args = Namespace(
        task='cosine',
        config='default',
        submit=False,
        work_dir=None,
        num_workers=None,
        num_nodes=None,
        trial=None,
        debug=True,
    )

    stages_fnc = importer(definition='stages', module='.'.join(['task', args.task, 'stages']))
    load_config_and_update_args(args=args, print_args=True)
    submit(args=args, stages_fnc=stages_fnc)