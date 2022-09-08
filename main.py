import os.path, pkgutil
import dlhpcstarter
pkgpath = os.path.dirname(dlhpcstarter.__file__)


print([name for _, name, _ in pkgutil.iter_modules([pkgpath])])

# from dlhpcstarter import main
#
# main()