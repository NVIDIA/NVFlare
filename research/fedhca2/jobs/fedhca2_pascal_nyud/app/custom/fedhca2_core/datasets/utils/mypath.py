import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]


class MyPath(object):
    """
    User-specific path configuration.
    """

    @staticmethod
    def db_root_dir(database=''):
        # Use absolute path to avoid issues with NVFLARE workspace copying
        db_root = '/home/suizhi/NVFlare/research/fedhca2/data'

        db_names = {'PASCALContext', 'NYUDv2'}

        if database in db_names:
            return os.path.join(db_root, database)

        elif not database:
            return db_root

        else:
            raise NotImplementedError
