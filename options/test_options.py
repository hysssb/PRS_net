from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load cached model')
        self.parser.add_argument('--num_aug', type=int, default=1, help='# of augmentation files')
        self.parser.add_argument('--maxlsd', type=float, default=0.0001, help='be removed if gt maxlsd')
        self.parser.add_argument('--maxcosval', type=float, default=0.866,
                                 help='max cos dihedral angle, gt will be removed worse one')
        self.parser.add_argument('--accGTE', default=False, help='save lsd auto, set True to save GTE accuracy')
        self.is_train = False


