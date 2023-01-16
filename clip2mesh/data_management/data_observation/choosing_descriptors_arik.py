from clip2mesh.utils import Utils
from clip2mesh.data_management.data_observation.choosing_descriptors import ChoosingDescriptors


class ChoosingDescriptorsArik(ChoosingDescriptors):
    def __init__(self, args):
        super().__init__()
        self.utils = Utils()

    pass
