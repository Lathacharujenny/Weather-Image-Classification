from src.components.data_splitting import DataSplitting
from src.config.configuration import ConfigurationManager


class DataSplittingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_splitting_config = config.get_data_splitting_config()
        data_splitting = DataSplitting(config=data_splitting_config)
        data_splitting.load_data()
        data_splitting.data_splitting()