import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    config_path = "GiuliaAI/config/preprocess.yaml" 

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
