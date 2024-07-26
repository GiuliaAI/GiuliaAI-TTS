import yaml

from preprocessor import giuliaai


def main(config):
    if "GiuliaAI" in config["dataset"]:
        giuliaai.prepare_align(config)

if __name__ == "__main__":
    config_path = "GiuliaAI/config/preprocess.yaml" 

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    main(config)