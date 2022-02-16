import hydra

@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg: dict):
    model_config = cfg.configs.model
    training_config = cfg.configs.training

    print(f"{model_config = }")
    print(f"{training_config = }")


if __name__ == "__main__":
    main()
