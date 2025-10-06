import yaml

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def main():
        config = load_config('config.yaml')
        print(config)

if __name__ == "__main__":
    main()