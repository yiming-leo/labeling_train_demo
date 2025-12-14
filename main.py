from pathlib import Path

import demo_code


# get keys from ./keys/.api_keys
def load_keys_from_env_file():
    keys = {}
    keys_dir = Path(__file__).parent / "keys"
    env_file = keys_dir / ".api_keys"

    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                keys[key.strip()] = value.strip()
    return keys


def main():
    api_keys = load_keys_from_env_file()
    qwen_key = api_keys.get('QWEN_API_KEY')

    demo_code.label_them(qwen_key)


if __name__ == "__main__":
    main()
