"""
Pick-a-Pic Dataset Loader Script (2025.4.13 by user 110003)

Description:
    This script loads the Pick-a-Pic v2 dataset from Hugging Face,
    demonstrates basic operations, and saves a subset for quick testing.

Might set if you want to download a new version of the dataset:
    - export HF_ENDPOINT=https://hf-mirror.com
    - export HF_HOME="your local cache directory" if you dont't specify the `cache_dir`
"""


from datasets import load_dataset



def load_pickapic_dataset(cache_dir=None):
    try:
        print("⏳ Loading dataset...")
        dataset = load_dataset(
            "yuvalkirstain/pickapic_v2",
            cache_dir=cache_dir
        )
        print("✅ Dataset loaded successfully!")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise


def info(dataset):
    print("\n📊 Dataset Structure:")
    print(f"- Number of splits: {len(dataset)}")
    print(f"- Train samples: {len(dataset['train'])}")
    print(f"- Validation samples: {len(dataset['validation'])}")
    
    print("\n📝 Sample columns:")
    print(dataset['train'].column_names)



if __name__ == "__main__":
    DATA_CACHE_DIR="/ifs/root/ipa01/110/user_110003/picapic_v2"
    dataset = load_pickapic_dataset(cache_dir=DATA_CACHE_DIR)
    info(dataset)
