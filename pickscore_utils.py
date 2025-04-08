# Reference implementation of PickScore from the official repo:
# https://github.com/yuvalkirstain/PickScore

# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

# load model
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


class Selector():
    
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    def score(self, images, prompt, softmax=False):

        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores =  (text_embs @ image_embs.T)[0]

            if softmax:
                scores = self.model.logit_scale.exp() * scores
                # get probabilities if you have multiple images to choose from
                probs = torch.softmax(scores, dim=-1)
                return probs.cpu().tolist()
            else:
                return scores.cpu().tolist()

if __name__ == '__main__':
    # 💡 first download the model from huggingface
    # export HF_ENDPOINT="https://hf-mirror.com"
    # export HF_HOME="/ifs/root/ipa01/110/user_110003/download"

    
    pil_images = [Image.open("/ifs/root/ipa01/110/user_110003/code/DiffusionDPO/jpg_0.png"), Image.open("/ifs/root/ipa01/110/user_110003/code/DiffusionDPO/jpg_1.png")]
    prompt = "An anime girl, masterpiece, good line art, trending in pixiv"

    selector = Selector('cpu')
    scores = selector.score(pil_images, prompt)
    print(scores)