import numpy as np
import bpu_infer_lib
import cv2
import json
import os


class ModelInference:
    def __init__(self, model_path, labels_path, img_path):
        self.model_path = (model_path)
        self.labels_path = (labels_path)
        self.img_path = (img_path)
        self.infer_obj = bpu_infer_lib.Infer(False)
        self.load_model()
        self.labels = self.load_labels()

    def load_model(self):
        """Load the model file"""
        try:
            self.infer_obj.load_model(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def load_labels(self):
        """Load ImageNet labels"""
        try:
            with open(self.labels_path, 'r', encoding='utf-8') as json_file:
                return json.load(json_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load labels: {e}")

    def preprocess_image(self):
        """Load and preprocess the image"""
        try:
            return self.infer_obj.read_img_to_nv12(self.img_path, 0)
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image: {e}")

    def inference(self):
        """Perform model inference and get results"""
        try:
            self.infer_obj.forward()
            ptr1 = self.infer_obj.get_infer_res(0, 1000)  # Number of classes
            array1 = [ptr1[i] for i in range(1000)]
            return np.array(array1).astype(np.float32)  # Unnormalized confidence scores
        except Exception as e:
            raise RuntimeError(f"Failed during inference: {e}")

    @staticmethod
    def np_softmax(arr):
        """Compute softmax"""
        exp_out = np.exp(arr)
        sum_exp_out = np.sum(exp_out, axis=-1, keepdims=True)
        return exp_out / sum_exp_out

    def postprocess(self, logits):
        """Process the output using softmax and select the class with the highest probability"""
        probs = self.np_softmax(logits)
        max_index = np.argmax(probs)
        return max_index, probs
    
    def postprocess_tf(self, logits):
        """Process the output using softmax and select the class with the highest probability"""
        prob = np.squeeze(logits)
        idx = np.argsort(-prob)
        top_five_label_probs = [(idx[i], prob[idx[i]]) for i in range(1)]
        max_index = top_five_label_probs[0][0]
        probs = top_five_label_probs[0][1]
        return max_index, probs

    def annotate_image(self, max_ind, probs):
        """Annotate the inference result on the image"""
        image = cv2.imread(self.img_path)
        text = f"{self.labels[max_ind]}: {probs[max_ind]:.4f}"
        org = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 255, 255)  # color
        thickness = 1

        cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def annotate_image_tf(self, max_ind, probs):
        """Annotate the inference result on the image"""
        image = cv2.imread(self.img_path)
        text = f"{self.labels[max_ind]}: {probs:.4f}"
        org = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 255, 255)  # color
        thickness = 1

        cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def visualize(self, image, save_path='output_image.png'):
        """Save the image to a file"""
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Image saved at: {save_path}")
      
        
def load_config(json_file, model_name):
    with open(json_file, 'r') as f:
        config = json.load(f)
    if model_name in config:
        return config[model_name]
    else:
        raise ValueError(f"Model {model_name} not found in config file")


def run_inference(model_name):
    config = load_config("data/model_list.json", model_name)

    # Define paths
    model_path = (config["model_path"])
    labels_path = (config["labels_path"])
    img_path = (config["img_path"])
    img_save_path = (config["img_save_path"])

    # Create inference object
    model_infer = ModelInference(model_path, labels_path, img_path)
    
    # Image preprocessing and inference
    model_infer.preprocess_image()
    logits = model_infer.inference()
    
    # Postprocessing and result display
    max_ind, probs = model_infer.postprocess(logits)
    image_with_text = model_infer.annotate_image(max_ind, probs)
    model_infer.visualize(image_with_text, save_path=img_save_path)
    
    return model_infer.labels[max_ind], probs[max_ind], image_with_text


def run_inference_tf(model_name):
    config = load_config("/root/model_test/data/model_list.json", model_name)

    # Define paths
    model_path = (config["model_path"])
    labels_path = (config["labels_path"])
    img_path = (config["img_path"])
    img_save_path = (config["img_save_path"])

    # Create inference object
    model_infer = ModelInference(model_path, labels_path, img_path)
    
    # Image preprocessing and inference
    model_infer.preprocess_image()
    logits = model_infer.inference()
    
    # Postprocessing and result display
    max_ind, probs = model_infer.postprocess_tf(logits)
    image_with_text = model_infer.annotate_image_tf(max_ind, probs)
    model_infer.visualize(image_with_text, save_path=img_save_path)
    
    return model_infer.labels[max_ind], probs, image_with_text


if __name__ == "__main__":
    run_inference("googlenet")
