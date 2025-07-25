import os
import pickle
import numpy as np
import torch
import cv2
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import find_objects

class ObjectClassifier:
    """
    Handles feature extraction and classification.
    """
    def __init__(self, crop_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ObjectClassifier using device: {self.device}")

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = torch.nn.Identity()
        self.backbone.to(self.device)
        self.backbone.eval()

        # UPDATED: Changed solver to 'lbfgs' to avoid the FutureWarning
        self.classifier = LogisticRegression(solver='lbfgs', max_iter=200)
        self.is_trained = False
        self.crop_size = crop_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((crop_size, crop_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train_and_predict(self, all_images, all_masks, all_labels, predict_image, predict_masks) -> dict:
        """Trains on all labeled data and predicts on the current image."""
        print("Gathering training data from all images...")
        all_train_ids, all_train_labels = [], []
        for i, (img, msk, lab) in enumerate(zip(all_images, all_masks, all_labels)):
            for obj_id, label_id in lab.items():
                if label_id != 0:
                    all_train_ids.append((i, obj_id))
                    all_train_labels.append(label_id)

        if len(all_train_ids) < 2 or len(np.unique(all_train_labels)) < 2:
            print("⚠️ Insufficient data for training.")
            return {}

        print(f"Extracting features for {len(all_train_labels)} total annotations...")
        train_crops = [self._extract_crops_and_preprocess(all_images[i], all_masks[i], [oid]) for i, oid in all_train_ids]
        train_crops = [t for t in train_crops if t.nelement() > 0]
        if not train_crops: return {}
        
        X_train = self._get_embeddings(torch.cat(train_crops))
        
        print(f"Training on {len(all_train_labels)} annotations...")
        self.classifier.fit(X_train, all_train_labels)
        self.is_trained = True

        return self.predict_only(predict_image, predict_masks)

    def predict_only(self, image: np.ndarray, masks: np.ndarray) -> dict:
        """Predicts labels for a single image using the existing classifier."""
        if not self.is_trained: return {}
        all_ids_current = sorted([i for i in np.unique(masks) if i != 0])
        if not all_ids_current: return {}
        predict_tensors = self._extract_crops_and_preprocess(image, masks, all_ids_current)
        if predict_tensors.nelement() == 0: return {}
        X_predict = self._get_embeddings(predict_tensors)
        predictions = self.classifier.predict(X_predict)
        return {int(obj_id): int(pred) for obj_id, pred in zip(all_ids_current, predictions)}
    
    def _extract_crops_and_preprocess(self, image: np.ndarray, masks: np.ndarray, object_ids: list) -> torch.Tensor:
        crop_tensors = []
        locations = find_objects(masks)
        for obj_id in object_ids:
            if obj_id == 0 or obj_id > len(locations) or locations[obj_id - 1] is None: continue
            bbox = locations[obj_id - 1]
            img_crop = image[bbox]
            mask_crop = (masks[bbox] == obj_id).astype(np.uint8)
            masked_img_crop = img_crop * np.stack([mask_crop] * 3, axis=-1)
            crop_tensors.append(self.transform(masked_img_crop))
        if not crop_tensors: return torch.tensor([])
        return torch.stack(crop_tensors).to(self.device)

    @torch.no_grad()
    def _get_embeddings(self, preprocessed_crops: torch.Tensor) -> np.ndarray:
        if preprocessed_crops.nelement() == 0: return np.array([])
        return self.backbone(preprocessed_crops).cpu().numpy()

    def save_state(self, path: str):
        """Saves the essential classifier state to a file."""
        state = {
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"✅ Classifier state saved to {path}")

    def load_state(self, path: str):
        """Loads classifier state into the current instance."""
        print(f"✅ Loading classifier state from {path}")
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.classifier = state.get('classifier', LogisticRegression(solver='lbfgs', max_iter=200))
        self.is_trained = state.get('is_trained', False)
