import os
import json
import threading
import numpy as np
from tifffile import imwrite # Import imwrite
from .interactive_editor import InteractiveEditor

class CurationController:
    """
    Manages the labelling session. This version is backward-compatible with older scripts.
    """
    def __init__(self, images: list, titles: list,
                 mode: str = None,
                 session_path: str = None,
                 initial_masks: list = None,
                 initial_labels: list = None,
                 mask_save_paths: list = None, # Add new argument for mask save paths
                 object_classifier=None,
                 classifier_path: str = None,
                 **kwargs): # Accept and ignore legacy arguments

        # --- Backward Compatibility Logic ---
        if mode is None:
            if initial_masks is not None:
                self.mode = 'mask'
                print("INFO: No 'mode' specified. Inferred 'mask' mode based on provided masks.")
            else:
                self.mode = 'point'
                print("INFO: No 'mode' specified. Defaulting to 'point' mode.")
        else:
            self.mode = mode

        if 'output_paths' in kwargs:
            print("⚠️ WARNING: The 'output_paths' argument is deprecated and has no effect. "
                  "Saving is now handled by the 'session_path' argument.")

        self.images = images
        self.titles = titles
        self.session_path = session_path
        self.mask_save_paths = mask_save_paths # Store the new argument

        # --- Mode-specific initialisation ---
        if self.mode == 'point':
            self.point_labels = initial_labels if initial_labels is not None else [[] for _ in images]
            self.masks = None
            self.labels = None
            self.object_classifier = None
            self.classifier_path = None

        elif self.mode == 'mask':
            if initial_masks is None:
                raise ValueError("In 'mask' mode, 'initial_masks' must be provided.")

            self.masks = [m.copy() for m in initial_masks]

            # Auto-generate labels if not provided, ensuring old scripts don't fail
            if initial_labels is None:
                print("INFO: 'initial_labels' not provided. Generating default labels from masks.")
                self.labels = [{int(i): 0 for i in np.unique(m) if i != 0} for m in self.masks]
            else:
                self.labels = [l.copy() for l in initial_labels]

            self.point_labels = None
            self.object_classifier = object_classifier
            self.classifier_path = classifier_path
        else:
            raise ValueError(f"Invalid mode specified: {self.mode}")

        self.editor = InteractiveEditor(event_callback=self.handle_editor_event)
        self.idx = 0
        self.is_running = False

    def _save_session_to_disk(self):
        if not self.session_path and not self.mask_save_paths:
             print("\n⚠️ No 'session_path' or 'mask_save_paths' provided. Skipping save.")
             return

        # --- Save Session Labels (JSON) ---
        if self.session_path:
            print(f"\n💾 Saving session data to {self.session_path}...")
            if self.mode == 'point':
                all_points = []
                for i, points in enumerate(self.point_labels):
                    title = self.titles[i]
                    for x, y, label_id in points:
                        label_name = self.editor.POINT_LABEL_MAP.get(label_id, 'unknown')
                        all_points.append({'image': title, 'x': x, 'y': y, 'label': label_name})
                with open(self.session_path, 'w') as f:
                    json.dump(all_points, f, indent=2)
                print(f"✅ Session point labels saved.")
            else: # mask mode
                labels_to_save = [{str(k): v for k, v in d.items()} for d in self.labels]
                with open(self.session_path, 'w') as f:
                    json.dump(labels_to_save, f, indent=2)
                print(f"✅ Session object labels saved.")
                if self.object_classifier and self.classifier_path:
                    self.object_classifier.save_state(self.classifier_path)

        # --- Save Curated Masks (TIFF) ---
        if self.mode == 'mask' and self.mask_save_paths:
            print(f"\n💾 Saving curated masks...")
            for i, mask in enumerate(self.masks):
                if i < len(self.mask_save_paths) and self.mask_save_paths[i]:
                    save_path = self.mask_save_paths[i]
                    try:
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        # Save mask as uint16, a standard for segmentation masks
                        imwrite(save_path, mask.astype(np.uint16))
                        print(f"  -> Saved mask {i+1}/{len(self.masks)} to {save_path}")
                    except Exception as e:
                        print(f"  -> ERROR saving mask {i+1}: {e}")

    # The rest of the class methods (start, _update_state_from_editor, etc.) remain unchanged.
    def start(self):
        if not self.images:
            print("No images to process.")
            return
        self.is_running = True
        self._load_data_into_editor()
        self.editor.start()

    def _update_state_from_editor(self):
        if self.mode == 'point':
            self.point_labels[self.idx] = self.editor.get_point_labels()
        else: # mask mode
            self.masks[self.idx] = self.editor.get_masks()
            self.labels[self.idx] = self.editor.get_annotations()

    def _load_data_into_editor(self):
        image = self.images[self.idx]
        title = f"({self.idx + 1}/{len(self.images)}) {self.titles[self.idx]}"

        if self.mode == 'point':
            points = self.point_labels[self.idx]
            self.editor.update_data(image, title, mode='point', point_labels=points)
        else: # mask mode
            masks = self.masks[self.idx]
            labels = self.labels[self.idx]
            self.editor.update_data(image, title, mode='mask', masks=masks, labels=labels)
            self.run_background_prediction()

    def run_background_prediction(self):
        if self.mode != 'mask' or not self.object_classifier or not self.object_classifier.is_trained:
            return
        pred_thread = threading.Thread(
            target=self._predict_and_display,
            args=(self.images[self.idx], self.masks[self.idx]), daemon=True)
        pred_thread.start()

    def _predict_and_display(self, image, masks):
        predictions = self.object_classifier.predict_only(image, masks)
        if predictions:
            self.editor.display_predictions(predictions)

    def handle_editor_event(self, event: str):
        if not self.is_running: return

        if event in ['next_image', 'prev_image']:
            self._update_state_from_editor()
            if event == 'next_image':
                if self.idx < len(self.images) - 1: self.idx += 1
                else: print("✨ Already at the last image.")
            elif event == 'prev_image':
                if self.idx > 0: self.idx -= 1
                else: print("✨ Already at the first image.")
            self._load_data_into_editor()

        elif self.mode == 'mask' and event == 'classify_objects':
            print("\n🔬 Training classifier on all annotations...")
            self._update_state_from_editor()
            predictions = self.object_classifier.train_and_predict(
                all_images=self.images, all_masks=self.masks, all_labels=self.labels,
                predict_image=self.images[self.idx], predict_masks=self.masks[self.idx])
            if predictions: self.editor.display_predictions(predictions)

        elif self.mode == 'mask' and event == 'predict_current':
            self.run_background_prediction()

        elif event == 'exit':
            self._update_state_from_editor()
            self._save_session_to_disk()
            self.is_running = False
            print("Session ended.")
