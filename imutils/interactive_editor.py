import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from skimage.segmentation import find_boundaries
from skimage.draw import polygon as draw_polygon

class InteractiveEditor:
    def __init__(self, event_callback):
        self.callback = event_callback
        self.base_img = None
        self.img_shape = None
        # --- Mode and Data Stores ---
        self.mode = 'mask'  # New: 'mask' or 'point'
        self.point_labels = []  # New: Store for point labels [(x, y, label_id), ...]
        self.masks = np.zeros((1, 1), dtype=np.uint16)
        self.object_labels = {}
        # --- Point Mode Specific ---
        self.POINT_LABEL_MAP = {1: 'alive', 2: 'dead'}
        self.POINT_COLOR_MAP = {'alive': 'g', 'dead': 'r'}
        
        # --- Existing State Variables (largely unchanged) ---
        self.predictions = {}
        self.in_merge_mode = False
        self.merge_candidates = []
        self.in_draw_mode = False
        self.is_actively_drawing = False
        self.draw_points = []
        self.mask_display_mode = 'rings'
        self.color_cache = {}
        self.in_label_mode = False
        self.label_view_mode = 'off'
        self.LABEL_MAP = {0: 'unlabelled', 1: 'alive', 2: 'dead', 3: 'junk'}
        self.COLOR_MAP = {'unlabelled': [0, 0, 255], 'alive': [0, 255, 0], 'dead': [255, 0, 0], 'junk': [128, 128, 128]}
        self.draw_artist = None
        self.background = None
        self.help_text_artist = None
        
        # --- Matplotlib Setup ---
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda evt: self.callback('exit'))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)

    def update_data(self, image: np.ndarray, title: str, mode: str = 'mask', masks: np.ndarray = None, labels: dict = None, point_labels: list = None):
        self.mode = mode
        self.base_img = image
        self.img_shape = image.shape
        
        # Reset state based on mode
        self.predictions = {}
        self.in_merge_mode = False
        self.in_draw_mode = False

        if self.mode == 'point':
            self.point_labels = point_labels.copy() if point_labels else []
            self.masks = np.zeros((1, 1), dtype=np.uint16)
            self.object_labels = {}
        else: # 'mask' mode
            self.point_labels = []
            self.masks = masks.copy() if masks is not None else np.zeros(self.img_shape[:2], dtype=np.uint16)
            self.object_labels = labels.copy() if labels else {int(i): 0 for i in np.unique(self.masks) if i != 0}
            self.label_view_mode = 'labels' if self.in_label_mode else 'off'

        self._redraw_canvas(title=title, maintain_zoom=False)

    def get_masks(self) -> np.ndarray: return self.masks.copy()
    def get_annotations(self) -> dict: return self.object_labels.copy()
    def get_point_labels(self) -> list: return self.point_labels.copy() # New method

    def display_predictions(self, predictions: dict):
        if self.mode != 'mask': return
        self.predictions = predictions
        if self.label_view_mode == 'predictions':
            print(f"Background predictions received for {len(predictions)} objects.")
            self._redraw_canvas()
            self.fig.canvas.draw_idle()

    def start(self): plt.show()
    
    def on_key(self, event):
        # --- Universal keys ---
        if event.key == 'n': self.callback('next_image')
        elif event.key == 'b': self.callback('prev_image')
        elif event.key in ['+', '=']: self.zoom(0.5)
        elif event.key == '-': self.zoom(2.0)
        
        is_zoomed = self.ax.get_xlim()[1] - self.ax.get_xlim()[0] < self.img_shape[1] - 1
        if event.key in ['up', 'down', 'left', 'right'] and is_zoomed: self.pan(event.key)

        # --- Mask-mode specific keys ---
        if self.mode == 'mask':
            if event.key == 'u' and self.in_label_mode: self.callback('classify_objects')
            elif event.key == 'v':
                if self.in_label_mode:
                    modes = ['labels', 'predictions', 'off']
                    current_idx = modes.index(self.label_view_mode)
                    self.label_view_mode = modes[(current_idx + 1) % len(modes)]
                    if self.label_view_mode == 'predictions' and not self.predictions: self.callback('predict_current')
                else:
                    modes = ['rings', 'solid', 'off']
                    current_idx = modes.index(self.mask_display_mode)
                    self.mask_display_mode = modes[(current_idx + 1) % len(modes)]
                self._redraw_canvas()
            elif event.key == 'l':
                self.in_label_mode = not self.in_label_mode
                self.label_view_mode = 'labels' if self.in_label_mode else 'off'
                if self.in_label_mode: self.callback('predict_current')
                self._redraw_canvas()
            elif event.key == 'x':
                self.masks.fill(0); self.object_labels.clear(); self.predictions.clear()
                self._redraw_canvas()
            elif event.key == 'm': self._toggle_merge_mode()
            elif event.key == 'f': self._toggle_draw_mode()
            elif event.key == 'enter' and self.in_merge_mode: self._finalize_merge()

    def _redraw_canvas(self, title: str = "", maintain_zoom: bool = True):
        if self.base_img is None: return
        if maintain_zoom: xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        display_img = self.base_img.copy()
        
        # --- MASK DRAWING LOGIC (unchanged) ---
        if self.mode == 'mask':
            if self.in_label_mode and self.label_view_mode != 'off':
                source_dict = self.predictions if self.label_view_mode == 'predictions' else self.object_labels
                for mask_id in self.object_labels.keys():
                    if mask_id == 0: continue
                    label_id = source_dict.get(mask_id, 0)
                    label_name = self.LABEL_MAP.get(label_id, 'unlabelled')
                    color = self.COLOR_MAP.get(label_name, self.COLOR_MAP['unlabelled'])
                    boundaries = find_boundaries(self.masks == mask_id, mode='inner')
                    display_img[boundaries] = color
            elif self.masks.max() > 0 and self.mask_display_mode != 'off':
                if self.mask_display_mode == 'rings':
                    boundaries = find_boundaries(self.masks, mode='inner')
                    display_img[boundaries] = [0, 255, 255]
                elif self.mask_display_mode == 'solid':
                    self._generate_solid_mask_overlay(display_img)
            if self.in_merge_mode:
                for lbl in self.merge_candidates:
                    mask_region = self.masks == lbl
                    display_img[mask_region] = (0.5 * display_img[mask_region] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)

        self.ax.imshow(display_img)

        # --- POINT DRAWING LOGIC (New) ---
        if self.mode == 'point':
            for x, y, label_id in self.point_labels:
                label_name = self.POINT_LABEL_MAP.get(label_id)
                color = self.POINT_COLOR_MAP.get(label_name)
                self.ax.plot(x, y, marker='x', color=color, markersize=8, mew=2)

        self.ax.set_title(title if title else self.ax.get_title())
        self._update_help_text()
        if maintain_zoom: self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        else: self.ax.set_xlim(0, self.img_shape[1]); self.ax.set_ylim(self.img_shape[0], 0)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        x, y = int(round(event.xdata)), int(round(event.ydata))

        # --- POINT MODE CLICK ---
        if self.mode == 'point':
            # Check if clicking on an existing point to delete it
            point_to_delete = -1
            for i, (px, py, _label) in enumerate(self.point_labels):
                if abs(x - px) < 5 and abs(y - py) < 5: # 5-pixel tolerance
                    point_to_delete = i
                    break
            
            if point_to_delete != -1:
                del self.point_labels[point_to_delete]
                print(f"Point label removed at ({x}, {y}).")
            elif event.button == 1: # Left-click: Add 'alive'
                self.point_labels.append((x, y, 1))
                print(f"Added ALIVE label at ({x}, {y}).")
            elif event.button == 3: # Right-click: Add 'dead'
                self.point_labels.append((x, y, 2))
                print(f"Added DEAD label at ({x}, {y}).")
            self._redraw_canvas()
            return

        # --- MASK MODE CLICK (largely unchanged) ---
        lbl = self.masks[y, x]
        if self.in_label_mode and event.button == 1 and lbl != 0:
            current_label = self.object_labels.get(lbl, 0)
            next_label = (current_label + 1) % len(self.LABEL_MAP)
            self.object_labels[lbl] = next_label
            if self.label_view_mode != 'off': self._redraw_canvas()
        elif event.button == 3 and lbl != 0:
            self.masks[self.masks == lbl] = 0
            if lbl in self.object_labels: del self.object_labels[lbl]
            if lbl in self.predictions: del self.predictions[lbl]
            self._redraw_canvas()
        elif event.button == 1:
            if self.in_merge_mode and lbl != 0 and lbl not in self.merge_candidates:
                self.merge_candidates.append(lbl)
                self._redraw_canvas()
            elif self.in_draw_mode:
                self.is_actively_drawing = True
                self.draw_points = [(x, y)]
                self.draw_artist = Polygon(self.draw_points, animated=True, closed=False, edgecolor='r', linewidth=1, fill=False)
                self.ax.add_patch(self.draw_artist)
                self.fig.canvas.draw()
                
    def _update_help_text(self):
        if self.help_text_artist: self.help_text_artist.remove()
        if self.mode == 'point':
            controls = ["--- POINT LABEL MODE ---", "Left-Click: Add ALIVE", "Right-Click: Add DEAD", "Click on Point: Delete", "'n'/'b': Next/Back Image", "'+'/'-': Zoom"]
        elif self.in_label_mode:
            controls = ["--- LABEL MODE ---", "'l': Toggle Label Mode", "Left-Click: Cycle Label", "'v': Toggle View", "'u': Update Classifier", "'n'/'b': Next/Back"]
        else: # Mask edit mode
            controls = ["--- MASK EDIT MODE ---", "'l': Toggle Label Mode", "'n'/'b': Next/Back", "'v': Toggle View", "'x': Clear Masks", "Right-Click: Delete", "'f': Draw", "'m': Merge"]
        self.help_text_artist = self.fig.text(0.01, 0.99, "\n".join(controls), transform=self.fig.transFigure, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))

    # --- Unchanged Helper Methods ---
    def on_draw(self, event): self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
    def _generate_solid_mask_overlay(self, display_img):
        for label in np.unique(self.masks):
            if label == 0: continue
            if label not in self.color_cache: self.color_cache[label] = np.random.rand(3)
            mask_region = self.masks == label; color = self.color_cache[label]
            display_img[mask_region] = (0.6 * display_img[mask_region] + 0.4 * np.array(color) * 255).astype(np.uint8)
    def on_release(self, event):
        if not self.is_actively_drawing: return
        self.is_actively_drawing = False; self._finalize_drawing()
    def on_motion(self, event):
        if not self.is_actively_drawing or event.inaxes != self.ax or event.xdata is None: return
        x, y = int(event.xdata), int(event.ydata); self.draw_points.append((x, y))
        self.draw_artist.set_xy(self.draw_points)
        if self.background is None: return
        self.fig.canvas.restore_region(self.background); self.ax.draw_artist(self.draw_artist); self.fig.canvas.blit(self.ax.bbox)
    def _toggle_merge_mode(self):
        self.in_merge_mode = not self.in_merge_mode; self.merge_candidates = []
        self._redraw_canvas()
    def _toggle_draw_mode(self): self.in_draw_mode = not self.in_draw_mode
    def _finalize_merge(self):
        if len(self.merge_candidates) > 1:
            target_lbl = self.merge_candidates[0]
            for lbl in self.merge_candidates[1:]: self.masks[self.masks == lbl] = target_lbl
        self._toggle_merge_mode()
    def _finalize_drawing(self):
        if len(self.draw_points) > 2:
            xs, ys = [p[0] for p in self.draw_points], [p[1] for p in self.draw_points]
            if max(xs) - min(xs) > 1 and max(ys) - min(ys) > 1:
                rr, cc = draw_polygon(ys, xs, self.img_shape)
                new_label = self.masks.max() + 1
                self.masks[rr, cc] = new_label
                if self.in_label_mode: self.object_labels[new_label] = 0
        if self.draw_artist: self.draw_artist.remove()
        self.draw_points = []; self.draw_artist = None; self._redraw_canvas()
    def zoom(self, factor: float):
        x, y = self.ax.get_xlim(), self.ax.get_ylim(); cx, cy = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        new_w, new_h = (x[1] - x[0]) * factor, (y[1] - y[0]) * factor
        self.ax.set_xlim(cx - new_w / 2, cx + new_w / 2); self.ax.set_ylim(cy - new_h / 2, cy + new_h / 2); self.fig.canvas.draw()
    def pan(self, direction: str):
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim(); width, height = xlim[1] - xlim[0], ylim[0] - ylim[1]
        dx = dy = 0; pan_factor = 0.1
        if direction == 'up': dy = -height * pan_factor
        if direction == 'down': dy = height * pan_factor
        if direction == 'left': dx = -width * pan_factor
        if direction == 'right': dx = width * pan_factor
        self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx); self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy); self.fig.canvas.draw()
