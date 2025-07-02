import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import json
from datetime import datetime

from utils.utils import lane_line_mask, select_device, LoadImages, letterbox

class YOLOPv2LaneClassifier:
    def __init__(self, weights_path='weights/yolopv2.pt', device='cuda'):
        # Handle device selection
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Load YOLOPv2 model
        self.model = torch.jit.load(weights_path)
        self.model = self.model.to(self.device)
        if self.device.type != 'cpu':
            self.model.half()  # FP16
        self.model.eval()
        
        # Lane classification setup
        self.lane_colors = {
            'white_solid': (255, 255, 255),
            'yellow_solid': (0, 255, 255),
            'white_dashed': (255, 200, 255),
            'yellow_dashed': (0, 200, 255),
            'road_edge': (200, 200, 200)
        }
        
        self.setup_classifier()
        
        # Initialize frame data storage
        self.frame_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': 'YOLOPv2',
                'weights': weights_path
            },
            'frames': []
        }
    
    def setup_classifier(self):
        """Setup color detection parameters"""
        self.color_ranges = {
            'white': {
                'lower_hsv': np.array([0, 0, 150]),
                'upper_hsv': np.array([180, 40, 255])
            },
            'yellow': {
                'lower_hsv': np.array([15, 50, 100]),
                'upper_hsv': np.array([35, 255, 255])
            }
        }
    
    def resize_mask_with_aspect_ratio(self, mask, target_width, target_height):
        """Resize mask maintaining aspect ratio and crop/pad as needed"""
        mask_h, mask_w = mask.shape
        target_aspect = target_width / target_height
        mask_aspect = mask_w / mask_h
        
        if mask_aspect > target_aspect:
            # Mask is wider - fit to width and crop height
            new_width = target_width
            new_height = int(target_width / mask_aspect)
            resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Crop top and bottom to fit target height
            if new_height > target_height:
                crop_top = (new_height - target_height) // 2
                resized = resized[crop_top:crop_top + target_height, :]
            else:
                # Pad if needed
                pad_top = (target_height - new_height) // 2
                pad_bottom = target_height - new_height - pad_top
                resized = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            # Mask is taller - fit to height and crop width
            new_height = target_height
            new_width = int(target_height * mask_aspect)
            resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Crop left and right to fit target width
            if new_width > target_width:
                crop_left = (new_width - target_width) // 2
                resized = resized[:, crop_left:crop_left + target_width]
            else:
                # Pad if needed
                pad_left = (target_width - new_width) // 2
                pad_right = target_width - new_width - pad_left
                resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        
        return resized
    
    def extract_lane_segments(self, lane_mask):
        """Extract individual lane segments from binary mask"""
        # Apply slight morphological closing to connect very close segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
        closed_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and angle
        min_area = 200  # Increased minimum area
        valid_contours = []
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
                
            # Fit line to contour
            if len(c) < 5:  # Need at least 5 points for ellipse fit
                continue
                
            # Get rotated rectangle
            rect = cv2.minAreaRect(c)
            angle = rect[2]
            
            # Normalize angle to 0-90 range
            angle = abs(angle) if abs(angle) < 45 else 90 - abs(angle)
            
            # Only accept near-vertical lines (within 30 degrees of vertical)
            if angle < 30:
                valid_contours.append(c)
        
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        # Group nearby contours that belong to same lane
        grouped_lanes = []
        used = set()
        
        for i, contour in enumerate(valid_contours):
            if i in used:
                continue
            
            # Get contour properties
            rect = cv2.minAreaRect(contour)
            center = rect[0]
            x_center = center[0]
            
            group = [contour]
            used.add(i)
            
            # Look for vertically aligned contours
            for j, other_contour in enumerate(valid_contours[i+1:], i+1):
                if j in used:
                    continue
                    
                other_rect = cv2.minAreaRect(other_contour)
                other_center = other_rect[0]
                other_x_center = other_center[0]
                
                # Check if contours are aligned (similar x-coordinate and angle)
                if abs(x_center - other_x_center) < 50:
                    # Check angles are similar
                    angle_diff = abs(rect[2] - other_rect[2]) % 90
                    if angle_diff < 15 or angle_diff > 75:  # Allow parallel or anti-parallel
                        group.append(other_contour)
                        used.add(j)
            
            # Only keep groups that look like lanes (long enough)
            if len(group) > 1:
                points = np.vstack([c.reshape(-1, 2) for c in group])
                y_range = points[:, 1].max() - points[:, 1].min()
                if y_range > 100:  # Minimum vertical span
                    grouped_lanes.append(group)
            else:
                # For single contours, check vertical span
                points = contour.reshape(-1, 2)
                y_range = points[:, 1].max() - points[:, 1].min()
                if y_range > 150:  # Higher threshold for single contours
                    grouped_lanes.append(group)
        
        return grouped_lanes
    
    def classify_lane_type(self, frame, lane_contours):
        """Classify lane type based on color and pattern"""
        # Create mask from contours
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, lane_contours, -1, 255, -1)
        
        # Dilate mask slightly to capture more color info
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Get masked pixels
        masked_pixels = cv2.bitwise_and(frame, frame, mask=dilated_mask)
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(masked_pixels, cv2.COLOR_BGR2HSV)
        
        # Also use HLS for better white detection
        hls = cv2.cvtColor(masked_pixels, cv2.COLOR_BGR2HLS)
        
        # Detect white in HLS
        white_lower_hls = np.array([0, 150, 0])
        white_upper_hls = np.array([180, 255, 255])
        white_mask = cv2.inRange(hls, white_lower_hls, white_upper_hls)
        
        # Detect yellow in HSV
        yellow_mask = cv2.inRange(hsv, 
                                 self.color_ranges['yellow']['lower_hsv'],
                                 self.color_ranges['yellow']['upper_hsv'])
        
        # Apply mask
        white_mask = cv2.bitwise_and(white_mask, dilated_mask)
        yellow_mask = cv2.bitwise_and(yellow_mask, dilated_mask)
        
        # Count pixels
        white_pixels = cv2.countNonZero(white_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        total_pixels = cv2.countNonZero(dilated_mask)
        
        # Calculate mean color in the original mask area
        mean_bgr = cv2.mean(frame, mask=mask)[:3]
        b, g, r = mean_bgr
        
        # Determine color
        color = 'white'
        if total_pixels > 0:
            yellow_ratio = yellow_pixels / total_pixels
            white_ratio = white_pixels / total_pixels
            
            # Debug print for first few detections
            if hasattr(self, 'debug_count') and self.debug_count < 5:
                print(f"Lane color detection - Yellow ratio: {yellow_ratio:.3f}, White ratio: {white_ratio:.3f}, RGB: ({r:.0f},{g:.0f},{b:.0f})")
                self.debug_count += 1
            
            if yellow_ratio > 0.05:  # Low threshold for yellow
                color = 'yellow'
            elif white_ratio < 0.1 and r > 150 and g > 120 and b < 100:
                # Fallback: check if it looks yellow by RGB
                color = 'yellow'
        
        # Detect if dashed
        is_dashed = len(lane_contours) > 2
        
        return f"{color}_{'dashed' if is_dashed else 'solid'}"
    
    def process_frame(self, frame, frame_idx):
        """Process single frame with correct mask alignment"""
        orig_h, orig_w = frame.shape[:2]
        
        # DEBUG: Print info for first frame
        if frame_idx == 0:
            print(f"\nProcessing video:")
            print(f"Original frame size: {orig_w}x{orig_h}")
            print(f"Frame aspect ratio: {orig_w/orig_h:.3f}")
        
        # Prepare input for model - YOLOPv2 expects 640x640
        # Use letterbox to maintain aspect ratio
        img_letterboxed, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640), auto=True, stride=32)
        img = img_letterboxed[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type != 'cpu' else img.float()
        img = img / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(img)
            
            # Extract lane mask
            ll_seg_mask = lane_line_mask(ll)
            
            # DEBUG: Print mask info for first frame
            if frame_idx == 0:
                print(f"Raw model output shape: {ll.shape}")
                print(f"Lane mask shape after lane_line_mask: {ll_seg_mask.shape}")
            
            # The lane_line_mask function returns a mask of shape (720, 1280)
            # We need to properly scale and unpad this to match our original image dimensions
            
            # First threshold the mask
            ll_seg_mask = (ll_seg_mask > 0.6).astype(np.uint8) * 255
            
            # Remove padding added by letterbox
            if dw > 0:
                ll_seg_mask = ll_seg_mask[:, int(dw):ll_seg_mask.shape[1]-int(dw)]
            if dh > 0:
                ll_seg_mask = ll_seg_mask[int(dh):ll_seg_mask.shape[0]-int(dh), :]
            
            # Calculate scaling factors based on the unpadded size
            scale_y = orig_h / ll_seg_mask.shape[0]
            scale_x = orig_w / ll_seg_mask.shape[1]
            
            if frame_idx == 0:
                print(f"Scaling factors after removing padding - x: {scale_x:.3f}, y: {scale_y:.3f}")
            
            # Resize mask to match original frame dimensions
            ll_seg_mask_resized = cv2.resize(
                ll_seg_mask, 
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Re-threshold after resize
            ll_seg_mask_resized = (ll_seg_mask_resized > 127).astype(np.uint8) * 255
            
            if frame_idx == 0:
                print(f"Final mask shape: {ll_seg_mask_resized.shape}")
        
        # Extract and classify lane segments
        lane_groups = self.extract_lane_segments(ll_seg_mask_resized)
        
        # Create result frame
        result_frame = frame.copy()
        
        # Debug mode - save alignment check
        if hasattr(self, 'debug_mode') and self.debug_mode and frame_idx < 5:
            # Create side-by-side comparison
            debug_vis = np.zeros((orig_h, orig_w * 3, 3), dtype=np.uint8)
            
            # Original frame
            debug_vis[:, :orig_w] = frame
            
            # Lane mask overlaid on frame (to check alignment)
            mask_overlay = frame.copy()
            mask_colored = np.zeros_like(frame)
            mask_colored[:,:,2] = ll_seg_mask_resized  # Red channel
            
            # Draw original lane positions in blue for comparison
            lane_positions = np.zeros_like(frame)
            for lane_group in lane_groups:
                cv2.drawContours(lane_positions, lane_group, -1, (255, 0, 0), 2)
            
            # Combine overlays
            cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0, mask_overlay)
            cv2.addWeighted(mask_overlay, 1.0, lane_positions, 0.5, 0, mask_overlay)
            
            debug_vis[:, orig_w:orig_w*2] = mask_overlay
            
            # Add grid lines for alignment check
            grid_overlay = mask_overlay.copy()
            for x in range(0, orig_w, 100):
                cv2.line(grid_overlay, (x, 0), (x, orig_h), (0, 255, 0), 1)
            for y in range(0, orig_h, 100):
                cv2.line(grid_overlay, (0, y), (orig_w, y), (0, 255, 0), 1)
            cv2.addWeighted(mask_overlay, 0.8, grid_overlay, 0.2, 0, debug_vis[:, orig_w:orig_w*2])
            
            # Final result
            debug_vis[:, orig_w*2:] = result_frame
            
            # Add labels and debug info
            cv2.putText(debug_vis, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(debug_vis, f"Mask Alignment (Red) + Contours (Blue)", (orig_w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(debug_vis, "Final Result", (orig_w*2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add grid coordinates
            for x in range(0, orig_w, 200):
                cv2.putText(debug_vis, f"x:{x}", (orig_w + x, orig_h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite(f'debug_alignment_{frame_idx:04d}.jpg', debug_vis)
            print(f"Saved debug_alignment_{frame_idx:04d}.jpg with alignment grid")
        
        # Process each lane group
        for i, lane_group in enumerate(lane_groups):
            # Classify lane type
            lane_type = self.classify_lane_type(frame, lane_group)
            color = self.lane_colors.get(lane_type, (255, 255, 255))
            
            # Create overlay for this lane
            overlay = np.zeros_like(frame)
            
            # Draw contours with moderate thickness
            cv2.drawContours(overlay, lane_group, -1, color, 8)
            
            # Also fill the contours with semi-transparent color
            cv2.drawContours(overlay, lane_group, -1, color, -1)
            
            # Blend with result - simplified blending operation
            mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0
            result_frame[mask] = cv2.addWeighted(result_frame, 0.6, overlay, 0.4, 0)[mask]
            
            # Add label
            if lane_group:
                # Get topmost point of first contour
                topmost = tuple(lane_group[0][lane_group[0][:,:,1].argmin()][0])
                
                # Draw label with background
                label = lane_type
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                label_x = max(10, topmost[0] - text_w // 2)
                label_y = max(text_h + 10, topmost[1] - 10)
                
                # Background
                cv2.rectangle(result_frame, 
                            (label_x - 5, label_y - text_h - 5),
                            (label_x + text_w + 5, label_y + 5),
                            (0, 0, 0), -1)
                
                # Text
                cv2.putText(result_frame, label, (label_x, label_y),
                           font, font_scale, color, thickness)
        
        # Draw legend
        self.draw_legend(result_frame)
        
        return result_frame, ll_seg_mask_resized
    
    def draw_legend(self, frame):
        """Draw simple legend"""
        y_offset = 20
        x_offset = 10
        
        # Semi-transparent background
        legend_bg = np.zeros((len(self.lane_colors) * 20 + 10, 180, 3), dtype=np.uint8)
        
        for lane_type, color in self.lane_colors.items():
            cv2.rectangle(legend_bg, (5, y_offset - 15), (25, y_offset), color, -1)
            cv2.putText(legend_bg, lane_type, (35, y_offset - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Blend onto frame
        roi = frame[10:10+legend_bg.shape[0], 10:10+legend_bg.shape[1]]
        frame[10:10+legend_bg.shape[0], 10:10+legend_bg.shape[1]] = cv2.addWeighted(roi, 0.3, legend_bg, 0.7, 0)
    
    def process_video(self, input_path, output_path, json_path=None, debug=False):
        """Process video with debug option"""
        self.debug_mode = debug
        
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result_frame, lane_mask = self.process_frame(frame, frame_count)
            
            # Write frame
            out.write(result_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        out.release()
        print(f"Video saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='test_video.mp4')
    parser.add_argument('--output', type=str, default='output_aspect_fixed.mp4')
    parser.add_argument('--weights', type=str, default='weights/yolopv2.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    # Create classifier
    classifier = YOLOPv2LaneClassifier(weights_path=args.weights, device=args.device)
    
    # Process video
    classifier.process_video(args.source, args.output, debug=args.debug)