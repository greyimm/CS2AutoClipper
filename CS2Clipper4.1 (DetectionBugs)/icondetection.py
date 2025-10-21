"""
Icon Detection Module for CS2 Auto Clipper

This module handles weapon icon detection within red rectangle regions.
It uses multi-scale template matching with transparent PNG icons to identify weapons.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def get_icon_folder_path():
    """
    Get the path to the weapon icons folder.
    Always looks relative to the script directory (CS2Clipper).
    
    Returns:
        str: Full path to CS2 Icons/Weapons folder
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'CS2 Icons', 'Weapons')

# ========================================
# ICON PRIORITY (from IconPriority.txt)
# ========================================

ICON_PRIORITY = [
    'ak47',
    'awp',
    'm4a4',
    'm4a1s',
    'scout',
    'galil',
    'famas',
    'aug',
    'sg556',
    'scar20',
    'g3sg1',
    'mac10',
    'mp9',
    'mp7',
    'ump45',
    'p90',
    'ppbizon',
    'usp',
    'glock',
    'deagle',
    'tec9',
    'dualies',
    'p2000',
    'p250',
    'fiveseven',
    'cz75',
    'r8revolver',
    'zeus',
    'mag7',
    'xm1014',
    'nova',
    'sawedoff',
    'negev',
    'm249'
]


# ========================================
# ICON LOADING AND CACHING
# ========================================

class WeaponIconDetector:
    """
    Handles loading and matching weapon icons from template images.
    Uses multi-scale template matching to handle size variations.
    """
    
    def __init__(self, icon_folder=None, debug=True):
        """
        Initialize the weapon icon detector.
        
        Args:
            icon_folder (str): Path to folder containing weapon icon PNGs (default: auto-detect)
            debug (bool): Enable debug output
        """
        # Use default path if not specified
        if icon_folder is None:
            icon_folder = get_icon_folder_path()
        
        self.icon_folder = Path(icon_folder)
        self.debug = debug
        self.icons = {}
        self.icon_sizes = {}
        
        # Detection thresholds - UPDATED AS REQUESTED
        self.certainty_threshold = 0.70  # 70% match = minimum for consideration
        self.minimum_threshold = 0.70    # Same as certainty - accept 0.7+
        
        # Multi-scale parameters
        self.scale_range = (0.1, 1.0)    # Test scales from 10% to 100%
        self.scale_step = 0.05           # Test every 5% increment
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"INITIALIZING WEAPON ICON DETECTOR")
            print(f"{'='*60}")
            print(f"Icon folder: {self.icon_folder.absolute()}")
            print(f"Certainty threshold: {self.certainty_threshold} (LOWERED TO 0.7+)")
            print(f"Minimum threshold: {self.minimum_threshold}")
            print(f"Scale range: {self.scale_range[0]}-{self.scale_range[1]} (step: {self.scale_step})")
            print(f"{'='*60}\n")
        
        self._load_icons()
    
    
    def _load_icons(self):
        """
        Load weapon icon templates from disk.
        Only loads icons that are in ICON_PRIORITY list.
        """
        if not self.icon_folder.exists():
            print(f"⚠️ WARNING: Icon folder not found: {self.icon_folder}")
            print(f"   Please create the folder and add weapon icon PNGs")
            return
        
        loaded_count = 0
        
        # Load icons in priority order
        for weapon_name in ICON_PRIORITY:
            icon_path = self.icon_folder / f"{weapon_name}.png"
            
            if icon_path.exists():
                # Load with alpha channel (transparency)
                icon = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
                
                if icon is not None:
                    self.icons[weapon_name] = icon
                    self.icon_sizes[weapon_name] = (icon.shape[1], icon.shape[0])  # width, height
                    loaded_count += 1
                    
                    if self.debug:
                        has_alpha = icon.shape[2] == 4 if len(icon.shape) == 3 else False
                        print(f"  ✓ Loaded {weapon_name}.png ({icon.shape[1]}x{icon.shape[0]}, alpha: {has_alpha})")
                else:
                    if self.debug:
                        print(f"  ✗ Failed to load {weapon_name}.png (corrupt file?)")
            else:
                if self.debug:
                    print(f"  - {weapon_name}.png not found (skipping)")
        
        print(f"\n📦 Loaded {loaded_count}/{len(ICON_PRIORITY)} weapon icons")
        
        if loaded_count == 0:
            print(f"⚠️ WARNING: No icons loaded! Detection will not work.")
            print(f"   Add PNG files to: {self.icon_folder.absolute()}")
    
    
    def detect_weapon_in_region(self, region_bgr, check_only=None):
        """
        Detect which weapon icon is present in the given region using multi-scale matching.
        NOW CHECKS ALL WEAPONS and returns the highest confidence match if >= 0.7
        
        Args:
            region_bgr (numpy.ndarray): BGR image region to search (from killfeed)
            check_only (list): DEPRECATED - Now always checks all weapons
        
        Returns:
            tuple: (weapon_name: str or None, confidence: float, debug_info: dict)
        """
        if len(self.icons) == 0:
            if self.debug:
                print("   ❌ ERROR: No icons loaded!")
            return (None, 0.0, {'error': 'No icons loaded'})
        
        # ALWAYS check ALL weapons now (ignore check_only parameter)
        weapons_to_check = [w for w in ICON_PRIORITY if w in self.icons]
        
        if self.debug:
            print(f"\n🔍 ICON DETECTION DEBUG:")
            print(f"   Region size: {region_bgr.shape[1]}x{region_bgr.shape[0]} (width x height)")
            print(f"   Checking ALL {len(weapons_to_check)} weapons")
            print(f"   Multi-scale range: {self.scale_range[0]:.2f}-{self.scale_range[1]:.2f}")
        
        best_match = None
        best_confidence = 0.0
        best_scale = 1.0
        best_position = (0, 0)
        match_details = []
        
        # Check ALL weapons in priority order
        for weapon_name in weapons_to_check:
            template = self.icons[weapon_name]
            
            # Perform multi-scale template matching
            result = self._match_template_multiscale(region_bgr, template, weapon_name)
            
            if result is None:
                continue
            
            confidence, scale, position = result
            
            match_details.append({
                'weapon': weapon_name,
                'confidence': confidence,
                'scale': scale,
                'position': position
            })
            
            if self.debug:
                # Detailed status messages
                if confidence >= self.certainty_threshold:
                    status = "✓✓ MATCH!"
                else:
                    status = "✗ No"
                
                print(f"   {status} {weapon_name}: {confidence:.3f} @ {scale:.2f}x scale, pos=({position[0]}, {position[1]})")
            
            # Update best match
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = weapon_name
                best_scale = scale
                best_position = position
        
        # Final result - NO EARLY STOPPING, check all weapons
        debug_info = {
            'matches_checked': len(match_details),
            'all_scores': match_details,
            'region_size': (region_bgr.shape[1], region_bgr.shape[0]),
            'weapons_to_check': weapons_to_check,
            'best_scale': best_scale,
            'best_position': best_position
        }
        
        # Return best match if >= 0.7 threshold
        if best_match and best_confidence >= self.minimum_threshold:
            if self.debug:
                print(f"   ✅ DETECTED: {best_match} (confidence: {best_confidence:.3f})")
                print(f"      Scale: {best_scale:.2f}x, Position: ({best_position[0]}, {best_position[1]})")
            return (best_match, best_confidence, debug_info)
        else:
            if self.debug:
                print(f"   ❌ NO MATCH FOUND")
                print(f"   📊 Best score: {best_match or 'none'} @ {best_confidence:.3f} (below threshold {self.minimum_threshold})")
                if len(match_details) > 0:
                    print(f"   💡 Top 5 scores:")
                    sorted_matches = sorted(match_details, key=lambda x: x['confidence'], reverse=True)
                    for i, match in enumerate(sorted_matches[:5], 1):
                        print(f"      {i}. {match['weapon']}: {match['confidence']:.3f} @ {match['scale']:.2f}x")
            return (None, best_confidence, debug_info)
    
    def _match_template(self, region_bgr, template, weapon_name):
        """
        Perform template matching with optional alpha mask.
        Logs shape information for white-only icons.
        
        Args:
            region_bgr (numpy.ndarray): Search region in BGR
            template (numpy.ndarray): Template image (BGR or BGRA)
            weapon_name (str): Name of weapon (for debugging)
        
        Returns:
            float: Match confidence (0.0 to 1.0)
        """
        try:
            # Handle alpha channel if present
            if len(template.shape) == 3 and template.shape[2] == 4:
                # Template has alpha channel
                template_bgr = template[:, :, :3]
                alpha = template[:, :, 3]
                
                # Ensure mask is binary (0 or 255)
                _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
                
                # NEW: Log shape dimensions for white icons
                if self.debug:
                    # Get actual icon dimensions from alpha mask
                    coords = cv2.findNonZero(alpha)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        aspect_ratio = w / h if h > 0 else 0
                        print(f"      Shape: {weapon_name:12s} -> w={w:3d}px, h={h:2d}px, aspect={aspect_ratio:.2f}")
                
                # Use masked template matching
                result = cv2.matchTemplate(region_bgr, template_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
            else:
                # No alpha channel, standard matching
                if self.debug:
                    h, w = template.shape[:2]
                    aspect_ratio = w / h if h > 0 else 0
                    print(f"      Shape: {weapon_name:12s} -> w={w:3d}px, h={h:2d}px, aspect={aspect_ratio:.2f}")
                
                result = cv2.matchTemplate(region_bgr, template, cv2.TM_CCOEFF_NORMED)
            
            # Get best match score
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            return max_val
            
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Error matching {weapon_name}: {e}")
            return 0.0
    
    def _match_template_multiscale(self, region_bgr, template, weapon_name):
        """
        Perform multi-scale template matching to handle size variations.
        Tests template at multiple scales and returns the best match.
        
        Args:
            region_bgr (numpy.ndarray): Search region in BGR
            template (numpy.ndarray): Template image (BGR or BGRA)
            weapon_name (str): Name of weapon (for debugging)
        
        Returns:
            tuple: (confidence: float, scale: float, position: tuple) or None if no match
        """
        best_confidence = 0.0
        best_scale = 1.0
        best_position = (0, 0)
        
        # Generate scales to test
        scales = np.arange(self.scale_range[0], self.scale_range[1] + self.scale_step, self.scale_step)
        
        for scale in scales:
            # Calculate scaled template size
            if len(template.shape) == 3 and template.shape[2] == 4:
                template_h, template_w = template.shape[:2]
            else:
                template_h, template_w = template.shape[:2]
            
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)
            
            # Skip if scaled template is larger than region
            if scaled_w > region_bgr.shape[1] or scaled_h > region_bgr.shape[0]:
                continue
            
            # Skip if scaled template is too small (less than 10 pixels in any dimension)
            if scaled_w < 10 or scaled_h < 10:
                continue
            
            # Resize template
            scaled_template = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            
            try:
                # Perform template matching at this scale
                if len(scaled_template.shape) == 3 and scaled_template.shape[2] == 4:
                    # Template has alpha channel
                    template_bgr = scaled_template[:, :, :3]
                    alpha = scaled_template[:, :, 3]
                    
                    # Ensure mask is binary (0 or 255)
                    _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
                    
                    # Use masked template matching
                    result = cv2.matchTemplate(region_bgr, template_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
                else:
                    # No alpha channel, standard matching
                    result = cv2.matchTemplate(region_bgr, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # Get best match score and position at this scale
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # Update best match if this scale is better
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_scale = scale
                    best_position = max_loc
                    
            except Exception as e:
                if self.debug:
                    print(f"      Error at scale {scale:.2f}x for {weapon_name}: {e}")
                continue
        
        # Log shape information at best scale
        if self.debug and best_confidence > 0:
            # Get actual icon dimensions at best scale
            if len(template.shape) == 3 and template.shape[2] == 4:
                alpha = template[:, :, 3]
                coords = cv2.findNonZero(alpha)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    scaled_w = int(w * best_scale)
                    scaled_h = int(h * best_scale)
                    aspect_ratio = w / h if h > 0 else 0
                    print(f"      Shape @ {best_scale:.2f}x: {weapon_name:12s} -> w={scaled_w:3d}px, h={scaled_h:2d}px, aspect={aspect_ratio:.2f}")
            else:
                h, w = template.shape[:2]
                scaled_w = int(w * best_scale)
                scaled_h = int(h * best_scale)
                aspect_ratio = w / h if h > 0 else 0
                print(f"      Shape @ {best_scale:.2f}x: {weapon_name:12s} -> w={scaled_w:3d}px, h={scaled_h:2d}px, aspect={aspect_ratio:.2f}")
        
        if best_confidence > 0:
            return (best_confidence, best_scale, best_position)
        else:
            return None

    def set_thresholds(self, certainty=None, minimum=None):
        """
        Adjust detection thresholds for tuning.
        
        Args:
            certainty (float): Threshold for "100% certainty" (0.0-1.0)
            minimum (float): Minimum threshold to consider a match (0.0-1.0)
        """
        if certainty is not None:
            self.certainty_threshold = certainty
            print(f"✓ Certainty threshold set to: {certainty}")
        
        if minimum is not None:
            self.minimum_threshold = minimum
            print(f"✓ Minimum threshold set to: {minimum}")
    
    
    def set_scale_range(self, min_scale=None, max_scale=None, step=None):
        """
        Adjust multi-scale parameters for tuning.
        
        Args:
            min_scale (float): Minimum scale to test (e.g., 0.4)
            max_scale (float): Maximum scale to test (e.g., 0.6)
            step (float): Scale increment (e.g., 0.05)
        """
        if min_scale is not None and max_scale is not None:
            self.scale_range = (min_scale, max_scale)
            print(f"✓ Scale range set to: {min_scale}-{max_scale}")
        
        if step is not None:
            self.scale_step = step
            print(f"✓ Scale step set to: {step}")


def diagnose_detection_issue(region_bgr, detector):
    """
    Detailed diagnostic function to understand detection failures.
    Tests ALL weapons at ALL scales without early stopping.
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC MODE - COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # 1. Check region properties
    print(f"\nRegion Analysis:")
    print(f"  Size: {region_bgr.shape[1]}x{region_bgr.shape[0]}")
    print(f"  Mean brightness: {np.mean(region_bgr):.1f}")
    print(f"  Color channels: B={np.mean(region_bgr[:,:,0]):.1f}, "
          f"G={np.mean(region_bgr[:,:,1]):.1f}, R={np.mean(region_bgr[:,:,2]):.1f}")
    
    # 2. Extract white pixels
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixel_ratio = np.sum(white_mask > 0) / white_mask.size
    print(f"  White pixel ratio: {white_pixel_ratio:.3f}")
    
    # 3. Check ALL weapons at ALL scales without early stopping
    print(f"\nChecking ALL weapons at ALL scales:")
    print(f"Scale range: {detector.scale_range[0]}-{detector.scale_range[1]}, step: {detector.scale_step}")
    
    all_scores = []
    
    for weapon_name in detector.icons.keys():
        template = detector.icons[weapon_name]
        result = detector._match_template_multiscale(region_bgr, template, weapon_name)
        
        if result is not None:
            confidence, scale, position = result
            all_scores.append((weapon_name, confidence, scale, position))
    
    # Sort by confidence
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 15 matches (with scale and position):")
    for i, (weapon, conf, scale, pos) in enumerate(all_scores[:15], 1):
        status = "✓✓" if conf >= 0.70 else "✗"
        print(f"  {i:2d}. {status} {weapon:12s}: {conf:.4f} @ {scale:.2f}x scale, pos=({pos[0]}, {pos[1]})")
    
    # 4. Check for ambiguous matches
    if len(all_scores) >= 2:
        top1_conf = all_scores[0][1]
        top2_conf = all_scores[1][1]
        diff = top1_conf - top2_conf
        
        print(f"\nAmbiguity Analysis:")
        print(f"  Top match: {all_scores[0][0]} ({top1_conf:.4f} @ {all_scores[0][2]:.2f}x)")
        print(f"  2nd match: {all_scores[1][0]} ({top2_conf:.4f} @ {all_scores[1][2]:.2f}x)")
        print(f"  Difference: {diff:.4f}")
        
        if diff < 0.05:
            print(f"  ⚠️ WARNING: Very close match! Potential misidentification.")
        elif diff < 0.15:
            print(f"  ⚠️ Moderately ambiguous match.")
        else:
            print(f"  ✓ Clear winner.")
    
    print("="*60 + "\n")
    
    return all_scores


# ========================================
# CONVENIENCE FUNCTION
# ========================================

def detect_weapon_icon(red_rect_region, detector, check_only=None):
    """
    Convenience wrapper for detect_weapon_in_region.
    
    Args:
        red_rect_region (numpy.ndarray): The region inside a detected red rectangle
        detector (WeaponIconDetector): The detector instance
        check_only (list): DEPRECATED - Now always checks all weapons
    
    Returns:
        tuple: (weapon_name: str or None, confidence: float)
    """
    weapon, confidence, debug_info = detector.detect_weapon_in_region(red_rect_region, check_only)
    return weapon, confidence


# ========================================
# TESTING / STANDALONE EXECUTION
# ========================================

def test_icon_detection():
    """
    Test the icon detection system with sample images.
    Run this file directly to test: python icondetection.py
    """
    print("="*60)
    print("ICON DETECTION TEST - MULTI-SCALE VERSION (0.7+ THRESHOLD)")
    print("="*60)
    
    # Initialize detector - uses default path automatically
    detector = WeaponIconDetector(debug=True)
    
    if len(detector.icons) == 0:
        print("\n❌ No icons loaded. Cannot test.")
        print("   Please add weapon icon PNGs to the 'CS2 Icons/Weapons' folder")
        return
    
    print("\n" + "="*60)
    print("TEST 1: Check loaded icons")
    print("="*60)
    print(f"Loaded icons: {list(detector.icons.keys())}")
    
    print("\n" + "="*60)
    print("TEST 2: Multi-scale parameters")
    print("="*60)
    print(f"Scale range: {detector.scale_range[0]}-{detector.scale_range[1]}")
    print(f"Scale step: {detector.scale_step}")
    scales = np.arange(detector.scale_range[0], detector.scale_range[1] + detector.scale_step, detector.scale_step)
    print(f"Scales to test: {[f'{s:.2f}' for s in scales]}")
    
    print("\n" + "="*60)
    print("CHANGES APPLIED:")
    print("="*60)
    print("  ✓ Threshold lowered to 0.7 (70% certainty)")
    print("  ✓ Always checks ALL weapons (no early stopping)")
    print("  ✓ Returns highest confidence match >= 0.7")
    
    print("\n" + "="*60)
    print("To test with real images:")
    print("="*60)
    print("  1. Take a screenshot of the killfeed with a weapon kill")
    print("  2. Crop the red rectangle region containing the weapon icon")
    print("  3. Save it as 'test_region.png' in the CS2Clipper folder")
    print("  4. Load and test:")
    print("     region = cv2.imread('test_region.png')")
    print("     detector.detect_weapon_in_region(region)")
    print("="*60)


if __name__ == "__main__":
    test_icon_detection()