import cv2
import numpy as np
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
from datetime import datetime
import uuid

class FingerprintMatcher:
    def __init__(self):
        # Get the NBIS installation path
        self.nbis_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nbis')
        self.mindtct_path = os.path.join(self.nbis_path, 'mindtct', 'bin', 'mindtct.exe')
        self.bozorth3_path = os.path.join(self.nbis_path, 'bozorth3', 'bin', 'bozorth3.exe')
        
        # Check if NBIS is available
        self.nbis_available = os.path.exists(self.mindtct_path) and os.path.exists(self.bozorth3_path)
        if not self.nbis_available:
            print("Warning: NBIS executables not found. Falling back to OpenCV.")
    
    def _read_image(self, image_path):
        """Read and validate image"""
        try:
            # Convert path to string and normalize
            image_path = str(Path(image_path).resolve())
            print(f"Reading image from: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                raise Exception(f"Image file does not exist: {image_path}")
            
            # Try reading with PIL first
            try:
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                print("Image read successfully using PIL")
            except Exception as pil_error:
                print(f"PIL reading failed: {str(pil_error)}")
                # Fallback to OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    raise Exception(f"Failed to read image with OpenCV: {image_path}")
                print("Image read successfully using OpenCV")
            
            # Validate image
            if image.size == 0:
                raise Exception("Image is empty")
            
            print(f"Image shape: {image.shape}")
            print(f"Image type: {image.dtype}")
            print(f"Image min/max values: {image.min()}/{image.max()}")
            
            return image
        except Exception as e:
            print(f"Error reading image: {str(e)}")
            raise
    
    def _preprocess_image(self, image):
        """Preprocess image for better feature detection"""
        try:
            print("Starting image preprocessing...")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize image if too small
            min_size = 300
            if min(image.shape) < min_size:
                scale = min_size / min(image.shape)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            # Apply histogram equalization
            image = cv2.equalizeHist(image)
            
            # Apply bilateral filter to preserve edges while removing noise
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Apply adaptive thresholding
            image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to enhance ridges
            kernel = np.ones((3,3), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            # Apply distance transform
            dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
            image = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            
            print("Image preprocessing completed")
            print(f"Preprocessed image shape: {image.shape}")
            print(f"Preprocessed image type: {image.dtype}")
            print(f"Preprocessed image min/max values: {image.min()}/{image.max()}")
            
            return image
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def match_fingerprints(self, image1, image2, method='nbis'):
        """
        Match fingerprints using NBIS or OpenCV as fallback
        """
        print(f"\n=== Starting fingerprint matching ===")
        print(f"Image 1: {image1}")
        print(f"Image 2: {image2}")
        print(f"Method: {method}")
        
        if method == 'nbis' and self.nbis_available:
            try:
                # Create temporary directory for NBIS files
                temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
                os.makedirs(temp_dir, exist_ok=True)
                
                # Generate unique filenames for temporary files
                base1 = os.path.join(temp_dir, 'fp1')
                base2 = os.path.join(temp_dir, 'fp2')
                
                print(f"\nRunning mindtct on {image1}")
                # Run mindtct to extract minutiae
                result1 = subprocess.run([self.mindtct_path, image1, base1], capture_output=True, text=True)
                print(f"mindtct output for image 1:\n{result1.stdout}\n{result1.stderr}")
                
                print(f"\nRunning mindtct on {image2}")
                result2 = subprocess.run([self.mindtct_path, image2, base2], capture_output=True, text=True)
                print(f"mindtct output for image 2:\n{result2.stdout}\n{result2.stderr}")
                
                print("\nRunning bozorth3 for matching")
                # Run bozorth3 to match fingerprints
                result = subprocess.run(
                    [self.bozorth3_path, '-m1', '-A', '-p', base1 + '.xyt', base2 + '.xyt'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                print(f"bozorth3 output:\n{result.stdout}\n{result.stderr}")
                
                # Parse the score from bozorth3 output
                score = float(result.stdout.strip())
                print(f"NBIS matching score: {score}")
                
                # Convert score to percentage (NBIS typically returns 0-100)
                return min(score, 100)
                
            except Exception as e:
                print(f"NBIS matching failed: {str(e)}")
                print("Falling back to OpenCV matching")
                return self._match_with_opencv(image1, image2)
            finally:
                # Clean up temporary files
                try:
                    for base in [base1, base2]:
                        for ext in ['.xyt', '.brw', '.dm', '.hcm', '.lcm', '.lfm', '.min']:
                            file_path = os.path.join(os.path.dirname(base), os.path.basename(base) + ext)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                except Exception as e:
                    print(f"Cleanup failed: {str(e)}")
        else:
            print("Using OpenCV for matching")
            return self._match_with_opencv(image1, image2)
    
    def _match_with_opencv(self, image1, image2):
        """
        Match fingerprints using OpenCV SIFT
        """
        try:
            print("\n=== OpenCV Matching Process ===")
            
            # Read images
            img1 = self._read_image(image1)
            img2 = self._read_image(image2)
            
            # Preprocess images
            img1 = self._preprocess_image(img1)
            img2 = self._preprocess_image(img2)
            
            print(f"Image 1 shape: {img1.shape}")
            print(f"Image 2 shape: {img2.shape}")
            
            # Initialize SIFT detector with custom parameters
            sift = cv2.SIFT.create(
                nfeatures=0,
                nOctaveLayers=3,
                contrastThreshold=0.01,  # Lower threshold to detect more features
                edgeThreshold=30,        # Higher threshold to focus on strong edges
                sigma=1.6
            )
            
            # Create empty mask
            mask = np.zeros(img1.shape, dtype=np.uint8)
            
            # Find keypoints and descriptors
            kp1, des1 = sift.detectAndCompute(img1, mask)
            kp2, des2 = sift.detectAndCompute(img2, mask)
            
            print(f"Number of keypoints in image 1: {len(kp1)}")
            print(f"Number of keypoints in image 2: {len(kp2)}")
            
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                print("No descriptors found in one or both images")
                return 0
            
            # FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params: Dict[str, Union[int, float, bool, str]] = {
                'algorithm': FLANN_INDEX_KDTREE,
                'trees': 5
            }
            search_params: Dict[str, Union[int, float, bool, str]] = {
                'checks': 50
            }
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Find matches
            matches = flann.knnMatch(des1, des2, k=2)
            print(f"Total matches found: {len(matches)}")
            
            # Apply ratio test with more lenient threshold
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:  # More lenient ratio test
                    good_matches.append(m)
            
            print(f"Good matches after ratio test: {len(good_matches)}")
            
            # Calculate score
            score = (len(good_matches) / max(len(kp1), len(kp2))) * 100
            print(f"Final OpenCV matching score: {score}")
            return min(score, 100)
            
        except Exception as e:
            print(f"OpenCV matching failed: {str(e)}")
            return 0
    
    def visualize_minutiae(self, image):
        """
        Visualize minutiae points on the fingerprint image using OpenCV
        """
        try:
            # Read and preprocess image
            if isinstance(image, str):
                image = self._read_image(image)
            
            processed = self._preprocess_image(image)
            
            # Initialize SIFT detector with custom parameters
            sift = cv2.SIFT.create(
                nfeatures=0,
                nOctaveLayers=3,
                contrastThreshold=0.02,
                edgeThreshold=20,
                sigma=1.6
            )
            
            # Create empty mask
            mask = np.zeros(processed.shape, dtype=np.uint8)
            
            # Detect keypoints
            keypoints = sift.detect(processed, mask)
            
            # Create color image for visualization
            if len(image.shape) == 2:
                vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                vis_image = image.copy()
            
            # Draw keypoints
            for kp in keypoints:
                x, y = map(int, kp.pt)
                cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
                # Draw direction line
                angle = kp.angle * np.pi / 180.0
                length = 10
                end_x = int(x + length * np.cos(angle))
                end_y = int(y + length * np.sin(angle))
                cv2.line(vis_image, (x, y), (end_x, end_y), (0, 255, 0), 1)
            
            return vis_image
            
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
            return None

def get_matching_result(score):
    """Get matching result based on score"""
    if score >= 80:
        return "مطابقة عالية", "success"
    elif score >= 60:
        return "مطابقة متوسطة", "warning"
    else:
        return "لا يوجد تطابق", "danger"

def match_fingerprint(img1_path, img2_path, results_folder):
    """Match two fingerprint images using OpenCV"""
    try:
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images")
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0.0, 0, 0, 0, None, None, None, 0
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Find matches
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Calculate match score
        score = (len(good_matches) / max(len(kp1), len(kp2))) * 100 if max(len(kp1), len(kp2)) > 0 else 0
        
        # Create visualizations
        minutiae1 = visualize_minutiae(img1, kp1)
        minutiae2 = visualize_minutiae(img2, kp2)
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Generate filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        
        minutiae1_filename = f"minutiae1_{timestamp}_{unique_id}.jpg"
        minutiae2_filename = f"minutiae2_{timestamp}_{unique_id}.jpg"
        match_filename = f"match_{timestamp}_{unique_id}.jpg"
        
        # Ensure results directory exists
        os.makedirs(results_folder, exist_ok=True)
        
        # Save images with full paths
        minutiae1_path = os.path.join(results_folder, minutiae1_filename)
        minutiae2_path = os.path.join(results_folder, minutiae2_filename)
        match_path = os.path.join(results_folder, match_filename)
        
        # Save images and verify
        cv2.imwrite(minutiae1_path, minutiae1)
        cv2.imwrite(minutiae2_path, minutiae2)
        cv2.imwrite(match_path, match_img)
        
        # Verify files were created
        if not all(os.path.exists(p) for p in [minutiae1_path, minutiae2_path, match_path]):
            raise Exception("Failed to save one or more visualization images")
        
        print(f"Saved images to: {results_folder}")
        print(f"Files: {minutiae1_filename}, {minutiae2_filename}, {match_filename}")
        
        return score, len(kp1), len(kp2), len(good_matches), match_filename, minutiae1_filename, minutiae2_filename, 0
        
    except Exception as e:
        print(f"Error in match_fingerprint: {str(e)}")
        return 0.0, 0, 0, 0, None, None, None, 0

def visualize_minutiae(image, keypoints):
    """Visualize minutiae points on the image"""
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw keypoints
    for kp in keypoints:
        x, y = map(int, kp.pt)
        cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
        # Draw direction line
        angle = kp.angle * np.pi / 180.0
        length = 10
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))
        cv2.line(vis_image, (x, y), (end_x, end_y), (0, 255, 0), 1)
    
    return vis_image 