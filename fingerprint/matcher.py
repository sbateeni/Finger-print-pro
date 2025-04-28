import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import cv2
import math
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

def align_fingerprints(points1, points2):
    """
    محاذاة البصمتين باستخدام التحويل الأمثل
    """
    if not points1 or not points2:
        return None, None
    
    # تحويل النقاط إلى مصفوفات
    coords1 = np.array([[p.x, p.y] for p in points1])
    coords2 = np.array([[p.x, p.y] for p in points2])
    
    # حساب مركز الكتلة
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    
    # إزاحة النقاط إلى المركز
    centered1 = coords1 - center1
    centered2 = coords2 - center2
    
    # حساب زوايا النقاط
    angles1 = np.arctan2(centered1[:, 1], centered1[:, 0])
    angles2 = np.arctan2(centered2[:, 1], centered2[:, 0])
    
    # حساب متوسط فرق الزوايا
    angle_diff = np.mean(angles1) - np.mean(angles2)
    
    # تدوير النقاط
    rotation_matrix = np.array([
        [np.cos(angle_diff), -np.sin(angle_diff)],
        [np.sin(angle_diff), np.cos(angle_diff)]
    ])
    
    aligned2 = np.dot(centered2, rotation_matrix.T) + center1
    
    return coords1, aligned2

def compute_similarity(point1, point2):
    """
    حساب درجة التشابه بين نقطتين
    """
    # حساب المسافة المكانية
    spatial_dist = np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    # حساب الفرق في الزاوية
    angle_diff = abs(point1.angle - point2.angle)
    angle_diff = min(angle_diff, 2*np.pi - angle_diff)
    
    # حساب درجة التشابه المركبة
    similarity = np.exp(-spatial_dist/50.0) * np.exp(-angle_diff/np.pi)
    
    return similarity

def find_matching_points(points1, points2, threshold=0.7):
    """
    العثور على النقاط المتطابقة بين البصمتين
    """
    if not points1 or not points2:
        return [], 0.0
    
    # إنشاء مصفوفة التكلفة
    cost_matrix = np.zeros((len(points1), len(points2)))
    
    for i, p1 in enumerate(points1):
        for j, p2 in enumerate(points2):
            # حساب درجة التشابه
            similarity = compute_similarity(p1, p2)
            # تحويل التشابه إلى تكلفة
            cost_matrix[i, j] = 1 - similarity
    
    # تطبيق خوارزمية المطابقة المجرية
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # تجميع النقاط المتطابقة
    matching_pairs = []
    total_similarity = 0
    
    for i, j in zip(row_ind, col_ind):
        similarity = 1 - cost_matrix[i, j]
        if similarity >= threshold:
            matching_pairs.append((points1[i], points2[j]))
            total_similarity += similarity
    
    # حساب درجة التطابق الإجمالية
    if matching_pairs:
        match_score = total_similarity / len(matching_pairs)
    else:
        match_score = 0.0
    
    return matching_pairs, match_score

def compare_fingerprints(features1, features2):
    """
    مقارنة البصمتين وحساب درجة التطابق
    """
    # محاذاة البصمتين
    coords1, aligned2 = align_fingerprints(features1, features2)
    
    if coords1 is None:
        return 0.0, []
    
    # العثور على النقاط المتطابقة
    matching_pairs, match_score = find_matching_points(features1, features2)
    
    # تحويل النقاط المتطابقة إلى الشكل المطلوب للعرض
    matching_points = []
    for p1, p2 in matching_pairs:
        matching_points.append([
            {'x': int(p1.x), 'y': int(p1.y)},
            {'x': int(p2.x), 'y': int(p2.y)}
        ])
    
    return match_score, matching_points

def match_minutiae_points(points1, points2, distance_threshold=30, angle_threshold=np.pi/6):
    """
    Match minutiae points between two sets based on position and angle
    """
    if not points1 or not points2:
        return []

    # Create coordinate arrays
    coords1 = np.array([[p.x, p.y] for p in points1])
    coords2 = np.array([[p.x, p.y] for p in points2])

    # Calculate pairwise distances
    distances = cdist(coords1, coords2)

    # Calculate angle differences
    angles1 = np.array([p.angle for p in points1])
    angles2 = np.array([p.angle for p in points2])
    angle_diffs = np.abs(angles1[:, np.newaxis] - angles2)
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)

    # Find valid matches
    matches = []
    used2 = set()

    for i in range(len(points1)):
        best_j = -1
        best_dist = float('inf')

        for j in range(len(points2)):
            if j in used2:
                continue

            if distances[i,j] <= distance_threshold and angle_diffs[i,j] <= angle_threshold:
                if distances[i,j] < best_dist:
                    best_dist = distances[i,j]
                    best_j = j

        if best_j != -1:
            matches.append((points1[i], points2[best_j]))
            used2.add(best_j)

    return matches

def calculate_spatial_consistency(matches):
    """
    Calculate how well the spatial relationships between matching points are preserved
    """
    if len(matches) < 2:
        return 0.0

    # Calculate pairwise distances between points in first set
    coords1 = np.array([[p1.x, p1.y] for p1, _ in matches])
    dists1 = cdist(coords1, coords1)

    # Calculate pairwise distances between corresponding points in second set
    coords2 = np.array([[p2.x, p2.y] for _, p2 in matches])
    dists2 = cdist(coords2, coords2)

    # Compare distance matrices
    diff = np.abs(dists1 - dists2)
    max_diff = np.maximum(dists1, dists2)
    where_nonzero = max_diff > 0
    if not where_nonzero.any():
        return 1.0

    relative_diff = diff[where_nonzero] / max_diff[where_nonzero]
    consistency_score = 1.0 - np.mean(relative_diff)

    return max(0.0, min(1.0, consistency_score))

def calculate_match_quality(matching_pairs):
    """
    Calculate additional quality metrics for the match
    """
    if not matching_pairs:
        return {
            'num_matches': 0,
            'avg_distance': 0,
            'avg_angle_diff': 0
        }
    
    distances = []
    angle_diffs = []
    
    for p1, p2 in matching_pairs:
        # Calculate spatial distance
        distance = cdist([[p1.x, p1.y]], [[p2.x, p2.y]])[0][0]
        distances.append(distance)
        
        # Calculate angle difference
        angle_diff = abs(p1.angle - p2.angle)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        angle_diffs.append(angle_diff)
    
    return {
        'num_matches': len(matching_pairs),
        'avg_distance': np.mean(distances),
        'avg_angle_diff': np.mean(angle_diffs)
    }

def draw_matching_lines(image1, image2, matching_pairs):
    """
    رسم خطوط التطابق بين البصمتين
    """
    # تحديد أبعاد الصورة المجمعة
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2 + 20  # إضافة مسافة بين الصورتين
    
    # إنشاء صورة مجمعة
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # نسخ الصورتين إلى الصورة المجمعة
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    result[0:h1, 0:w1] = image1
    result[0:h2, w1+20:] = image2
    
    # رسم خطوط التطابق
    colors = [
        (255, 0, 0),   # أزرق
        (0, 255, 0),   # أخضر
        (255, 0, 255), # وردي
        (0, 255, 255), # أصفر
        (128, 0, 255), # بنفسجي
        (255, 128, 0)  # برتقالي
    ]
    
    for i, (p1, p2) in enumerate(matching_pairs):
        color = colors[i % len(colors)]
        x1, y1 = int(p1.x), int(p1.y)
        x2, y2 = int(p2.x) + w1 + 20, int(p2.y)
        
        # رسم دوائر عند نقاط التطابق
        cv2.circle(result, (x1, y1), 5, color, -1)
        cv2.circle(result, (x2, y2), 5, color, -1)
        
        # رسم خط التطابق
        cv2.line(result, (x1, y1), (x2, y2), color, 2)
        
        # إضافة رقم للنقطة المتطابقة
        cv2.putText(result, str(i+1), (x1-10, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(result, str(i+1), (x2-10, y2-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result 

class FingerprintMatcher:
    def __init__(self):
        self.distance_threshold = 50.0
        self.angle_threshold = 30.0
        self.min_matching_points = 10
        self.distance_weight = 0.7
        self.angle_weight = 0.3
        self.quality_weight = 0.5
        self.spatial_weight = 0.4
        self.ridge_weight = 0.3
        self.minutiae_weight = 0.3
        self.min_ridge_distance = 5
        self.max_ridge_distance = 15
        self.min_quality_score = 0.6
        self.max_rotation = 30.0
        self.max_scale = 1.2

    def match_fingerprints(self, features1, features2, threshold=0.6):
        """مطابقة بصمتين باستخدام النقاط المميزة"""
        try:
            # استخراج النقاط المميزة
            points1 = features1.get('minutiae_points', [])
            points2 = features2.get('minutiae_points', [])
            
            if not points1 or not points2:
                return 0.0, []
            
            # حساب مصفوفة المسافات والزوايا
            matches = []
            for p1 in points1:
                best_match = None
                min_distance = float('inf')
                
                for p2 in points2:
                    # التحقق من نوع النقطة
                    if p1.get('type') != p2.get('type'):
                        continue
                    
                    # حساب المسافة الإقليدية
                    dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                    
                    # حساب فرق الزاوية
                    angle_diff = abs(p1.get('angle', 0) - p2.get('angle', 0))
                    angle_diff = min(angle_diff, 360 - angle_diff)  # أخذ أصغر فرق زاوية
                    
                    # حساب درجة التطابق بناءً على المسافة والزاوية
                    match_score = 1.0
                    if dist > 20:  # تجاهل النقاط البعيدة جداً
                        continue
                    match_score *= (1 - dist/20)  # تقليل درجة التطابق مع زيادة المسافة
                    
                    if angle_diff > 30:  # تجاهل النقاط بفرق زاوية كبير
                        continue
                    match_score *= (1 - angle_diff/30)  # تقليل درجة التطابق مع زيادة فرق الزاوية
                    
                    # تحديث أفضل تطابق
                    if match_score > min_distance:
                        min_distance = match_score
                        best_match = (p1, p2, match_score)
                
                if best_match and min_distance > threshold:
                    matches.append(best_match)
            
            # حساب درجة التطابق النهائية
            if not matches:
                return 0.0, []
            
            # حساب متوسط درجة التطابق
            match_score = np.mean([m[2] for m in matches])
            
            # حساب نسبة النقاط المتطابقة
            match_ratio = len(matches) / min(len(points1), len(points2))
            
            # الجمع بين درجة التطابق ونسبة النقاط المتطابقة
            final_score = match_score * match_ratio
            
            return final_score, matches
            
        except Exception as e:
            logger.error(f'خطأ في مطابقة البصمات: {str(e)}')
            return 0.0, []

    def _align_fingerprints(self, features1: List[Dict], features2: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        محاذاة البصمتين
        """
        try:
            # تحويل النقاط إلى مصفوفات
            coords1 = np.array([[f['x'], f['y']] for f in features1])
            coords2 = np.array([[f['x'], f['y']] for f in features2])

            # حساب مركز الكتلة
            center1 = np.mean(coords1, axis=0)
            center2 = np.mean(coords2, axis=0)

            # إزاحة النقاط إلى المركز
            centered1 = coords1 - center1
            centered2 = coords2 - center2

            # حساب زوايا النقاط
            angles1 = np.array([f['angle'] for f in features1])
            angles2 = np.array([f['angle'] for f in features2])

            # حساب متوسط فرق الزوايا
            angle_diff = np.mean(angles1) - np.mean(angles2)
            angle_diff = min(angle_diff, 360 - angle_diff)

            # تدوير النقاط
            rotation_matrix = np.array([
                [np.cos(np.radians(angle_diff)), -np.sin(np.radians(angle_diff))],
                [np.sin(np.radians(angle_diff)), np.cos(np.radians(angle_diff))]
            ])

            aligned2 = np.dot(centered2, rotation_matrix.T) + center1

            # تحديث إحداثيات النقاط
            aligned_features1 = features1.copy()
            aligned_features2 = []
            for i, f in enumerate(features2):
                aligned_f = f.copy()
                aligned_f['x'] = aligned2[i, 0]
                aligned_f['y'] = aligned2[i, 1]
                aligned_f['angle'] = (f['angle'] + angle_diff) % 360
                aligned_features2.append(aligned_f)

            return aligned_features1, aligned_features2

        except Exception as e:
            logger.error(f'خطأ في محاذاة البصمات: {str(e)}')
            raise

    def _create_distance_matrix(self, features1: List[Dict], features2: List[Dict]) -> np.ndarray:
        """
        إنشاء مصفوفة المسافات بين المميزات
        """
        try:
            n_features1 = len(features1)
            n_features2 = len(features2)
            distance_matrix = np.zeros((n_features1, n_features2))

            for i, f1 in enumerate(features1):
                for j, f2 in enumerate(features2):
                    distance_matrix[i, j] = self._calculate_feature_distance(f1, f2)

            return distance_matrix

        except Exception as e:
            logger.error(f'خطأ في إنشاء مصفوفة المسافات: {str(e)}')
            raise

    def _calculate_feature_distance(self, feature1: Dict, feature2: Dict) -> float:
        """
        حساب المسافة بين مميزتين
        """
        try:
            # حساب المسافة المكانية
            spatial_distance = math.sqrt(
                (feature1['x'] - feature2['x'])**2 +
                (feature1['y'] - feature2['y'])**2
            )

            # حساب الفرق في الزوايا
            angle_diff = min(
                abs(feature1['angle'] - feature2['angle']),
                360 - abs(feature1['angle'] - feature2['angle'])
            )

            # حساب المسافة بين الخطوط
            ridge_distance = self._calculate_ridge_distance(feature1, feature2)

            # حساب المسافة الكلية (مرجحة)
            total_distance = (
                self.spatial_weight * spatial_distance +
                self.angle_weight * angle_diff +
                self.ridge_weight * ridge_distance
            )

            # تطبيق وزن الجودة
            if 'quality' in feature1 and 'quality' in feature2:
                quality_factor = (
                    self.quality_weight * feature1['quality'] +
                    self.quality_weight * feature2['quality']
                )
                total_distance *= (1 - quality_factor)

            return total_distance

        except Exception as e:
            logger.error(f'خطأ في حساب المسافة بين المميزات: {str(e)}')
            raise

    def _calculate_ridge_distance(self, feature1: Dict, feature2: Dict) -> float:
        """
        حساب المسافة بين الخطوط
        """
        try:
            # حساب المسافة بين الخطوط
            ridge_distance = abs(feature1.get('ridge_distance', 0) - feature2.get('ridge_distance', 0))

            # تطبيع المسافة
            if ridge_distance < self.min_ridge_distance:
                ridge_distance = 0
            elif ridge_distance > self.max_ridge_distance:
                ridge_distance = 1
            else:
                ridge_distance = (ridge_distance - self.min_ridge_distance) / (self.max_ridge_distance - self.min_ridge_distance)

            return ridge_distance

        except Exception as e:
            logger.error(f'خطأ في حساب المسافة بين الخطوط: {str(e)}')
            raise

    def _calculate_match_score(self, distance_matrix: np.ndarray,
                             row_ind: np.ndarray, col_ind: np.ndarray) -> float:
        """
        حساب درجة التطابق
        """
        try:
            # حساب المسافة الإجمالية للنقاط المتطابقة
            total_distance = distance_matrix[row_ind, col_ind].sum()

            # حساب الحد الأقصى للمسافة الممكنة
            max_possible_distance = len(row_ind) * 100.0

            # حساب درجة التطابق
            match_score = 1.0 - (total_distance / max_possible_distance)

            # تطبيق معامل الجودة
            if 'quality' in distance_matrix:
                quality_factor = np.mean([
                    distance_matrix[i, j]['quality']
                    for i, j in zip(row_ind, col_ind)
                ])
                match_score *= (1.0 + quality_factor)

            return max(0.0, min(1.0, match_score))

        except Exception as e:
            logger.error(f'خطأ في حساب درجة التطابق: {str(e)}')
            raise

    def _filter_matching_points(self, features1: List[Dict], features2: List[Dict],
                              row_ind: np.ndarray, col_ind: np.ndarray,
                              distance_matrix: np.ndarray) -> List[Tuple[Dict, Dict]]:
        """
        تصفية النقاط المتطابقة
        """
        try:
            matching_points = []

            for i, j in zip(row_ind, col_ind):
                if distance_matrix[i, j] < self.distance_threshold:
                    f1 = features1[i]
                    f2 = features2[j]

                    # التحقق من أن النقاط ليست على الحافة
                    if self._is_on_edge(f1) or self._is_on_edge(f2):
                        continue

                    # التحقق من أن النقاط ليست في منطقة مزدحمة
                    if self._is_crowded(f1, features1) or self._is_crowded(f2, features2):
                        continue

                    # التحقق من جودة النقاط
                    if 'quality' in f1 and 'quality' in f2:
                        if f1['quality'] < self.min_quality_score or f2['quality'] < self.min_quality_score:
                            continue

                    matching_points.append((f1, f2))

            return matching_points

        except Exception as e:
            logger.error(f'خطأ في تصفية النقاط المتطابقة: {str(e)}')
            raise

    def _is_on_edge(self, feature: Dict) -> bool:
        """
        التحقق من أن النقطة على الحافة
        """
        try:
            # التحقق من أن النقطة ليست قريبة من الحواف
            if (feature['x'] < self.distance_threshold or
                feature['y'] < self.distance_threshold):
                return True

            return False

        except Exception as e:
            logger.error(f'خطأ في التحقق من الحافة: {str(e)}')
            raise

    def _is_crowded(self, feature: Dict, all_features: List[Dict]) -> bool:
        """
        التحقق من أن النقطة في منطقة مزدحمة
        """
        try:
            count = 0
            for other_feature in all_features:
                if feature == other_feature:
                    continue

                distance = math.sqrt(
                    (feature['x'] - other_feature['x'])**2 +
                    (feature['y'] - other_feature['y'])**2
                )

                if distance < self.distance_threshold:
                    count += 1

            return count > 2

        except Exception as e:
            logger.error(f'خطأ في التحقق من الازدحام: {str(e)}')
            raise

    def _calculate_quality_score(self, matching_points: List[Tuple[Dict, Dict]]) -> float:
        """
        حساب درجة جودة التطابق
        """
        try:
            if not matching_points:
                return 0.0

            # حساب متوسط جودة النقاط المتطابقة
            quality_scores = []
            for f1, f2 in matching_points:
                if 'quality' in f1 and 'quality' in f2:
                    quality_scores.append((f1['quality'] + f2['quality']) / 2)

            if not quality_scores:
                return 0.0

            return np.mean(quality_scores)

        except Exception as e:
            logger.error(f'خطأ في حساب درجة الجودة: {str(e)}')
            raise

    def match_features(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, Any]:
        """مطابقة المميزات بين صورتين"""
        try:
            # التحقق من وجود نقاط كافية
            count1 = len(features1['minutiae'])
            count2 = len(features2['minutiae'])
            
            if count1 < self.min_points or count2 < self.min_points:
                raise ValueError("لا توجد نقاط مميزة كافية في إحدى الصورتين")
            
            # استخراج النقاط المميزة
            minutiae1 = features1['minutiae']
            minutiae2 = features2['minutiae']
            
            # حساب مصفوفة المسافات
            distances = self._calculate_distances(minutiae1, minutiae2)
            
            # تطبيق خوارزمية المطابقة
            matches = self._match_minutiae(distances)
            
            # حساب درجة التطابق
            score = self._calculate_score(matches, count1, count2)
            
            return {
                'score': score,
                'matches': matches,
                'count1': count1,
                'count2': count2
            }
            
        except Exception as e:
            logger.error(f'خطأ في مطابقة المميزات: {str(e)}')
            raise

    def _calculate_distances(self, minutiae1: List[Tuple[int, int, str, float]], 
                           minutiae2: List[Tuple[int, int, str, float]]) -> np.ndarray:
        """حساب مصفوفة المسافات بين النقاط المميزة"""
        n1 = len(minutiae1)
        n2 = len(minutiae2)
        distances = np.zeros((n1, n2))
        
        for i, m1 in enumerate(minutiae1):
            for j, m2 in enumerate(minutiae2):
                # حساب المسافة الإقليدية
                dist = np.sqrt((m1[0] - m2[0])**2 + (m1[1] - m2[1])**2)
                
                # حساب فرق الزاوية
                angle_diff = abs(m1[3] - m2[3])
                angle_diff = min(angle_diff, 360 - angle_diff)
                
                # حساب المسافة النهائية
                distances[i,j] = dist + self.angle_weight * angle_diff
                
        return distances

    def _match_minutiae(self, distances: np.ndarray) -> List[Tuple[int, int]]:
        """مطابقة النقاط المميزة باستخدام خوارزمية المجاورة القريبة"""
        matches = []
        used1 = set()
        used2 = set()
        
        # ترتيب المسافات
        sorted_distances = []
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                sorted_distances.append((distances[i,j], i, j))
        sorted_distances.sort()
        
        # تطبيق خوارزمية المجاورة القريبة
        for dist, i, j in sorted_distances:
            if i in used1 or j in used2:
                continue
                
            if dist > self.max_distance:
                break
                
            matches.append((i, j))
            used1.add(i)
            used2.add(j)
            
        return matches

    def _calculate_score(self, matches: List[Tuple[int, int]], count1: int, count2: int) -> float:
        """حساب درجة التطابق"""
        if not matches:
            return 0.0
            
        # حساب نسبة النقاط المتطابقة
        match_ratio = len(matches) / min(count1, count2)
        
        # حساب درجة التطابق النهائية
        score = match_ratio * 100.0
        
        return min(max(score, 0.0), 100.0) 