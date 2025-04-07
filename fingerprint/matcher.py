import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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