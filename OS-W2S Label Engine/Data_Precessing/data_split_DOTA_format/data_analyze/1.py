import re
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 若未下载wordnet数据，请运行：
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# 假设此时categories为合并重复后的最终类别列表：
# 例如：
# categories = [
#     ("basketballcourt", 12328),
#     ("aircraft", 17030),
#     ("airplane", 63517),
#     ("building", 440942),
#     ...
# ]
#
# 请在此处填入您的最终列表（此代码只是示例，需您提供最终数据）：

categories = [
    ("aircraft", 17030),
    ("aircraft-hangar", 393),
    ("airplane", 63517),
    ("airport", 3081),
    ("awning-tricycle", 8583),
    ("barge", 293),
    ("baseball-diamond", 12233),
    ("baseballfield", 5818),
    ("basketball-court", 12328),
    # ("basketballcourt", 3223),
    ("bicycle", 23093),
    ("bridge", 27121),
    ("building", 440942),
    ("bus", 23092),
    ("car", 328403),
    ("cargo-car", 2476),
    ("cargo-plane", 1139),
    ("cargo-truck", 7858),
    ("cement-mixer", 408),
    ("chimney", 1680),
    ("construction-site", 2198),
    ("container", 172122),
    ("container-crane", 684),
    ("container-ship", 580),
    ("crane-truck", 226),
    ("crossroad", 13888),
    ("dam", 1050),
    ("damaged-building", 1622),
    ("dump-truck", 1833),
    ("engineering-vehicle", 292),
    ("excavator", 1105),
    ("expressway-service-area", 2165),
    ("expressway-toll-station", 1298),
    ("facility", 1588),
    ("ferry", 320),
    ("fishing-vessel", 1093),
    ("fixed-wing-aircraft", 97),
    ("flat-car", 183),
    ("front-loader/bulldozer", 824),
    ("golffield", 1086),
    ("ground-grader", 102),
    ("ground-track-field", 15132),
    # ("groundtrackfield", 3047),
    ("harbor", 31562),
    ("haul-truck", 436),
    ("helicopter", 3085),
    ("helipad", 312),
    ("hut/tent", 988),
    ("large-vehicle", 77811),
    ("locomotive", 160),
    ("maritime-vessel", 1047),
    ("mobile-crane", 481),
    ("motor", 73259),
    ("motorboat", 1925),
    ("oil-tanker", 153),
    ("oiltank", 5911),
    ("overpass", 3824),
    ("parking-lot", 10493),
    ("passenger-car", 2163),
    ("passenger-vehicle", 3982),
    ("pedestrian", 178746),
    ("people", 67874),
    ("pickup-truck", 1424),
    ("plane", 19236),
    ("playground", 600),
    ("pylon", 520),
    ("railway-vehicle", 24),
    ("reach-stacker", 95),
    ("roundabout", 1493),
    ("sailboat", 859),
    ("scraper/tractor", 105),
    ("shed", 1705),
    ("ship", 245149),
    ("shipping-container", 1966),
    ("shipping-container-lot", 3419),
    ("small-aircraft", 496),
    ("small-car", 272897),
    ("small-vehicle", 946418),
    ("soccer-ball-field", 1444),
    ("stadium", 1267),
    ("storage-tank", 72529),
    ("storagetank", 26403),
    ("straddle-carrier", 93),
    ("swimming-pool", 41567),
    ("t-junction", 13559),
    ("tank-car", 150),
    ("tennis-court", 29433),
    # ("tenniscourt", 12241),
    ("tower", 123),
    ("tower-crane", 237),
    ("trailer", 5449),
    ("trainstation", 1010),
    ("tricycle", 11681),
    ("truck", 44828),
    ("truck-tractor", 1075),
    ("truck-w/box", 4705),
    ("truck-w/flatbed", 1247),
    ("truck-w/liquid", 203),
    ("tugboat", 325),
    ("utility-truck", 4672),
    ("van", 53675),
    ("vehicle", 49340),
    ("vehicle-lot", 6549),
    ("windmill", 40546),
    ("yacht", 644)
]

fallback_synset = wn.synset('entity.n.01')

def get_main_synset(category_name):
    """
    尝试为类别名找到最匹配的WordNet synset。
    策略：
    - 按非字母字符进行分词。
    - 尝试为每个子词查找synset。
    - 若有多个子词的synset可选，根据实际领域知识制定优先规则。
    - 若无匹配，则使用fallback_synset。
    """
    parts = re.split(r'[^a-zA-Z]+', category_name.lower())
    parts = [p for p in parts if p]

    if not parts:
        return fallback_synset

    # 收集子词的synsets
    candidates = []
    for p in parts:
        ss = wn.synsets(p, pos=wn.NOUN)
        if ss:
            candidates.append((p, ss))

    if not candidates:
        return fallback_synset

    # 简化策略：如果只有一个子词有synset，返回该子词的第一个synset
    if len(candidates) == 1:
        return candidates[0][1][0]

    # 若有多个子词可选，您可在此处加入自定义逻辑。例如：
    # 优先选择表示场地、交通工具、建筑物等您关注的概念。
    # 此处示例：选择第一个候选synset作为默认策略
    return candidates[0][1][0]

# 为每个类别映射到synset
cat_names = []
cat_synsets = []
cat_counts = []

for (c_name, c_count) in categories:
    s = get_main_synset(c_name)
    cat_names.append(c_name)
    cat_synsets.append(s)
    cat_counts.append(c_count)

N = len(cat_names)

# 构建相似度矩阵
similarity_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            similarity_matrix[i, j] = 1.0
        elif i < j:
            sim = cat_synsets[i].wup_similarity(cat_synsets[j])
            if sim is None:
                sim = 0.0
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

distance_matrix = 1 - similarity_matrix

# 层次聚类
Z = linkage(squareform(distance_matrix, checks=False), method='average')

# 根据需要决定分簇数，这里以 N//2 为例，可根据需求调整
max_clust = N // 2 if N > 2 else 1
clusters = fcluster(Z, t=max_clust, criterion='maxclust')

cluster_dict = {}
for idx, cl in enumerate(clusters):
    cluster_dict.setdefault(cl, []).append(idx)

base_classes = []
novel_classes = []

# 相似度阈值，可适当降低以增加Novel类数量
similarity_threshold = 0.95

for cl, indices in cluster_dict.items():
    # 从当前簇中提取类别
    cluster_cats = [(cat_names[i], cat_counts[i], i) for i in indices]
    cluster_cats_sorted = sorted(cluster_cats, key=lambda x: x[1], reverse=True)

    # 第一个作为Base类
    base_class_name, base_class_count, base_idx = cluster_cats_sorted[0]
    base_classes.append(base_class_name)

    # 寻找符合相似度要求的Novel类
    # 可允许多项Novel类以增加其数量，将下面逻辑改为对所有达标的类加入Novel
    novel_candidates = []
    for (c_name, c_cnt, c_idx) in cluster_cats_sorted[1:]:
        sim = cat_synsets[base_idx].wup_similarity(cat_synsets[c_idx])
        if sim is not None and sim >= similarity_threshold:
            novel_candidates.append((c_name, sim))

    # 若想增加Novel类数量，可将novel_candidates全部加入Novel类
    # 这里将全部符合要求的添加为Novel类
    for (n_name, n_sim) in novel_candidates:
        novel_classes.append(n_name)

    # 将不在novel_candidates的剩余类别加入Base类
    novel_set = {x[0] for x in novel_candidates}
    for (c_name, c_cnt, c_idx) in cluster_cats_sorted[1:]:
        if c_name not in novel_set:
            base_classes.append(c_name)

# 去重
base_classes = list(set(base_classes))
novel_classes = list(set(novel_classes))

# 确保所有类都被覆盖
all_final_classes = set(base_classes) | set(novel_classes)
if len(all_final_classes) != N:
    print("Warning: Not all classes accounted for. Check logic.")
else:
    print("All categories accounted for.")

print("Number of categories:", N)
print("Number of Base classes:", len(base_classes))
print("Number of Novel classes:", len(novel_classes))
print("Total Base+Novel:", len(base_classes) + len(novel_classes))
print("\nBase Classes:")
print(base_classes)
print("\nNovel Classes:")
print(novel_classes)
