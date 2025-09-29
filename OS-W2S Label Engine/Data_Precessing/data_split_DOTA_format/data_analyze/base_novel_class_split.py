import nltk
from nltk.corpus import wordnet as wn
import re
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 若尚未下载WordNet数据，请先执行以下两行：
# nltk.download('wordnet')
# nltk.download('omw-1.4')



def normalize_category(name):
    # 去除非字母字符并转小写
    return re.sub(r'[^a-zA-Z]', '', name).lower()

# 合并同义重复类别
merged_dict = {}
for cat, count in raw_categories:
    norm = normalize_category(cat)
    if norm in merged_dict:
        merged_dict[norm] += count
    else:
        merged_dict[norm] = count

# 若找不到synset，为保证覆盖全部类别，使用"entity.n.01"作为回退
fallback_synset = wn.synset('entity.n.01')

def get_main_synset(category_name):
    synsets = wn.synsets(category_name, pos=wn.NOUN)
    if synsets:
        return synsets[0]
    # 没找到则返回回退synset
    return fallback_synset

categories = []
for c, cnt in merged_dict.items():
    s = get_main_synset(c)
    categories.append((c, cnt, s))

cat_names = [c[0] for c in categories]
cat_synsets = [c[2] for c in categories]
N = len(categories)

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

max_clust = N // 2 if N > 2 else 1
clusters = fcluster(Z, t=max_clust, criterion='maxclust')

cluster_dict = {}
for idx, cl in enumerate(clusters):
    cluster_dict.setdefault(cl, []).append(idx)

base_classes = []
novel_classes = []

similarity_threshold = 0.7

for cl, indices in cluster_dict.items():
    # 取出该簇的类别信息
    cluster_cats = [(categories[i][0], categories[i][1], i) for i in indices]
    # 按实例数降序排序
    cluster_cats_sorted = sorted(cluster_cats, key=lambda x: x[1], reverse=True)

    # 第一个为Base类
    base_class_name, base_class_count, base_idx = cluster_cats_sorted[0]
    base_classes.append(base_class_name)

    # 尝试寻找Novel类
    # 在剩余类别中，寻找与Base类相似度最高的一个，且大于阈值
    best_novel = None
    best_sim = 0.0

    for (c_name, c_cnt, c_idx) in cluster_cats_sorted[1:]:
        sim = cat_synsets[base_idx].wup_similarity(cat_synsets[c_idx])
        if sim is not None and sim > best_sim:
            best_sim = sim
            best_novel = (c_name, c_cnt, c_idx)

    # 如果找到符合相似度要求的Novel类，则添加Novel类
    if best_novel and best_sim >= similarity_threshold:
        novel_classes.append(best_novel[0])
        # 其余未选为Novel的类全部划入Base类
        for (c_name, c_cnt, c_idx) in cluster_cats_sorted[1:]:
            if c_name != best_novel[0]:
                base_classes.append(c_name)
    else:
        # 无符合要求的Novel类，全簇类别都加入Base类（除第一个已在Base_classes外，其余也加入Base）
        for (c_name, c_cnt, c_idx) in cluster_cats_sorted[1:]:
            base_classes.append(c_name)

# 去重
base_classes = list(set(base_classes))
novel_classes = list(set(novel_classes))

# 检查总数是否匹配
all_final_classes = set(base_classes) | set(novel_classes)
if len(all_final_classes) != N:
    print("Warning: The total number of final classes does not match. Check logic.")
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
