K核分解，逐层剥离
代码里画的是一个能剥离出三层的无向图，在开局print()的输出中可见
具体图的样子通过运用networkx画出

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

G = nx.Graph()   # 生成一个空的无向图
G.add_edges_from([(1,2),(1,3),(2,3),(2,4),(3,5),(5,6),(5,7),(5,8),(5,9),(6,7),(6,9),(7,8),(7,9),(8,9),(8,10),(9,10),(10,11)])
# 手动画一个无向图，定义节点间的连接，每个元组（a，b）表示a与b之间的边

k_shell = nx.core_number(G)          # 返回每个节点的Ks值
ks_values = list(k_shell.values())
print(k_shell)

# 创建标准化对象
norm = mcolors.Normalize(vmin=min(ks_values), vmax=max(ks_values))
cmap = plt.cm.cool                   # 数值低到高对应蓝到粉渐变
node_colors = [cmap(norm(value)) for value in ks_values]

# 用不同颜色区分Ks值

# 图形布局，seed使每次运行布局一致：模拟弹簧力导向算法排列节点位置
pos = nx.spring_layout(G,seed=42)

nodes = nx.draw_networkx_nodes(
        G,pos,
        node_color=node_colors,      # 按Ks值映射颜色
        node_size=500,               # 节点的大小
        alpha=0.8
        )
nx.draw_networkx_edges(G,pos,width=1,alpha=0.5)      # 绘制边
nx.draw_networkx_labels(G,pos,font_size=12)          # 绘制节点标签

# 创建独立的ScalarMappable:颜色条标准化，确保颜色条范围与Ks值匹配
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(ks_values)

ax = plt.gca()      # 获取当前轴对象，以正确添加颜色条  <-------------------如果不用这个函数，由于matplotlib里的colorbar函数无法自动找到合适的轴来放置颜色条，就会报错
plt.colorbar(sm,ax=ax).set_label('K-Shell Value')    # 要关联到这个绘图轴

plt.title('K-Shell Decomposition Visualization',fontsize=15)
plt.axis('off')     # 隐藏坐标轴
plt.tight_layout()  # 自动调整子图参数 避免重叠
plt.show()


---------------------------运行结果----------------------------
{1: 2, 2: 2, 3: 2, 4: 1, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 2, 11: 1}

我还没有掌握往file里放图片的技能，反正跑出来还是比较清晰可观，赏心悦目的......
