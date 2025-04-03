import numpy as np
import cv2
import math

def validity(p1, p2, maze):
    x1, y1 = p1
    x2, y2 = p2
    pts = max(int(np.hypot(y2-y1, x2-x1)), 1)
    for i in np.linspace(0, 1, pts):
        x = int(round(x1*(1-i) + x2*i))
        y = int(round(y1*(1-i) + y2*i))
        if y < 0 or x < 0 or y >= maze.shape[0] or x >= maze.shape[1]:
            return False
        if maze[y, x] == 255:
            return False
    return True

def generate_nodes(maze, limit, start, end):
    nodes = [start]
    height, width = maze.shape
    while len(nodes) < limit:
        x = np.random.randint(138, 445)
        y = np.random.randint(20, 330)
        if 0 <= y < height and 0 <= x < width and maze[y, x] == 0:
            nodes.append((x, y))
    nodes.append(end)
    return nodes

def dijkstra(nodes, nbrs, cost, start_idx, end_idx):
    n = len(nodes)
    dist = [float('inf')] * n
    dist[start_idx] = 0
    checked = [False] * n
    prev = [-1] * n

    for _ in range(n):
        min_dist = float('inf')
        u = -1
        for i in range(n):
            if not checked[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i
        
        if u == -1 or u == end_idx:
            break
        
        checked[u] = True

        for i in range(min(10, len(nbrs[u]))):
            v = nbrs[u][i]
            if v == -1:
                continue
            if not checked[v] and dist[u] + cost[u][i] < dist[v]:
                dist[v] = dist[u] + cost[u][i]
                prev[v] = u

    if dist[end_idx] == float('inf'):
        return (float('inf'), [])
    
    path = []
    current = end_idx
    while current != -1:
        path.append(current)
        current = prev[current]
    path.reverse()
    return (dist[end_idx], path)

result = (np.inf, [])
while result == (np.inf, []):
   response = requests.get("https://raw.githubusercontent.com/akshatkkaushik/ARK-Perception-Task/refs/heads/main/maze.png", stream=True).raw
image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)

img=cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.imread("maze.png", cv2.IMREAD_GRAYSCALE)
    _, maze = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    maze2 = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
    
    start = (160, 20)
    end = (445, 305)
    
    cv2.circle(maze2, start, 3, (0, 0, 255), -1)
    cv2.circle(maze2, end, 3, (0, 0, 255), -1)
    
    limit = 500  #we have to increase limit if it's not working
    nodes = generate_nodes(maze, limit, start, end)
    nbrs = []
    cost = []
    
    for k in range(len(nodes)):
        nbrs.append([])
        cost.append([])
        
        dis = []
        for j in range(len(nodes)):
            if k != j and validity(nodes[k], nodes[j], maze):
                d = math.dist(nodes[k], nodes[j])
                dis.append((d, j))
        
        dis.sort()
        for d, j in dis[:10]:
            nbrs[k].append(j)
            cost[k].append(d)
            cv2.line(maze2, nodes[k], nodes[j], (153,0,0), 1)
            
        while len(nbrs[k]) < 10:
            nbrs[k].append(-1)
            cost[k].append(float('inf'))
            
        if k % 50 == 0:
            cv2.imshow("MAZE", maze2)
            cv2.waitKey(1)
    
    result = dijkstra(nodes, nbrs, cost, len(nodes)-1, 0)
    print(result)

if result[0] != float('inf'):
    path_points = result[1]
    for i in range(len(path_points)-1):
        cv2.line(maze2, nodes[path_points[i]], nodes[path_points[i+1]], (0,255, 0), 2)
        cv2.imshow("MAZE", maze2)
        cv2.waitKey(1)

cv2.imshow("MAZE", maze2)
cv2.waitKey(0)
cv2.destroyAllWindows()
