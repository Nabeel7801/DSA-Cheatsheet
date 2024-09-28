# Dijkstra’s Algorithm
```
const graph = [];
function dijkstra(start) {
    let distances = {};
    let visited = new Set();
    
    for (let node in graph) {
        distances[node] = Infinity;
    }
    distances[start] = 0;
    
    while (visited.size !== Object.keys(graph).length) {
        let closestNode = null;
        for (let node in distances) {
            if (!visited.has(node)) {
                if (closestNode === null || distances[node] < distances[closestNode]) {
                    closestNode = node;
                }
            }
        }
        
        visited.add(closestNode);
        
        for (let neighbor in graph[closestNode]) {
            let newDist = distances[closestNode] + graph[closestNode][neighbor];
            if (newDist < distances[neighbor]) {
                distances[neighbor] = newDist;
            }
        }
    }
    
    return distances;
}
```

# Floyd-Warshall Algorithm
A dynamic programming algorithm for finding the shortest paths in a weighted graph with Positive or Negative edge weights.
```
function floydWarshall(graph) {
    let dist = [];
    const V = graph.length;
    
    for (let i = 0; i < V; i++) {
        dist[i] = [];
        for (let j = 0; j < V; j++) {
            dist[i][j] = graph[i][j];
        }
    }
    
    for (let k = 0; k < V; k++) {
        for (let i = 0; i < V; i++) {
            for (let j = 0; j < V; j++) {
                dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
    
    return dist;
}
```
