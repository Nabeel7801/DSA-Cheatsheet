# Cheat sheet for DSA - Leetcode competitions
Here's a cheat sheet for Data Structures and Algorithms (DSA) that can help you excel in Leetcode competitions. This collection includes essential algorithms such as sorting, searching, dynamic programming, and graph traversal, all with efficient time complexities to tackle competitive programming challenges. Each algorithm is provided with a clear and concise JavaScript implementation, allowing you to quickly refer to them during contests. Whether you're solving problems related to arrays, strings, trees, or graphs, having these ready-to-use snippets can boost your speed and accuracy.


## Sorting
<details>
  <summary>Merge Sort</summary>
    
  ### Merge Sort
  
  An efficient, stable, divide and conquer sorting algorithm.
    
  ```javascript
    function mergeSort(arr) {
        if (arr.length <= 1) return arr;
        
        const mid = Math.floor(arr.length / 2);
        const left = mergeSort(arr.slice(0, mid));
        const right = mergeSort(arr.slice(mid));
        
        return merge(left, right);
    }
    
    function merge(left, right) {
        let result = [];
        let i = 0, j = 0;
        
        while (i < left.length && j < right.length) {
            if (left[i] < right[j]) {
                result.push(left[i++]);
            } else {
                result.push(right[j++]);
            }
        }
        
        return result.concat(left.slice(i)).concat(right.slice(j));
    }
  ```
</details>

<details>
  <summary>Quick Sort</summary>
    
  ### Quick Sort
  
  An algorithm to find the maximum sum of a contiguous subarray.
    
  ```javascript
    function quickSort(arr) {
        if (arr.length <= 1) return arr;
        
        const pivot = arr[arr.length - 1];
        const left = [], right = [];
        
        for (let i = 0; i < arr.length - 1; i++) {
            if (arr[i] < pivot) {
                left.push(arr[i]);
            } else {
                right.push(arr[i]);
            }
        }
        
        return [...quickSort(left), pivot, ...quickSort(right)];
    }
  ```
</details>


## Searching
<details>
  <summary>Binary Search</summary>
    
  ### Binary Search
  
  A divide and conquer algorithm to find the position of an element in a sorted array.
    
  ```javascript
   function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1; // Target not found
  }
  ```
</details>

<details>
  <summary>Depth First Search (DFS)</summary>
    
  ### Depth First Search (DFS)
  
  A graph traversal algorithm that explores as far as possible along each branch before backtracking.
    
  ```javascript
    const graph = [];
    const visited = new Set();
    
    function dfs(index) {
        if (visited.has(index));
        visited.add(index);
        
        graph[index].forEach((neighbor) => {
            if (!visited.has(neighbor)) {
                dfs(neighbor);
            }
        });
    }
  ```
</details>

<details>
  <summary>Breadth First Search (BFS)</summary>
    
  ### Breadth First Search (BFS)
  
  A graph traversal algorithm that explores all the neighbors at the present depth before moving on to nodes at the next depth level.
    
  ```javascript
    const graph = [];
    function bfs(index) {
        let queue = [index];
    
        let visited = new Set();
        visited.add(index);
        
        while (queue.length > 0) {
            let node = queue.shift();
            
            graph[node].forEach((neighbor) => {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    queue.push(neighbor);
                }
            });
        }
    }
  ```
</details>


## Shortest Path
<details> 
  <summary>Dijkstra’s Algorithm</summary>

  ### Dijkstra’s Algorithm

  An algorithm for finding the shortest paths between nodes in a graph, which may represent, for example, road networks.

  ```javascript
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
</details>

<details> 
  <summary>Floyd-Warshall Algorithm (Positive/Negative Edges)</summary>

  ### Floyd-Warshall Algorithm

  A dynamic programming algorithm for finding shortest paths in a weighted graph with positive or negative edge weights.

  ```javascript
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
</details>

<details> 
  <summary>Bellman Ford Algorithm (Path to every other vertex)</summary>

  ### Bellman Ford Algorithm

  Find the shortest paths from a single source vertex to all other vertices in a weighted graph, handling negative weights.

  ```javascript
  function bellmanFord(graph, source) {
    const distances = Array(graph.length).fill(Infinity);
    distances[source] = 0;

    for (let i = 1; i < graph.length - 1; i++) {
        for (const [u, v, weight] of graph) {
            if (distances[u] !== Infinity && distances[u] + weight < distances[v]) {
                distances[v] = distances[u] + weight;
            }
        }
    }

    for (const [u, v, weight] of graph) {
        if (distances[u] !== Infinity && distances[u] + weight < distances[v]) {
            throw new Error("Graph contains a negative-weight cycle");
        }
    }

    return distances;
  }

  ```
</details>


## Strings
<details> 
  <summary>KMP Pattern Matching (Prefix Function)</summary>

  ### KMP Pattern Matching (Prefix Function)

  Efficient string searching algorithm (Knuth-Morris-Pratt). Find the length of the longest proper prefix of the substring which is also a suffix of this substring

  ```javascript
  function kmpPrefixFunction(s) {
      const prefix = Array(s.length).fill(0);
      for (let i = 1, j = 0; i < s.length; i++) {
          while (j > 0 && s[i] !== s[j]) j = prefix[j - 1];
          if (s[i] === s[j]) j++;
          prefix[i] = j;
      }
      return prefix;
  }

  const str = "ababcab"
  kmpPrefixFunction(str);
  ```
</details>

<details>
  <summary>Z Function (Pattern Matching)</summary>
  
  ### Z Function (Pattern Matching)
  
  The Z-function for a string is an array where the value at index `i` is the length of the longest substring starting from `i` that is also a prefix of the string.
  
  ```javascript
  function zFunction(s) {
      const Z = Array(s.length).fill(0);
      let L = 0, R = 0;
      for (let i = 1; i < s.length; i++) {
          if (i <= R) Z[i] = Math.min(R - i + 1, Z[i - L]);
          while (i + Z[i] < s.length && s[Z[i]] === s[i + Z[i]]) Z[i]++;
          if (i + Z[i] - 1 > R) [L, R] = [i, i + Z[i] - 1];
      }
      return Z;
  }

  const pattern = "abc"
  const str = "ababc"
  zFunction(pattern + '#' + str);
  ```
</details>


<details>
  <summary>Manacher's Algorithm (Longest Palindromic substring)</summary>
  
  ### Manacher's Algorithm
  
  This algorithm is used to find the longest palindromic substring in a given string in linear time.
  
  ```javascript
  function manacher(s) {
    const modifiedStr = `#${s.split('').join('#')}#`;
    const n = modifiedStr.length;
    const p = Array(n).fill(0);
    let center = 0, right = 0;

    for (let i = 0; i < n; i++) {
        if (i < right) {
            p[i] = Math.min(right - i, p[2 * center - i]);
        }
        let a = i + (1 + p[i]);
        let b = i - (1 + p[i]);

        while (a < n && b >= 0 && modifiedStr[a] === modifiedStr[b]) {
            p[i]++;
            a++;
            b--;
        }

        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
    }

    let maxLength = 0, centerIndex = 0;
    for (let i = 0; i < n; i++) {
        if (p[i] > maxLength) {
            maxLength = p[i];
            centerIndex = i;
        }
    }

    return s.substring(Math.floor((centerIndex - maxLength) / 2), Math.floor((centerIndex + maxLength) / 2));
  }
  ```
</details>


## Sums
<details>
  <summary>Kadane’s Algorithm (Maximum Subarray Problem)</summary>
    
  ### Kadane’s Algorithm (Maximum Subarray Problem)
  
  An algorithm to find the maximum sum of a contiguous subarray.
    
  ```javascript
     function maxSubArray(arr) {
        let maxCurrent = arr[0];
        let maxGlobal = arr[0];
        
        for (let i = 1; i < arr.length; i++) {
            maxCurrent = Math.max(arr[i], maxCurrent + arr[i]);
            if (maxCurrent > maxGlobal) {
                maxGlobal = maxCurrent;
            }
        }
        
        return maxGlobal;
    }
  ```
</details>

<details> 
  <summary>Sum of Digits</summary>

  ### Sum of Digits

  Finds the sum of digits of a given number.

  ```javascript
  function sumOfDigits(n) {
      return n.toString().split('').reduce((sum, digit) => sum + parseInt(digit), 0);
  }
  ```
</details>


## Math
<details> 
  <summary>Sieve of Eratosthenes (Finding All Primes Up to N)</summary>

  ### Sieve of Eratosthenes (Finding All Primes Up to N)

  An efficient algorithm to find all primes less than or equal to n.

  ```javascript
  function sieveOfEratosthenes(n) {
    const primes = Array(n + 1).fill(true);
    primes[0] = primes[1] = false;
    
    for (let i = 2; i * i <= n; i++) {
        if (primes[i]) {
            for (let j = i * i; j <= n; j += i) {
                primes[j] = false;
            }
        }
    }
    
    return primes.map((isPrime, index) => isPrime ? index : null).filter(Boolean);
  }
  ```
</details>

<details> 
  <summary>Modified Sieve (Finding GCD of all Pairs in Array)</summary>

  ### Modified Sieve (Finding GCD of all Pairs in Array)

  An efficient algorithm to find gcd all pairs in Array.

  ```javascript
  function gcdOfAllPairs(arr) {

    const maxA = Math.max(...arr);
    
    // Frequency array to store the count of each number in the array
    const freq = new Array(maxA + 1).fill(0);
    arr.forEach(num => freq[num]++);

    // Accumulate the counts of multiples of each number
    for (let i = 1; i <= maxA; i++) {
        for (let j = 2 * i; j <= maxA; j += i) {
            freq[i] += freq[j];
        }
    }

    // Calculate the combinations of pairs (choose 2) for each number's count
    for (let i = 1; i <= maxA; i++) {
        freq[i] = (freq[i] + 1) * freq[i] / 2;
    }

    // Remove over-counting by subtracting the sums of multiples
    for (let i = Math.floor(maxA / 2); i >= 1; i--) {
        for (let j = 2 * i; j <= maxA; j += i) {
            freq[i] -= freq[j];
        }
    }

    // Adjust counts to remove the original occurrences of each number
    for (let x of arr) {
        freq[x]--;
    }

    return freq;
}

var gcdValues = function(nums, queries) {
    const pairs = gcdOfAllPairs(nums);

    const n = pairs.length;
    const MAX_NUM = 50000;

    for (let i = 2; i < n; i++) {
        pairs[i] += pairs[i - 1];
    }

    for (let i = n; i <= MAX_NUM; i++) {
        pairs[i] = pairs[i-1];
    }

    // Binary search
    let res = [];
    for (let q of queries) {
        q++;

        let left = 1, right = MAX_NUM;
        let ans = -1;
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            if (pairs[mid] >= q) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        res.push(ans);
    }

    return res;
};
  ```
</details>

<details> 
  <summary>Modular Inverse (Fermat's Little Theorem)</summary>

  ### Modular Inverse (Fermat's Little Theorem)

  Finds the modular inverse of a under modulo m when m is prime.

  ```javascript
  function modInverse(a, m) {
      return modPow(a, m - 2, m); // Using Fermat's Little Theorem
  }
  ```
</details>

<details> 
  <summary>Greatest Common Divisor (GCD)</summary>
  
  ### Greatest Common Divisor (GCD)
  
  The greatest common divisor of two numbers using Euclid's algorithm.
  
  ```javascript
  function gcd(a, b) {
      return b === 0 ? a : gcd(b, a % b);
  }
  ```
</details>

<details> 
  <summary>Least Common Multiple (LCM)</summary>
  
  ### Least Common Multiple (LCM)
  
  The least common multiple of two numbers.
  
  ```javascript
  function lcm(a, b) {
      return (a * b) / gcd(a, b);
  }
  ```
</details> 

<details> 
  <summary>nCr (Combinations)</summary>

  ### nCr (Combinations)

  Calculates the number of combinations (n choose r).

  ```javascript
  function nCr(n, r) {
      if (r > n) return 0;
      let res = 1;
      for (let i = 0; i < r; i++) {
          res *= (n - i);
          res /= (i + 1);
      }
      return res;
  }
  ```
</details>

<details> 
  <summary>Factorial</summary>

  ### Factorial

  Calculates the factorial of a number n recursively.

  ```javascript
  function factorial(n) {
      return n === 0 ? 1 : n * factorial(n - 1);
  }
  ```
</details>

<details> 
  <summary>Prime Check (Simple)</summary>

  ### Prime Check (Simple)

  A basic algorithm to check if a number is prime.

  ```javascript
  function isPrime(n) {
      if (n <= 1) return false;
      for (let i = 2; i * i <= n; i++) {
          if (n % i === 0) return false;
      }
      return true;
  }
  ```
</details>

<details> 
  <summary>Fibonacci Sequence</summary>
  
  ### Fibonacci Sequence
  
  Generates the nth Fibonacci number iteratively.
  
  ```javascript
  function fibonacci(n) {
      if (n <= 1) return n;
      let a = 0, b = 1;
      for (let i = 2; i <= n; i++) {
          [a, b] = [b, a + b];
      }
      return b;
  }
  ```
</details>

<details> 
  <summary>Power Function (Exponentiation by Squaring)</summary>
  
  ### Power Function (Exponentiation by Squaring)

  Efficiently calculates base^exp.

  ```javascript
  function power(base, exp) {
      if (exp === 0) return 1;
      const pow = Math.abs(exp);
      const half = power(base*base, Math.floor(pow / 2));
      const result = half * (exp % 2 === 0 ? 1 : base);
      return exp < 0 ? 1 / result : result;
  }
  ```
</details>

<details> 
  <summary>Binary Exponentiation (Modular)</summary>

  ### Binary Exponentiation (Modular)

  Computes (base^exp) % mod using efficient binary exponentiation.

  ```javascript
  function modPow(base, exp, mod) {
      let result = 1;
      base = base % mod;
      
      while (exp > 0) {
          if (exp % 2 === 1) result = (result * base) % mod;
          exp = Math.floor(exp / 2);
          base = (base * base) % mod;
      }
      
      return result;
  }
  ```
</details>


## Useful Classes
<details> 
  <summary>Union Find</summary>

  ### Union Find

  The Union-Find algorithm, also known as Disjoint Set Union (DSU), is a data structure that keeps track of a partition of a set into disjoint (non-overlapping) subsets

  ```javascript
      class UnionFind {
        constructor(size) {
            // Initialize the parent array, rank and size array
            this.parent = Array.from({ length: size }, (_, index) => index);
            this.rank = Array(size).fill(1);
            this.size = Array(size).fill(1);
        }
    
        // Find method with path compression
        find(p) {
            if (this.parent[p] !== p) {
                this.parent[p] = this.find(this.parent[p]); // Path compression
            }
            return this.parent[p];
        }
    
        // Union method with union by rank
        union(p, q) {
            const rootP = this.find(p);
            const rootQ = this.find(q);
    
            if (rootP === rootQ) return; // They are already in the same set
    
            // Union by rank
            if (this.rank[rootP] > this.rank[rootQ]) {
                this.parent[rootQ] = rootP;
                this.size[rootP] += this.size[rootQ];
    
            } else if (this.rank[rootP] < this.rank[rootQ]) {
                this.parent[rootP] = rootQ;
                this.size[rootQ] += this.size[rootP];
                
            } else {
                this.parent[rootQ] = rootP;
                this.size[rootP] += this.size[rootQ];
                this.rank[rootP] += 1; // Increment rank if they are of the same rank
            }
        }
    
        // Check if two elements are in the same set
        connected(p, q) {
            return this.find(p) === this.find(q);
        }
    
        // Return the maximum size of the set
        largestSetSize() {
            let maxSize = 0;
            for (const i in this.parent) {
                if (this.parent[i] == i) {
                    maxSize = Math.max(maxSize, this.size[i])
                }
            }
            return maxSize;
        }
    }

    // Example usage
    const uf = new UnionFind(10);
    uf.union(1, 2);
    uf.union(2, 3);
    console.log(uf.find(1)); // Output: 3 (or 1, depending on the union operation)
    console.log(uf.connected(1, 3)); // Output: true
    console.log(uf.connected(1, 4)); // Output: false
  ```
</details>

<details> 
  <summary>Trie</summary>

  ### Trie

  The Trie data structure is a tree-like data structure used for storing a dynamic set of strings

  ```javascript
    class TrieNode {
        constructor() {
            this.children = {};
            this.isEndOfWord = false;
        }
    }
    
    class Trie {
        constructor() {
            this.root = new TrieNode();
        }
    
        // Insert a word into the Trie
        insert(word) {
            let currentNode = this.root;
            for (let char of word) {
                if (!currentNode.children[char]) {
                    currentNode.children[char] = new TrieNode();
                }
                currentNode = currentNode.children[char];
            }
            currentNode.isEndOfWord = true;
        }
    
        // Search for a word in the Trie
        search(word) {
            let currentNode = this.root;
            for (let char of word) {
                if (!currentNode.children[char]) {
                    return false;
                }
                currentNode = currentNode.children[char];
            }
            return currentNode.isEndOfWord;
        }
    
        // Check if there is any word in the Trie that starts with the given prefix
        startsWith(prefix) {
            let currentNode = this.root;
            for (let char of prefix) {
                if (!currentNode.children[char]) {
                    return false;
                }
                currentNode = currentNode.children[char];
            }
            return true;
        }

        // Remove a word from the Trie
        remove(word) {
            const removeHelper = (node, word, depth) => {
                if (depth === word.length) {
                    if (!node.isEndOfWord) return false;
                    node.isEndOfWord = false;
    
                    return Object.keys(node.children).length === 0;
                }
    
                const char = word[depth];
                const childNode = node.children[char];
                if (!childNode) return false;
    
                const shouldDeleteChild = removeHelper(childNode, word, depth + 1);
                if (shouldDeleteChild) {
                    delete node.children[char];
    
                    return Object.keys(node.children).length === 0 && !node.isEndOfWord;
                }
    
                return false;
            };
    
            removeHelper(this.root, word, 0);
        }
    }
  ```
</details>

<details> 
  <summary>Priority Queue (Heaps)</summary>

  A priority queue is a data structure where each element has a priority, and elements with higher priority are dequeued before elements with lower priority.

  ### Min Heap

  The element with the lowest priority (smallest value) is always at the root and is dequeued first.

  ```javascript
    class MinHeap {
        constructor() {
            this.heap = [];
        }
    
        // Helper method to swap elements at two indices
        swap(i, j) {
            [this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]];
        }
    
        // Insert a new element into the heap
        insert(val) {
            this.heap.push(val);
            this.bubbleUp();
        }
    
        // Bubble up the last element to maintain the heap property
        bubbleUp() {
            let index = this.heap.length - 1;
            while (index > 0) {
                let parentIndex = Math.floor((index - 1) / 2);
                if (this.heap[parentIndex] <= this.heap[index]) break;  // Parent is smaller, heap property is satisfied
                this.swap(index, parentIndex);
                index = parentIndex;
            }
        }
    
        // Extract the minimum element (root) from the heap
        extractMin() {
            if (this.heap.length === 0) return null;
            if (this.heap.length === 1) return this.heap.pop();
    
            const min = this.heap[0];
            this.heap[0] = this.heap.pop();  // Move the last element to the root
            this.bubbleDown();
            return min;
        }
    
        // Bubble down the root element to maintain the heap property
        bubbleDown() {
            let index = 0;
            const length = this.heap.length;
            const element = this.heap[0];
    
            while (true) {
                let leftChildIndex = 2 * index + 1;
                let rightChildIndex = 2 * index + 2;
                let leftChild, rightChild;
                let swapIndex = null;
    
                if (leftChildIndex < length) {
                    leftChild = this.heap[leftChildIndex];
                    if (leftChild < element) {
                        swapIndex = leftChildIndex;
                    }
                }
    
                if (rightChildIndex < length) {
                    rightChild = this.heap[rightChildIndex];
                    if (
                        (swapIndex === null && rightChild < element) ||
                        (swapIndex !== null && rightChild < leftChild)
                    ) {
                        swapIndex = rightChildIndex;
                    }
                }
    
                if (swapIndex === null) break;  // No more swaps needed
                this.swap(index, swapIndex);
                index = swapIndex;
            }
        }
    
        // Peek at the minimum element (root) without removing it
        peek() {
            return this.heap[0];
        }
    }
  ```

  ### Max Heap

  The element with the highest priority (largest value) is at the root and dequeued first.

  ```javascript
    class MaxHeap {
        constructor() {
            this.heap = [];
        }
    
        // Helper method to swap elements at two indices
        swap(i, j) {
            [this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]];
        }
    
        // Insert a new element into the heap
        insert(val) {
            this.heap.push(val);
            this.bubbleUp();
        }
    
        // Bubble up the last element to maintain the heap property
        bubbleUp() {
            let index = this.heap.length - 1;
            while (index > 0) {
                let parentIndex = Math.floor((index - 1) / 2);
                if (this.heap[parentIndex] >= this.heap[index]) break;  // Parent is larger, heap property is satisfied
                this.swap(index, parentIndex);
                index = parentIndex;
            }
        }
    
        // Extract the maximum element (root) from the heap
        extractMax() {
            if (this.heap.length === 0) return null;
            if (this.heap.length === 1) return this.heap.pop();
    
            const max = this.heap[0];
            this.heap[0] = this.heap.pop();  // Move the last element to the root
            this.bubbleDown();
            return max;
        }
    
        // Bubble down the root element to maintain the heap property
        bubbleDown() {
            let index = 0;
            const length = this.heap.length;
            const element = this.heap[0];
    
            while (true) {
                let leftChildIndex = 2 * index + 1;
                let rightChildIndex = 2 * index + 2;
                let leftChild, rightChild;
                let swapIndex = null;
    
                if (leftChildIndex < length) {
                    leftChild = this.heap[leftChildIndex];
                    if (leftChild > element) {
                        swapIndex = leftChildIndex;
                    }
                }
    
                if (rightChildIndex < length) {
                    rightChild = this.heap[rightChildIndex];
                    if (
                        (swapIndex === null && rightChild > element) ||
                        (swapIndex !== null && rightChild > leftChild)
                    ) {
                        swapIndex = rightChildIndex;
                    }
                }
    
                if (swapIndex === null) break;  // No more swaps needed
                this.swap(index, swapIndex);
                index = swapIndex;
            }
        }
    
        // Peek at the maximum element (root) without removing it
        peek() {
            return this.heap[0];
        }
    }
  ```
</details>
